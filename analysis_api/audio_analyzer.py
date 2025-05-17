import datetime
import tempfile
import os
import time
import ffmpeg 
import static_ffmpeg 
import logging
import librosa # Importar Librosa
import numpy as np # Importar NumPy

from pdf_services.services import PdfService

from .interfaces.data_audio_interface import (
    PcmData,
    ExtractedFeatures,
    YamnetResult, # Necesitarás esta si la usas para tipar yamnet_results
    CrepePitchResult, # Necesitarás esta si la usas para tipar crepe_results
    SpeechEmotionResult, # Necesitarás esta si la usas para tipar ser_results
    CombinedAnalysisData,
    FinalAnalysisResponse
)
# Configurar el logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Definición de la constante para la duración mínima de pausa
DEFAULT_MIN_PAUSE_DURATION_MS = 350 
# Definición del umbral de silencio (ajustar experimentalmente)
DEFAULT_SILENCE_THRESHOLD_RMS = 0.005 

class AudioAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ffmpeg_path_to_use = "ffmpeg"
        self.ffprobe_path_to_use = "ffprobe"
        self.pdf_service = PdfService() # Instanciar PdfService
        
        try:
            self.logger.info("AudioAnalyzer_PY_INIT: Intentando añadir FFmpeg y FFprobe de static-ffmpeg al PATH del proceso...")
            static_ffmpeg.add_paths() 
            self.logger.info("AudioAnalyzer_PY_INIT: Rutas de static-ffmpeg deberían estar ahora en os.environ['PATH'].")
        except Exception as e:
            self.logger.error(f"AudioAnalyzer_PY_INIT: Error al llamar a static_ffmpeg.add_paths(): {e}. Se usará 'ffmpeg' y 'ffprobe' esperando que estén en el PATH del sistema.", exc_info=True)
        
        self.logger.info(f"AudioAnalyzer_PY_INIT: Se intentará usar '{self.ffmpeg_path_to_use}' para comandos FFmpeg.")
        self.logger.info("AudioAnalyzer_PY_INIT: Instancia de AudioAnalyzer creada.")

    def _get_temp_filepath(self, original_filename: str, suffix: str = "tmp", extension: str = "") -> str:
        temp_dir = tempfile.gettempdir()
        safe_base_name = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in os.path.splitext(original_filename)[0])
        timestamp = int(time.time() * 1000)
        return os.path.join(temp_dir, f"{safe_base_name}_{timestamp}_{suffix}{extension}")

    def decode_and_prepare_audio(self, input_file_path: str, original_filename: str) -> str | None:
        output_wav_path = self._get_temp_filepath(original_filename, suffix="processed", extension=".wav")
        target_sample_rate = 16000 # 16kHz es común para análisis de voz
        target_audio_channels = 1    # Mono
        
        self.logger.info(f"AudioAnalyzer_PY_DECODE_START: Decodificando: {original_filename}")
        self.logger.info(f"AudioAnalyzer_PY_DECODE_INPUT: {input_file_path}")
        self.logger.info(f"AudioAnalyzer_PY_DECODE_OUTPUT: {output_wav_path}")

        try:
            stream = ffmpeg.input(input_file_path)
            stream = ffmpeg.output(
                stream.audio,
                output_wav_path,
                acodec='pcm_s16le', 
                ar=str(target_sample_rate), # Asegurar que es string para ffmpeg-python
                ac=target_audio_channels
            )
            ffmpeg.run(stream, cmd=self.ffmpeg_path_to_use, 
                       capture_stdout=True, capture_stderr=True, overwrite_output=True)
            
            self.logger.info(f"AudioAnalyzer_PY_DECODE_SUCCESS: Audio decodificado a: {output_wav_path}")
            return output_wav_path
        except ffmpeg.Error as e:
            self.logger.error(f"AudioAnalyzer_PY_DECODE_FFMPEG_ERROR: Error de FFmpeg para {original_filename}.")
            stdout = e.stdout.decode('utf8', errors='ignore') if e.stdout else "N/A"
            stderr = e.stderr.decode('utf8', errors='ignore') if e.stderr else "N/A"
            self.logger.error(f"FFmpeg stdout: {stdout}")
            self.logger.error(f"FFmpeg stderr: {stderr}")
            if any(phrase in stderr.lower() or phrase in stdout.lower() for phrase in ["no such file or directory", "cannot find", "not found"]) or \
               (hasattr(e, 'errno') and e.errno == 2):
                 self.logger.error(f"CRITICAL: FFmpeg ejecutable ('{self.ffmpeg_path_to_use}') NO ENCONTRADO. Asegúrate de que 'static-ffmpeg' funcione o que FFmpeg esté en el PATH del sistema.")
            return None
        except Exception as e_gen:
            self.logger.error(f"AudioAnalyzer_PY_DECODE_GENERAL_ERROR: Error inesperado decodificando {original_filename}: {str(e_gen)}", exc_info=True)
            return None

    def extract_acoustic_features(self, wav_audio_path: str) -> dict | None:
        self.logger.info(f"AudioAnalyzer_PY_FEATURES_START: Extrayendo características de {wav_audio_path}")
        if not os.path.exists(wav_audio_path):
            self.logger.error(f"AudioAnalyzer_PY_FEATURES_ERROR: Archivo WAV no encontrado en {wav_audio_path}")
            return None
        
        try:
            # Cargar el archivo de audio. sr=None preserva el sample rate original.
            # FFmpeg ya lo convirtió a 16000 Hz, pero es bueno ser explícito si es un requisito.
            y, sr = librosa.load(wav_audio_path, sr=16000) 
            
            total_duration_sec = librosa.get_duration(y=y, sr=sr)

            # --- Calcular Energía RMS ---
            # Parámetros para el análisis de frames (puedes ajustarlos)
            frame_length_ms = 25  # Longitud del frame en ms
            hop_length_ms = 10    # Desplazamiento del frame (hop) en ms

            frame_length_samples = int(frame_length_ms / 1000 * sr)
            hop_length_samples = int(hop_length_ms / 1000 * sr)

            rms_frames = librosa.feature.rms(y=y, frame_length=frame_length_samples, hop_length=hop_length_samples)[0]
            average_energy_rms = float(np.mean(rms_frames)) if rms_frames.size > 0 else 0.0

            # --- Detección de Pausas ---
            silence_threshold_rms = DEFAULT_SILENCE_THRESHOLD_RMS # Umbral para considerar silencio
            min_pause_duration_ms = DEFAULT_MIN_PAUSE_DURATION_MS # Duración mínima de una pausa
            
            # Cada frame de RMS representa `hop_length_ms` de tiempo en la secuencia
            frame_duration_sec = hop_length_ms / 1000.0 
            min_pause_frames = int(min_pause_duration_ms / hop_length_ms)

            num_pauses = 0
            current_pause_frames = 0
            total_pause_frames = 0
            
            is_in_pause = False
            for rms_val in rms_frames:
                if rms_val < silence_threshold_rms:
                    current_pause_frames += 1
                    if not is_in_pause and current_pause_frames >= min_pause_frames:
                        is_in_pause = True
                        num_pauses += 1 
                else: # Hay sonido
                    if is_in_pause: # Estaba en pausa y ahora hay sonido
                        total_pause_frames += current_pause_frames
                    is_in_pause = False
                    current_pause_frames = 0
            
            # Si el audio termina en una pausa que cumple la duración mínima
            if is_in_pause: # current_pause_frames ya acumuló los frames de la pausa final
                 total_pause_frames += current_pause_frames
            elif current_pause_frames >= min_pause_frames: # Si terminó en silencio pero no se contó como pausa aún
                num_pauses +=1
                total_pause_frames += current_pause_frames


            total_pause_duration_sec = total_pause_frames * frame_duration_sec
            speaking_duration_sec = total_duration_sec - total_pause_duration_sec
            
            features = {
                "totalDurationSec": round(total_duration_sec, 4),
                "speakingDurationSec": round(max(0, speaking_duration_sec), 4),
                "numberOfPauses": num_pauses,
                "totalPauseDurationSec": round(total_pause_duration_sec, 4),
                "averageEnergyRMS": round(average_energy_rms, 6),
                "minPauseDurationMsUsed": min_pause_duration_ms,
                "notes": "Características acústicas básicas extraídas con Librosa."
            }
            self.logger.info(f"AudioAnalyzer_PY_FEATURES_SUCCESS: Características extraídas: {features}")
            return features

        except Exception as e:
            self.logger.error(f"AudioAnalyzer_PY_FEATURES_ERROR: Error extrayendo características con Librosa: {str(e)}", exc_info=True)
            return None

    def analyze_with_stutter_model(self, model_name: str, data_path_or_features: str | dict) -> dict | None:
        # ... (Placeholder como antes) ...
        self.logger.info(f"AudioAnalyzer_PY_MODEL_START: Analizando con modelo '{model_name}' usando: {type(data_path_or_features)}")
        predictions = {
            "model_name": model_name,
            "probability_score": 0.50 + (hash(str(data_path_or_features)) % 30) / 100.0,
            "detected_segments_sec": [[1.2, 1.5 + time.time() % 1], [3.3, 3.8 + time.time() % 1]],
            "notes": "Implementación real del modelo pendiente."
        }
        self.logger.info(f"AudioAnalyzer_PY_MODEL_SUCCESS: Análisis (placeholder) con '{model_name}' completado.")
        return predictions

    def process_exercise_submission(self, uploaded_file_path: str, original_filename: str, student_id: int | None = None):
        # ... (Lógica como antes, pero ahora `acoustic_features` tendrá valores reales) ...
        self.logger.info(f"AudioAnalyzer_PY_PROCESS_START: Procesando: {original_filename}, Estudiante ID: {student_id}")
        
        processed_wav_path = self.decode_and_prepare_audio(uploaded_file_path, original_filename)
        
        if not processed_wav_path or not os.path.exists(processed_wav_path):
            self.logger.error(f"AudioAnalyzer_PY_PROCESS_FAIL: Fallo en decodificación para {original_filename}. No se puede continuar.")
            try:
                if os.path.exists(uploaded_file_path): os.remove(uploaded_file_path)
            except Exception as e_clean_up:
                self.logger.warn(f"AudioAnalyzer_PY_CLEANUP_WARN: No se pudo borrar archivo temporal subido {uploaded_file_path} tras fallo: {e_clean_up}")
            return {"error": "No se pudo decodificar el archivo de audio/video.", "original_filename": original_filename}

        acoustic_features = self.extract_acoustic_features(processed_wav_path)
        stutternet_results = self.analyze_with_stutter_model("StutterNet", processed_wav_path)
        other_stutter_model_results = self.analyze_with_stutter_model("StutterDetectionApp", processed_wav_path)
        
        # Limpieza del archivo WAV procesado
        try:
            if processed_wav_path and os.path.exists(processed_wav_path):
                os.remove(processed_wav_path)
                self.logger.info(f"AudioAnalyzer_PY_CLEANUP: Archivo WAV temporal borrado: {processed_wav_path}")
        except Exception as e:
            self.logger.warn(f"AudioAnalyzer_PY_CLEANUP_WARN: No se pudo borrar WAV temporal {processed_wav_path}: {e}")
        
        # El archivo original subido (`uploaded_file_path`) debería ser borrado por la vista que lo llamó.

        student_name_for_report = f"Estudiante_{student_id}" if student_id else original_filename.split('.')[0]
        # ---
        yamnet_placeholder = {"topClasses": [], "notes": "Placeholder YAMNet"} # O los resultados reales si los tienes
        crepe_placeholder = {"meanF0Hz": None, "stdDevF0Hz": None, "minF0Hz": None, "maxF0Hz": None, "notes": "Placeholder CREPE"}
        ser_placeholder = {"emotions": [], "notes": "Placeholder SER"}

        combined_data: CombinedAnalysisData = {
            "fileName": original_filename,
            "analysisTimestamp": datetime.datetime.now().isoformat(),
            "basicAcousticFeatures": acoustic_features,
            "yamnetEvents": yamnet_placeholder, # Reemplaza con yamnet_results cuando esté
            "pitchAnalysisF0": crepe_placeholder, # Reemplaza con crepe_results cuando esté
            "emotionAnalysisSER": ser_placeholder, # Reemplaza con ser_results cuando esté
            # Para los modelos de tartamudeo, necesitarías añadirlos a CombinedAnalysisData
            # y pasarlos aquí. Por ahora, no están en la interfaz CombinedAnalysisData.
            # "stutternetAnalysis": stutternet_results, 
            # "otherStutterModelAnalysis": other_stutter_model_results,
            "warnings": [] # O tu variable local_warnings
        }

        pdf_report_url = None
        if self.pdf_service:
            try:
                pdf_report_url = self.pdf_service.generate_analysis_report(
                    combined_data, # Pasa el diccionario/objeto
                    original_filename
                )
                if not pdf_report_url:
                    self.logger.warn("Fallo al generar el informe PDF (desde PdfService).")
            except Exception as e:
                self.logger.error(f"Error crítico llamando a generateAnalysisReport: {e}", exc_info=True)

        final_response = {
            "message": "Análisis completado.",
            "original_filename": original_filename,
            "acoustic_features": acoustic_features,
            "stutternet_analysis": stutternet_results, # O el nombre correcto
            "other_stutter_model_analysis": other_stutter_model_results, # O el nombre correcto
            "pdf_report_url": pdf_report_url,
            "notes": "..."
        }
        return final_response
        
