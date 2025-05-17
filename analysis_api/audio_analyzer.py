import datetime
import tempfile
import os
import time
import ffmpeg 
import static_ffmpeg 
import logging
import librosa # Para carga de audio y análisis de F0
import numpy as np # Para cálculos numéricos con F0

from pathlib import Path

from mediapipe.tasks import python as mediapipe_python
from mediapipe.tasks.python import audio as mediapipe_audio
from mediapipe.tasks.python.components import containers as mediapipe_containers
from yolo_stutter_analysis.services import YoloStutterService

from .interfaces.data_audio_interface import (
    PcmData, ExtractedFeatures, YamnetResult, CrepePitchResult, 
    SpeechEmotionResult, CombinedAnalysisData, FinalAnalysisResponse
)
from pdf_services.services import PdfService


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

DEFAULT_MIN_PAUSE_DURATION_MS = 350 
DEFAULT_SILENCE_THRESHOLD_RMS = 0.005 

# Rangos típicos para F0 en Hz (puedes ajustarlos)
FMIN_HZ = librosa.note_to_hz('C2')  # Aprox. 65 Hz
FMAX_HZ = librosa.note_to_hz('C7')  # Aprox. 2093 Hz

class AudioAnalyzer:
    media_pipe_audio_classifier: mediapipe_audio.AudioClassifier | None = None
    yamnet_init_error: Exception | None = None
    _class_logger = logging.getLogger("AudioAnalyzerClassSetup")

    @classmethod
    def _initialize_yamnet_classifier_globally(cls):
        # ... (como en la versión audio_analyzer_py_v13_final_fixes) ...
        # Asegúrate de que la ruta a yamnet.tflite sea correcta.
        # Por ejemplo, si está en 'assets/models/' en la raíz de tu proyecto:
        # model_path = str(Path(os.getcwd()) / 'assets' / 'models' / 'yamnet.tflite')
        # O si está en la raíz del proyecto:
        if cls.media_pipe_audio_classifier is not None or cls.yamnet_init_error is not None:
            return
        try:
            cls._class_logger.info("AudioAnalyzer_PY_YAMNET_INIT_GLOBAL: Initializing MediaPipe AudioClassifier with YAMNet...")
            model_path = str(Path(os.getcwd()) / 'yamnet.tflite') # Asume que está en la raíz del proyecto
            if not os.path.exists(model_path):
                model_path_alt = str(Path(os.getcwd()) / 'assets' / 'models' / 'yamnet.tflite')
                if os.path.exists(model_path_alt):
                    model_path = model_path_alt
                else:
                    cls._class_logger.error(f"YAMNet model TFLite no encontrado ni en raíz ni en assets/models/. Probadas: {str(Path(os.getcwd()) / 'yamnet.tflite')} y {model_path_alt}")
                    cls.yamnet_init_error = FileNotFoundError(f"YAMNet model not found.")
                    return

            base_options = mediapipe_python.BaseOptions(model_asset_path=model_path)
            options = mediapipe_audio.AudioClassifierOptions(
                base_options=base_options,
                max_results=5 
            )
            cls.media_pipe_audio_classifier = mediapipe_audio.AudioClassifier.create_from_options(options)
            cls._class_logger.info("AudioAnalyzer_PY_YAMNET_INIT_GLOBAL: MediaPipe AudioClassifier (YAMNet) inicializado globalmente.")
        except Exception as e:
            cls._class_logger.error(f"AudioAnalyzer_PY_YAMNET_INIT_GLOBAL: Error inicializando MediaPipe AudioClassifier globalmente: {e}", exc_info=True)
            cls.yamnet_init_error = e


    def __init__(self):
        # ... (como en la versión audio_analyzer_py_v13_final_fixes) ...
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ffmpeg_path_to_use = "ffmpeg"
        self.pdf_service = PdfService()
        AudioAnalyzer._initialize_yamnet_classifier_globally()
        self.yolo_stutter_service = YoloStutterService()
        try:
            static_ffmpeg.add_paths() 
            self.logger.info("AudioAnalyzer_PY_INIT: Rutas de static-ffmpeg deberían estar ahora en os.environ['PATH'].")
        except Exception as e:
            self.logger.error(f"AudioAnalyzer_PY_INIT: Error con static_ffmpeg.add_paths(): {e}. Usando fallback.", exc_info=True)
        
        self.logger.info(f"AudioAnalyzer_PY_INIT: Se intentará usar '{self.ffmpeg_path_to_use}' para FFmpeg.")
        self.logger.info("AudioAnalyzer_PY_INIT: Instancia creada.")

    def _get_temp_filepath(self, original_filename: str, suffix: str = "tmp", extension: str = "") -> str:
        # ... (sin cambios) ...
        temp_dir = tempfile.gettempdir()
        safe_base_name = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in os.path.splitext(original_filename)[0])
        timestamp = int(time.time() * 1000)
        return os.path.join(temp_dir, f"{safe_base_name}_{timestamp}_{suffix}{extension}")

    def decode_and_prepare_audio(self, input_file_path: str, original_filename: str) -> tuple[str | None, int | None]:
        # ... (sin cambios, asegúrate que target_sample_rate sea 16000) ...
        output_wav_path = self._get_temp_filepath(original_filename, suffix="processed", extension=".wav")
        target_sample_rate = 16000 
        target_audio_channels = 1
        
        self.logger.info(f"AudioAnalyzer_PY_DECODE_START: Decodificando: {original_filename}")
        self.logger.info(f"AudioAnalyzer_PY_DECODE_INPUT: {input_file_path}")
        self.logger.info(f"AudioAnalyzer_PY_DECODE_OUTPUT: {output_wav_path}")

        try:
            stream = ffmpeg.input(input_file_path)
            stream = ffmpeg.output(
                stream.audio,
                output_wav_path,
                acodec='pcm_s16le', 
                ar=str(target_sample_rate),
                ac=target_audio_channels
            )
            ffmpeg.run(stream, cmd=self.ffmpeg_path_to_use, 
                       capture_stdout=True, capture_stderr=True, overwrite_output=True)
            
            actual_sample_rate = target_sample_rate 
            self.logger.info(f"AudioAnalyzer_PY_DECODE_SUCCESS: Audio decodificado a: {output_wav_path} con SR: {actual_sample_rate}Hz")
            return output_wav_path, actual_sample_rate
        except ffmpeg.Error as e:
            self.logger.error(f"AudioAnalyzer_PY_DECODE_FFMPEG_ERROR: Error de FFmpeg para {original_filename}.")
            stdout = e.stdout.decode('utf8', errors='ignore') if e.stdout else "N/A"
            stderr = e.stderr.decode('utf8', errors='ignore') if e.stderr else "N/A"
            self.logger.error(f"FFmpeg stdout: {stdout}")
            self.logger.error(f"FFmpeg stderr: {stderr}")
            if any(phrase in stderr.lower() or phrase in stdout.lower() for phrase in ["no such file or directory", "cannot find", "not found"]) or \
               (hasattr(e, 'errno') and e.errno == 2):
                 self.logger.error(f"CRITICAL: FFmpeg ejecutable ('{self.ffmpeg_path_to_use}') NO ENCONTRADO.")
            return None, None
        except Exception as e_gen:
            self.logger.error(f"AudioAnalyzer_PY_DECODE_GENERAL_ERROR: Error inesperado decodificando {original_filename}: {str(e_gen)}", exc_info=True)
            return None, None

    def extract_acoustic_features(self, wav_audio_path: str, sr: int) -> ExtractedFeatures | None:
        # ... (sin cambios, ya usa Librosa) ...
        self.logger.info(f"AudioAnalyzer_PY_FEATURES_START: Extrayendo características de {wav_audio_path} (SR: {sr} Hz)")
        if not os.path.exists(wav_audio_path): return None
        try:
            y, loaded_sr = librosa.load(wav_audio_path, sr=sr) 
            if loaded_sr != sr: self.logger.warn(f"Librosa SR mismatch: {loaded_sr} vs {sr}")
            
            total_duration_sec = librosa.get_duration(y=y, sr=sr)
            frame_length_ms = 25; hop_length_ms = 10
            frame_length_samples = int(frame_length_ms / 1000 * sr)
            hop_length_samples = int(hop_length_ms / 1000 * sr)
            rms_frames = librosa.feature.rms(y=y, frame_length=frame_length_samples, hop_length=hop_length_samples)[0]
            average_energy_rms = float(np.mean(rms_frames)) if rms_frames.size > 0 else 0.0
            
            silence_threshold_rms = DEFAULT_SILENCE_THRESHOLD_RMS
            min_pause_duration_ms = DEFAULT_MIN_PAUSE_DURATION_MS
            frame_duration_sec = hop_length_ms / 1000.0 
            min_pause_frames = int(min_pause_duration_ms / hop_length_ms)
            num_pauses = 0; current_pause_frames = 0; total_pause_frames = 0; is_in_pause = False
            for rms_val in rms_frames:
                if rms_val < silence_threshold_rms:
                    current_pause_frames += 1
                    if not is_in_pause and current_pause_frames >= min_pause_frames:
                        is_in_pause = True; num_pauses += 1 
                else:
                    if is_in_pause: total_pause_frames += current_pause_frames
                    is_in_pause = False; current_pause_frames = 0
            if is_in_pause: total_pause_frames += current_pause_frames
            elif current_pause_frames >= min_pause_frames: num_pauses +=1; total_pause_frames += current_pause_frames
            
            total_pause_duration_sec = total_pause_frames * frame_duration_sec
            speaking_duration_sec = total_duration_sec - total_pause_duration_sec
            
            features: ExtractedFeatures = {
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
            self.logger.error(f"AudioAnalyzer_PY_FEATURES_ERROR: Error extrayendo características: {str(e)}", exc_info=True)
            return None

    def analyze_with_yamnet(self, wav_audio_path: str, sample_rate: int) -> YamnetResult | None:
        # ... (como en la versión audio_analyzer_py_v13_final_fixes) ...
        if AudioAnalyzer.yamnet_init_error:
            self.logger.error(f"AudioAnalyzer_PY_YAMNET_SKIP: YAMNet inicialización global falló: {AudioAnalyzer.yamnet_init_error}")
            return {"topClasses": [], "notes": f"YAMNet model global initialization previously failed: {AudioAnalyzer.yamnet_init_error}"}
        if not AudioAnalyzer.media_pipe_audio_classifier:
            self.logger.warn("AudioAnalyzer_PY_YAMNET_SKIP: YAMNet classifier (global) no disponible.")
            return {"topClasses": [], "notes": "YAMNet model not loaded or failed to initialize globally."}
        
        if not os.path.exists(wav_audio_path):
            self.logger.error(f"AudioAnalyzer_PY_YAMNET_ERROR: Archivo WAV no encontrado en {wav_audio_path}")
            return {"topClasses": [], "notes": f"Archivo no encontrado: {wav_audio_path}"}
        
        if sample_rate != 16000:
            self.logger.warn(f"AudioAnalyzer_PY_YAMNET_SR_WARN: YAMNet requiere 16000Hz, se recibió {sample_rate}Hz.")
            return {"topClasses": [], "notes": f"YAMNet requiere 16000Hz, se recibió {sample_rate}Hz."}

        self.logger.info(f"AudioAnalyzer_PY_YAMNET_START: Analizando con YAMNet (MediaPipe): {wav_audio_path}")
        try:
            y_yamnet, sr_yamnet_loaded = librosa.load(wav_audio_path, sr=sample_rate) # Cargar para AudioData
            if sr_yamnet_loaded != 16000:
                 self.logger.error(f"YAMNet Error: Librosa cargó con SR={sr_yamnet_loaded} en vez de 16000Hz.")
                 return {"topClasses": [], "notes": f"Sample rate incorrecto para YAMNet: {sr_yamnet_loaded}Hz."}

            audio_clip_data = mediapipe_containers.AudioData.create_from_array(
                y_yamnet, sr_yamnet_loaded 
            )
            
            classification_result_list = AudioAnalyzer.media_pipe_audio_classifier.classify(audio_clip_data)
            
            if classification_result_list and len(classification_result_list) > 0:
                result_to_process = classification_result_list[0] if isinstance(classification_result_list, list) else classification_result_list
                if result_to_process.classifications and \
                   len(result_to_process.classifications) > 0 and \
                   result_to_process.classifications[0].categories:
                    
                    categories = result_to_process.classifications[0].categories
                    top_classes = sorted(
                        [{"className": cat.category_name, "score": round(cat.score, 4)} for cat in categories if cat.category_name],
                        key=lambda x: x["score"],
                        reverse=True
                    )[:5]
                    self.logger.info(f"AudioAnalyzer_PY_YAMNET_SUCCESS: Top 5 clases de YAMNet: {top_classes}")
                    return {"topClasses": top_classes}
                else:
                    self.logger.warn("AudioAnalyzer_PY_YAMNET_WARN: YAMNet devolvió una estructura de clasificaciones vacía o inesperada.")
                    return {"topClasses": [], "notes": "YAMNet no devolvió categorías en la estructura esperada."}
            else:
                self.logger.warn("AudioAnalyzer_PY_YAMNET_WARN: YAMNet no devolvió resultados de clasificación.")
                return {"topClasses": [], "notes": "YAMNet no devolvió resultados."}
        except Exception as e:
            self.logger.error(f"AudioAnalyzer_PY_YAMNET_ERROR: Error durante análisis YAMNet: {str(e)}", exc_info=True)
            return {"topClasses": [], "notes": f"Error en YAMNet: {str(e)}"}

    def run_crepe_pitch_analysis(self, wav_audio_path: str, sr: int) -> CrepePitchResult | None:
        self.logger.info(f"AudioAnalyzer_PY_PITCH_START: Analizando Tono (F0) con Librosa para {wav_audio_path}")
        if not os.path.exists(wav_audio_path):
            self.logger.error(f"AudioAnalyzer_PY_PITCH_ERROR: Archivo no encontrado: {wav_audio_path}")
            return {"meanF0Hz": None, "stdDevF0Hz": None, "minF0Hz": None, "maxF0Hz": None, "notes": "Archivo de audio no encontrado."}
        
        try:
            y, loaded_sr = librosa.load(wav_audio_path, sr=sr) # sr debería ser 16000Hz
            if loaded_sr != sr:
                self.logger.warn(f"PITCH_SR_WARN: Sample rate cargado ({loaded_sr}Hz) difiere del esperado ({sr}Hz).")

            # Estimar F0 usando pYIN
            f0, voiced_flag, voiced_prob = librosa.pyin(
                y,
                fmin=FMIN_HZ, # Usar constantes definidas globalmente
                fmax=FMAX_HZ,
                sr=sr,
                frame_length=1024, # Común para pYIN, ajusta si es necesario
                hop_length=512   # Común para pYIN
            )
            
            # Filtrar F0 donde hay sonoridad (voiced_flag es True o voiced_prob > umbral)
            # voiced_prob es a menudo más útil que voiced_flag directamente
            confident_f0 = f0[voiced_prob > 0.6] # Umbral de confianza, puedes ajustarlo
            # Eliminar NaNs que pYIN puede devolver para segmentos no sonoros
            confident_f0 = confident_f0[~np.isnan(confident_f0)]

            if confident_f0.size > 0:
                mean_f0 = float(np.mean(confident_f0))
                std_dev_f0 = float(np.std(confident_f0))
                min_f0 = float(np.min(confident_f0))
                max_f0 = float(np.max(confident_f0))
                
                # Opcional: crear un contorno de F0 simplificado
                # times = librosa.times_like(f0, sr=sr, hop_length=512)
                # f0_contour_data = [{"timeSec": round(t,3) , "frequencyHz": round(f,1) if not np.isnan(f) else 0, "confidence": round(vp,2)} 
                #                    for t, f, vp in zip(times, f0, voiced_prob)][::5] # Tomar cada 5ta muestra

                self.logger.info(f"AudioAnalyzer_PY_PITCH_SUCCESS: Análisis F0 completado. Mean F0: {mean_f0:.2f} Hz")
                return {
                    "meanF0Hz": round(mean_f0, 2),
                    "stdDevF0Hz": round(std_dev_f0, 2),
                    "minF0Hz": round(min_f0, 2),
                    "maxF0Hz": round(max_f0, 2),
                    # "f0Contour": f0_contour_data[:100], # Limitar el tamaño del contorno en la respuesta
                    "notes": "Análisis de F0 realizado con Librosa (pYIN)."
                }
            else:
                self.logger.warn("AudioAnalyzer_PY_PITCH_WARN: No se detectaron segmentos sonoros con F0 confiable.")
                return {"meanF0Hz": None, "stdDevF0Hz": None, "minF0Hz": None, "maxF0Hz": None, "notes": "No se detectó F0 confiable."}

        except Exception as e:
            self.logger.error(f"AudioAnalyzer_PY_PITCH_ERROR: Error durante análisis de F0 con Librosa: {str(e)}", exc_info=True)
            return {"meanF0Hz": None, "stdDevF0Hz": None, "minF0Hz": None, "maxF0Hz": None, "notes": f"Error en análisis F0: {str(e)}"}
        
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

    def run_speech_emotion_recognition(self, wav_audio_path: str, sr: int) -> SpeechEmotionResult | None:
        # ... (Placeholder como antes) ...
        self.logger.warn(f"AudioAnalyzer_PY_SER_PLACEHOLDER: SER no implementado para {wav_audio_path}. Devolviendo datos placeholder.")
        if not os.path.exists(wav_audio_path): return {"emotions": [], "notes": "Archivo no encontrado."}
        emotions_list = ["neutral", "happy", "sad", "angry", "fearful"]
        selected_emotion = emotions_list[int(time.time()) % len(emotions_list)]
        return {
            "emotions": [{"emotion": selected_emotion, "score": round(0.6 + (time.time() % 30)/100, 3)}],
            "notes": "Placeholder SER data. Integración real de modelo SER pendiente."
        }
            
    def process_exercise_submission(
        self, uploaded_file_path: str, original_filename: str, student_id: int | None = None
    ) -> FinalAnalysisResponse:
        self.logger.info(f"AudioAnalyzer_PY_PROCESS_START: Procesando: {original_filename}, Estudiante ID: {student_id}")
        
        warnings: list[str] = []
        # ... (inicialización de variables de resultados como antes) ...
        acoustic_features: ExtractedFeatures | None = None
        yamnet_results_val: YamnetResult | None = None
        crepe_results_val: CrepePitchResult | None = None
        ser_results_val: SpeechEmotionResult | None = None
        yolo_stutter_analysis_results: dict | None = None # Para YOLO-Stutter
        # ... otros placeholders de modelos de tartamudeo basados en audio ...
        stutternet_results: dict | None = None
        other_stutter_model_results: dict | None = None
        pdf_report_url: str | None = None
        
        current_timestamp = datetime.datetime.now().isoformat()

        processed_wav_path: str | None = None
        sample_rate_val: int | None = None

        # Asumimos que uploaded_file_path es la ruta al VIDEO ORIGINAL
        # Primero, decodificamos el audio del video para los análisis de audio
        decoded_info = self.decode_and_prepare_audio(uploaded_file_path, original_filename)
        if decoded_info and decoded_info[0] and os.path.exists(decoded_info[0]):
            processed_wav_path, sample_rate_val = decoded_info
        else:
            # ... (manejo de error de decodificación como antes, devolver respuesta de error) ...
            warnings.append("Fallo en la decodificación del audio. Análisis detallado no posible.")
            error_response_data: FinalAnalysisResponse = {
                "fileName": original_filename, "analysisTimestamp": current_timestamp,
                "basicAcousticFeatures": None, "yamnetEvents": None, "pitchAnalysisF0": None,
                "emotionAnalysisSER": None, "warnings": warnings, "pdfReportUrl": None,
                "yolo_stutter_analysis": None, # Añadir al error response
                "stutternet_analysis": None, "other_stutter_model_analysis": None,
                "message": "Fallo en la decodificación del audio.", "notes": "El análisis no pudo continuar."
            }
            # Limpiar el archivo subido original si la decodificación falla
            try:
                if os.path.exists(uploaded_file_path): os.remove(uploaded_file_path)
            except Exception as e_clean_up:
                self.logger.warn(f"CLEANUP_WARN: No se pudo borrar {uploaded_file_path} tras fallo: {e_clean_up}")
            return error_response_data # type: ignore

        # Análisis que usan el audio procesado (WAV)
        if processed_wav_path and sample_rate_val:
            acoustic_features = self.extract_acoustic_features(processed_wav_path, sample_rate_val)
            yamnet_results_val = self.analyze_with_yamnet(processed_wav_path, sample_rate_val)
            crepe_results_val = self.run_crepe_pitch_analysis(processed_wav_path, sample_rate_val)
            ser_results_val = self.run_speech_emotion_recognition(processed_wav_path, sample_rate_val)
            
            # Modelos de tartamudeo basados en audio (placeholders)
            stutternet_results = self.analyze_with_stutter_model("StutterNet", processed_wav_path)
            other_stutter_model_results = self.analyze_with_stutter_model("StutterDetectionApp", processed_wav_path)

        # Análisis YOLO-Stutter (usa el archivo de video original subido)
        if self.yolo_stutter_service: # Verificar si el servicio está disponible
            # Asumimos que uploaded_file_path es la ruta al video
            yolo_stutter_analysis_results = self.yolo_stutter_service.analyze_video_for_stuttering(uploaded_file_path)
            if not yolo_stutter_analysis_results or yolo_stutter_analysis_results.get("error"):
                warnings.append(f"Análisis YOLO-Stutter falló o devolvió error: {yolo_stutter_analysis_results.get('error', 'Error desconocido') if yolo_stutter_analysis_results else 'Servicio no devolvió resultado'}")
        else:
            warnings.append("YoloStutterService no está disponible.")
        
        # Limpieza del archivo WAV procesado
        try:
            if processed_wav_path and os.path.exists(processed_wav_path):
                os.remove(processed_wav_path)
                self.logger.info(f"AudioAnalyzer_PY_CLEANUP: Archivo WAV temporal borrado: {processed_wav_path}")
        except Exception as e:
            self.logger.warn(f"AudioAnalyzer_PY_CLEANUP_WARN: No se pudo borrar WAV temporal {processed_wav_path}: {e}")

        # El archivo original (`uploaded_file_path`) será borrado por la vista de Django
        # después de que esta función de servicio haya completado.

        combined_data: CombinedAnalysisData = {
            "fileName": original_filename,
            "analysisTimestamp": current_timestamp,
            "basicAcousticFeatures": acoustic_features,
            "yamnetEvents": yamnet_results_val,
            "pitchAnalysisF0": crepe_results_val,
            "emotionAnalysisSER": ser_results_val,
            # Necesitas añadir yolo_stutter_analysis a tu TypedDict CombinedAnalysisData
            # "yoloStutterAnalysis": yolo_stutter_analysis_results, 
            "warnings": warnings if len(warnings) > 0 else None,
        }

        if self.pdf_service:
            # ... (lógica de generación de PDF como antes, asegúrate de pasar combined_data) ...
            try:
                pdf_report_url = self.pdf_service.generate_analysis_report(combined_data, original_filename) # type: ignore
                if not pdf_report_url: 
                    # ... (manejo de warnings) ...
                    current_warnings = combined_data.get("warnings", []) or [] 
                    current_warnings.append('Fallo al generar o guardar el informe PDF (desde PdfService).')
                    combined_data["warnings"] = current_warnings # Actualizar combined_data
            except Exception as e:
                # ... (manejo de error y warnings) ...
                self.logger.error(f"Error llamando a generateAnalysisReport: {e}", exc_info=True)
                current_warnings = combined_data.get("warnings", []) or [] 
                current_warnings.append(f"Fallo crítico en PDF: {str(e)}")
                combined_data["warnings"] = current_warnings

        final_response_data: dict = {
            "fileName": combined_data["fileName"],
            "analysisTimestamp": combined_data["analysisTimestamp"],
            "basicAcousticFeatures": combined_data["basicAcousticFeatures"],
            "yamnetEvents": combined_data["yamnetEvents"],
            "pitchAnalysisF0": combined_data["pitchAnalysisF0"],
            "emotionAnalysisSER": combined_data["emotionAnalysisSER"],
            "yolo_stutter_analysis": yolo_stutter_analysis_results, # Añadir aquí
            "stutternet_analysis": stutternet_results,
            "other_stutter_model_analysis": other_stutter_model_results,
            "warnings": combined_data.get("warnings"), 
            "pdfReportUrl": pdf_report_url,
            "message": "Análisis completado (con placeholders para modelos específicos).",
            "notes": "Revisar logs. Implementación de modelos de tartamudeo y SER pendiente."
        }
        return final_response_data # type: ignore