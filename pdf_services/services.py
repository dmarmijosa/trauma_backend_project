import logging
import os
from pathlib import Path # Usar pathlib para un mejor manejo de rutas
import time
import datetime # Para formatear el timestamp

# Importaciones de ReportLab
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# Importar las interfaces/tipos de datos que usará este servicio
# Ajusta la ruta según la ubicación real de tus definiciones de interfaz/dataclass
# Si CombinedAnalysisData y otras están definidas en audio_analyzer.py o en un archivo de interfaces:
# from ..analysis_api.audio_analyzer import CombinedAnalysisData # Ejemplo si estuviera en audio_analyzer.py
# O si tienes un archivo de interfaces central:
from analysis_api.interfaces.data_audio_interface import CombinedAnalysisData # Nota: Para Python, en lugar de interfaces, usarías dataclasses o Pydantic para definir estas estructuras
# y luego importarías esas clases. Por ahora, asumiré que CombinedAnalysisData es un diccionario
# bien estructurado como lo hemos estado manejando.

# Si tienes esta constante definida en otro lugar y la necesitas aquí:
DEFAULT_MIN_PAUSE_DURATION_MS = 350

# Configuración del logger para este servicio
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO) # Esto ya debería estar configurado en audio_analyzer o settings.py

class PdfService:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        # ReportLab no necesita una instancia de "printer" como pdfmake.
        # Las fuentes se manejan de forma diferente si necesitas fuentes personalizadas no estándar,
        # pero para las básicas (Helvetica, Times-Roman, Courier) no se necesita setup.
        # Si quieres Roboto, necesitarías registrar la fuente con ReportLab.
        self.logger.info("PdfService (Python/ReportLab) initialized.")

    def _get_unique_filepath_and_url(self, original_filename: str, output_directory_name: str = "reports") -> tuple[Path, str]:
        """Genera una ruta de archivo única para el PDF y la URL pública relativa."""
        from django.conf import settings # Importar dentro del método para evitar problemas de carga temprana

        safe_base_name = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in os.path.splitext(original_filename)[0])
        timestamp = int(time.time())
        pdf_filename = f"analisis_{safe_base_name}_{timestamp}.pdf"
        
        # Construir ruta de guardado dentro del directorio de media de Django
        # MEDIA_ROOT debe estar definido en settings.py (ej: BASE_DIR / 'public' / 'media')
        # MEDIA_URL debe estar definido en settings.py (ej: '/media/')
        media_root = Path(settings.MEDIA_ROOT if hasattr(settings, 'MEDIA_ROOT') else Path(os.getcwd()) / "media_files_placeholder")
        
        save_directory = media_root / output_directory_name
        save_directory.mkdir(parents=True, exist_ok=True) # Asegurar que el directorio exista
        
        full_pdf_path = save_directory / pdf_filename
        
        # Construir la URL pública
        media_url = settings.MEDIA_URL if hasattr(settings, 'MEDIA_URL') else "/media/"
        public_url = f"{media_url.rstrip('/')}/{output_directory_name}/{pdf_filename}"

        return full_pdf_path, public_url

    def generate_analysis_report(
        self,
        analysis_data: CombinedAnalysisData, # Espera un diccionario o un objeto Dataclass/Pydantic
        original_filename: str,
    ) -> str | None: # Devuelve la URL pública del PDF o None si falla

        full_pdf_path, public_pdf_url = self._get_unique_filepath_and_url(original_filename)
        self.logger.info(f"PdfService: Generando PDF para: {original_filename} en {full_pdf_path}")

        try:
            doc = SimpleDocTemplate(str(full_pdf_path), pagesize=letter)
            styles = getSampleStyleSheet()
            
            # Estilos personalizados
            styles.add(ParagraphStyle(name='Justify', alignment=4)) # 4 = TA_JUSTIFY
            styles.add(ParagraphStyle(name='ReportTitle', parent=styles['h1'], alignment=1, spaceAfter=16)) # 1 = TA_CENTER
            styles.add(ParagraphStyle(name='Subheader', parent=styles['Normal'], alignment=1, spaceAfter=12, textColor=colors.HexColor("#555555")))
            styles.add(ParagraphStyle(name='SectionHeader', parent=styles['h2'], spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#333366")))
            styles.add(ParagraphStyle(name='ListItem', parent=styles['Normal'], leftIndent=0.25*inch, spaceBefore=3))
            styles.add(ParagraphStyle(name='NotesStyle', parent=styles['Italic'], fontSize=9, textColor=colors.grey))


            story = []

            # Título y Metadatos
            story.append(Paragraph("Informe de Análisis de Voz y Lectura", styles['ReportTitle']))
            story.append(Paragraph(f"Archivo Analizado: {analysis_data.get('fileName', original_filename)}", styles['Subheader']))
            timestamp_str = analysis_data.get('analysisTimestamp', datetime.datetime.now().isoformat())
            try:
                # Intentar parsear y formatear si es un string ISO
                formatted_timestamp = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")).strftime('%d de %B de %Y, %H:%M:%S %Z')
            except:
                formatted_timestamp = timestamp_str # Usar como está si falla el parseo
            story.append(Paragraph(f"Fecha y Hora del Análisis: {formatted_timestamp}", styles['Subheader']))
            story.append(Spacer(1, 0.3*inch))

            # --- Sección de Características Acústicas Básicas ---
            acoustic_features = analysis_data.get('basicAcousticFeatures')
            story.append(Paragraph("1. Características Acústicas Básicas", styles['SectionHeader']))
            if acoustic_features:
                # Definir los datos para la tabla
                data = [
                    [Paragraph("<b>Métrica</b>", styles['Normal']), Paragraph("<b>Valor</b>", styles['Normal'])],
                    ["Duración Total del Audio:", f"{acoustic_features.get('totalDurationSec', 0):.2f} segundos"],
                    ["Duración Estimada del Habla:", f"{acoustic_features.get('speakingDurationSec', 0):.2f} segundos"],
                    ["Número de Pausas (silencios > {0}ms):".format(acoustic_features.get('minPauseDurationMsUsed', DEFAULT_MIN_PAUSE_DURATION_MS)), 
                     str(acoustic_features.get('numberOfPauses', 0))],
                    ["Duración Total de Pausas:", f"{acoustic_features.get('totalPauseDurationSec', 0):.2f} segundos"],
                    ["Energía Promedio (RMS General):", f"{acoustic_features.get('averageEnergyRMS', 0):.5f}"],
                ]
                # Crear la tabla
                table = Table(data, colWidths=[3*inch, 2.5*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
                    ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                    ('ALIGN',(0,0),(-1,-1),'LEFT'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), # ReportLab usa Helvetica por defecto
                    ('FONTSIZE', (0,0), (-1,0), 10),
                    ('BOTTOMPADDING', (0,0), (-1,0), 10),
                    ('BACKGROUND',(0,1),(-1,-1),colors.lightblue),
                    ('TEXTCOLOR',(0,1),(-1,-1),colors.black),
                    ('GRID',(0,0),(-1,-1),0.5,colors.black),
                    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                ]))
                story.append(table)
                story.append(Spacer(1, 0.2*inch))
                if acoustic_features.get('notes'):
                     story.append(Paragraph(f"<i>Notas: {acoustic_features['notes']}</i>", styles['NotesStyle']))
            else:
                story.append(Paragraph("No se pudieron extraer características acústicas básicas.", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # --- Sección YAMNet (Eventos de Audio) ---
            yamnet_events = analysis_data.get('yamnetEvents')
            story.append(Paragraph("2. Análisis de Eventos Sonoros (YAMNet)", styles['SectionHeader']))
            if yamnet_events and yamnet_events.get('topClasses') and len(yamnet_events['topClasses']) > 0:
                story.append(Paragraph("Principales eventos de audio detectados:", styles['Normal']))
                for yc in yamnet_events['topClasses']:
                    story.append(Paragraph(f"- {yc.get('className', 'N/A')}: (Puntuación: {yc.get('score', 0):.3f})", styles['ListItem']))
            else:
                story.append(Paragraph(yamnet_events.get('notes', 'No se detectaron eventos principales o el análisis YAMNet falló.'), styles['Normal']))
            story.append(Spacer(1, 0.3*inch))

            # --- Sección Análisis de Tono (F0 - Pitch con CREPE Placeholder) ---
            pitch_analysis = analysis_data.get('pitchAnalysisF0')
            story.append(Paragraph("3. Análisis de Tono (F0 - Pitch)", styles['SectionHeader']))
            if pitch_analysis:
                data_pitch = [
                    [Paragraph("<b>Métrica de Tono</b>", styles['Normal']), Paragraph("<b>Valor</b>", styles['Normal'])],
                    ["Tono Promedio (F0):", f"{pitch_analysis.get('meanF0Hz', 'N/A'):.1f} Hz" if pitch_analysis.get('meanF0Hz') is not None else "N/A"],
                    ["Desviación Estándar F0:", f"{pitch_analysis.get('stdDevF0Hz', 'N/A'):.1f} Hz" if pitch_analysis.get('stdDevF0Hz') is not None else "N/A"],
                    ["Tono Mínimo F0:", f"{pitch_analysis.get('minF0Hz', 'N/A'):.1f} Hz" if pitch_analysis.get('minF0Hz') is not None else "N/A"],
                    ["Tono Máximo F0:", f"{pitch_analysis.get('maxF0Hz', 'N/A'):.1f} Hz" if pitch_analysis.get('maxF0Hz') is not None else "N/A"],
                ]
                table_pitch = Table(data_pitch, colWidths=[3*inch, 2.5*inch])
                table_pitch.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.teal),
                    ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                    ('ALIGN',(0,0),(-1,-1),'LEFT'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,0), 10),
                    ('BOTTOMPADDING', (0,0), (-1,0), 10),
                    ('BACKGROUND',(0,1),(-1,-1),colors.paleturquoise),
                    ('GRID',(0,0),(-1,-1),0.5,colors.black),
                    ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                ]))
                story.append(table_pitch)
                story.append(Spacer(1, 0.2*inch))
                if pitch_analysis.get('notes'):
                    story.append(Paragraph(f"<i>Notas: {pitch_analysis['notes']}</i>", styles['NotesStyle']))
            else:
                story.append(Paragraph("Análisis de tono no disponible.", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))

            # --- Sección de Modelos de Tartamudeo (Placeholders) ---
            # StutterNet
            stutternet_data = analysis_data.get('stutternet_analysis') # Nombre que usaste en la respuesta JSON
            story.append(Paragraph("4. Análisis Modelo 'StutterNet' (Placeholder)", styles['SectionHeader']))
            if stutternet_data:
                story.append(Paragraph(f"Puntuación de Probabilidad: {stutternet_data.get('probability_score', 'N/A'):.2f}", styles['Normal']))
                if stutternet_data.get('detected_segments_sec'):
                    story.append(Paragraph("Segmentos Detectados (segundos):", styles['Normal']))
                    for seg in stutternet_data['detected_segments_sec']:
                        story.append(Paragraph(f"- De {seg[0]:.2f} a {seg[1]:.2f}", styles['ListItem']))
                if stutternet_data.get('notes'):
                     story.append(Paragraph(f"<i>Notas: {stutternet_data['notes']}</i>", styles['NotesStyle']))
            else:
                story.append(Paragraph("Análisis con StutterNet no disponible.", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))

            # Other Stutter Model
            other_model_data = analysis_data.get('other_stutter_model_analysis') # Nombre que usaste
            story.append(Paragraph("5. Análisis Modelo 'StutterDetectionApp' (Placeholder)", styles['SectionHeader']))
            if other_model_data:
                story.append(Paragraph(f"Puntuación de Probabilidad: {other_model_data.get('probability_score', 'N/A'):.2f}", styles['Normal']))
                # ... (similar para segmentos si los hay) ...
                if other_model_data.get('notes'):
                     story.append(Paragraph(f"<i>Notas: {other_model_data['notes']}</i>", styles['NotesStyle']))
            else:
                story.append(Paragraph("Análisis con StutterDetectionApp no disponible.", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # --- Sección de Emoción (SER Placeholder) ---
            emotion_data = analysis_data.get('emotionAnalysisSER')
            story.append(Paragraph("6. Análisis de Emoción en la Voz (SER - Placeholder)", styles['SectionHeader']))
            if emotion_data:
                if emotion_data.get('emotions') and len(emotion_data['emotions']) > 0:
                    story.append(Paragraph("Emociones detectadas (principales):", styles['Normal']))
                    for em in emotion_data['emotions']:
                        story.append(Paragraph(f"- {em.get('emotion', 'N/A')}: (Puntuación: {em.get('score', 0):.3f})", styles['ListItem']))
                if emotion_data.get('notes'):
                    story.append(Paragraph(f"<i>Notas: {emotion_data['notes']}</i>", styles['NotesStyle']))
            else:
                story.append(Paragraph("Análisis de emoción no disponible.", styles['Normal']))


            # Advertencias
            if analysis_data.get('warnings') and len(analysis_data['warnings']) > 0:
                story.append(Spacer(1, 0.3*inch))
                story.append(Paragraph("Advertencias Durante el Análisis", style_name='SectionHeader', textColor=colors.orange))
                for warning_text in analysis_data['warnings']:
                    story.append(Paragraph(f"- {warning_text}", styles['ListItem']))

            # Pie de página (opcional)
            # def myLaterPages(canvas, doc):
            #     canvas.saveState()
            #     canvas.setFont('Times-Roman', 9)
            #     canvas.drawString(inch, 0.75 * inch, "Página %d" % doc.page)
            #     canvas.restoreState()
            # doc.build(story, onLaterPages=myLaterPages)
            
            doc.build(story)
            self.logger.info(f"PdfService: PDF generado exitosamente y guardado en: {full_pdf_path}")
            return public_pdf_url

        except Exception as e:
            self.logger.error(f"PdfService: Error crítico generando PDF para {original_filename}: {str(e)}", exc_info=True)
            return None