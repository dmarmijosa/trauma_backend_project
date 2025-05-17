from django.shortcuts import render
from rest_framework import viewsets
from .models import Student
from .serializers import StudentSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser # Para subida de archivos
from rest_framework import status
from django.core.files.storage import FileSystemStorage
import os
import tempfile
import logging

from .audio_analyzer import AudioAnalyzer # Importa tu clase

logger = logging.getLogger(__name__)
class ExerciseAnalysisView(APIView):
    parser_classes = (MultiPartParser, FormParser) # Permite subida de archivos

    def post(self, request, *args, **kwargs):
        logger.info("API_POST_ANALYZE_EXERCISE: Petición POST recibida en /api/v1/analysis/process-exercise/")
        
        file_obj = request.data.get('exerciseFile') # 'exerciseFile' será el nombre del campo en el form-data

        if not file_obj:
            logger.warn("API_POST_ANALYZE_EXERCISE_NO_FILE: No se encontró 'exerciseFile' en la petición.")
            return Response(
                {"error": "No se proporcionó ningún archivo con la clave 'exerciseFile'."},
                status=status.HTTP_400_BAD_REQUEST
            )

        original_filename = file_obj.name
        logger.info(f"API_POST_ANALYZE_EXERCISE_FILE_RECEIVED: Archivo recibido: {original_filename}, Tamaño: {file_obj.size}")

        # Guardar el archivo temporalmente para que FFmpeg pueda acceder a él por ruta
        # Django guarda los archivos subidos en memoria o disco dependiendo del tamaño.
        # Para pasarlo a FFmpeg, es más fácil tenerlo en una ruta de archivo.
        
        # Usar FileSystemStorage para guardarlo en un lugar temporal conocido
        fs = FileSystemStorage(location=tempfile.gettempdir())
        temp_file_name = fs.save(original_filename, file_obj) # Guarda y devuelve un nombre único si ya existe
        uploaded_file_path = fs.path(temp_file_name)
        
        logger.info(f"API_POST_ANALYZE_EXERCISE_TEMP_SAVE: Archivo guardado temporalmente en: {uploaded_file_path}")

        analyzer = AudioAnalyzer() # Crear una instancia de tu analizador
        analysis_result = analyzer.process_exercise_submission(uploaded_file_path, original_filename)

        # Limpiar el archivo subido temporalmente DESPUÉS de que FFmpeg lo haya usado
        # (FFmpeg crea su propio archivo de salida, el procesado)
        try:
            if fs.exists(temp_file_name): # Verificar si aún existe (podría haber sido movido o renombrado por fs.save)
                fs.delete(temp_file_name) # Borrar el archivo subido original temporalmente
                logger.info(f"API_POST_ANALYZE_EXERCISE_TEMP_DELETE: Archivo temporal subido borrado: {uploaded_file_path}")
        except Exception as e:
            logger.error(f"API_POST_ANALYZE_EXERCISE_TEMP_DELETE_FAIL: No se pudo borrar el archivo temporal {uploaded_file_path}: {e}")


        if "error" in analysis_result:
            return Response(analysis_result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return Response(analysis_result, status=status.HTTP_200_OK)


class StudentViewSet(viewsets.ModelViewSet):
    queryset = Student.objects.all()
    serializer_class = StudentSerializer
    # Aquí añadiremos permisos más adelante

