from django.urls import path, include
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import StudentViewSet, ExerciseAnalysisView # Añadir ExerciseAnalysisView

router = DefaultRouter()
router.register(r'students', StudentViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('analysis/process-exercise/', ExerciseAnalysisView.as_view(), name='process-exercise'), # Nueva ruta
]