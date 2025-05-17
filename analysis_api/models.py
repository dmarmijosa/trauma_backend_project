from django.db import models

class Student(models.Model):
    # Campos que el agente de IA preguntará y almacenará
    name = models.CharField(max_length=200)
    school = models.CharField(max_length=200, blank=True, null=True)
    # Otros datos que consideres relevantes (edad, grado, etc.)
    # age = models.IntegerField(blank=True, null=True)
    # grade = models.CharField(max_length=50, blank=True, null=True)

    # Campos para la "personalidad" (podrían ser más estructurados o un JSONField)
    likes = models.TextField(blank=True, null=True) # Qué le gusta
    personality_notes = models.TextField(blank=True, null=True) # Notas de la IA sobre la personalidad

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name