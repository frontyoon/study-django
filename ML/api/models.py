from django.db import models


class emotion_object(models.Model):
    emotion = models.CharField(max_length=200)