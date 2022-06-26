from dataclasses import field
from rest_framework import serializers
from api.models import emotion_object


class emotion_objectSerializer(serializers.ModelSerializer):
    class Meta:
        model = emotion_object
        field = '__all__'