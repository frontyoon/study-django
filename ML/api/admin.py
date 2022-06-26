from django.contrib import admin
from api.models import emotion_object


@admin.register(emotion_object)
class emotion_objectAdmin(admin.ModelAdmin):
    list_display = ["emotion"]