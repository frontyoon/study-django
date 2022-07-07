from django.db import models
from django.contrib.auth.models import UserManager, AbstractUser


class User(AbstractUser):
    objects = UserManager()

    nickname = models.CharField(max_length=50)
    email = models.EmailField(max_length=200)