from django.urls import path
from . import views
 
urlpatterns = [
    path('',views.index, name='index'),
    path('createTodo/', views.create_Todo, name='create_Todo'),
    path('deleteTodo/', views.delete_todo, name='delete_Todo'),
]