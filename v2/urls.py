from unicodedata import name
from django.urls import path

from . import views

urlpatterns = [

    path('v2/', views.index, name='v2_index'),

    path('v2/project/create', views.project_create, name='v2_project_create'),
    path('v2/project/import/<int:id>', views.project_import, name='v2_project_import'),
    path('v2/project/import/<int:id>/cupload', views.corpus_upload, name='v2_corpus_upload'),
]