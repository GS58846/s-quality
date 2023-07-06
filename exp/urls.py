from unicodedata import name
from django.urls import path

from . import views

urlpatterns = [

    path('exp/', views.index, name='exp_index'),
    path('exp/project/create', views.project_create, name='exp_project_create'),
    path('exp/project/delete/<int:id>', views.project_delete, name='exp_project_delete'),
    path('exp/project/assign/<int:id>', views.project_assign, name='exp_project_assign'),

    path('exp/project/cluster/<int:id>', views.project_cluster, name='exp_project_cluster'),

    path('exp/project/generate/diagram/<int:project_id>', views.generate_ms_diagram, name='exp_generate_ms_diagram'),
    path('exp/project/generate/radar/<int:project_id>', views.generate_ms_radar_chart, name='exp_generate_ms_radar_chart'),
]