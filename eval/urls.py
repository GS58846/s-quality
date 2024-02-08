from unicodedata import name
from django.urls import path

from . import views

urlpatterns = [

    path('eval/', views.index, name='eval_index'),
    path('eval/project/create', views.project_create, name='eval_project_create'),
    path('eval/project/delete/<int:id>', views.project_delete, name='eval_project_delete'),
    path('eval/project/assign/<int:id>', views.project_assign, name='eval_project_assign'),

    path('eval/project/cluster/<int:id>', views.project_cluster, name='eval_project_cluster'),

    path('eval/project/generate/diagram/<int:project_id>', views.generate_ms_diagram, name='eval_generate_ms_diagram'),
    path('eval/project/generate/radar/<int:project_id>', views.generate_ms_radar_chart, name='eval_generate_ms_radar_chart'),
]