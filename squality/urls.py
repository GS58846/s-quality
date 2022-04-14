from unicodedata import name
from django.urls import path

from . import views

urlpatterns = [

    path('squality/', views.index, name='index'),

    # Import
    path('squality/project/create', views.project_create, name='project_create'),
    path('squality/project/delete/<int:id>', views.project_delete, name='project_delete'),
    path('squality/project/details/<int:id>', views.project_details, name='project_details'),
    path('squality/project/details/<int:id>/sdupload', views.sdmetrics_upload, name='sdmetrics_upload'),
    path('squality/project/details/<int:id>/s101upload', views.s101_upload, name='s101_upload'),
    path('squality/project/details/<int:id>/clocupload', views.cloc_upload, name='cloc_upload'),

    # Clean
    path('squality/project/cleaning/<int:id>', views.project_clean, name='project_clean'),
    path('squality/project/cleaning/delete/<int:class_id>', views.clean_delete, name='clean_delete'),
    path('squality/project/cleaning/rename/<int:project_id>/<str:type>', views.clean_rename, name='clean_rename'),
    path('squality/project/cleaning/sync/cloc/<int:project_id>', views.clean_syn_cloc, name='clean_syn_cloc'),
    path('squality/project/cleaning/remove/usage/<int:project_id>/<str:type>', views.clean_remove_usage, name='clean_remove_usage'),
]