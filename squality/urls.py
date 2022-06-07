from unicodedata import name
from django.urls import path

from . import views

urlpatterns = [

    path('squality/', views.index, name='index'),

    # about
    path('squality/about', views.about, name='about'),

    # Import
    path('squality/project/create', views.project_create, name='project_create'),
    path('squality/project/delete/<int:id>', views.project_delete, name='project_delete'),
    path('squality/project/details/<int:id>', views.project_details, name='project_details'),
    path('squality/project/details/<int:id>/eaupload', views.ea_upload, name='eametrics_upload'),
    path('squality/project/details/<int:id>/sdupload', views.sdmetrics_upload, name='sdmetrics_upload'),
    path('squality/project/details/<int:id>/s101upload', views.s101_upload, name='s101_upload'),
    path('squality/project/details/<int:id>/clocupload', views.cloc_upload, name='cloc_upload'),

    # Clean
    path('squality/project/cleaning/<int:id>', views.project_clean, name='project_clean'),
    path('squality/project/cleaning/delete/<int:class_id>', views.clean_delete, name='clean_delete'),
    path('squality/project/cleaning/rename/<int:project_id>/<str:type>', views.clean_rename, name='clean_rename'),
    path('squality/project/cleaning/sync/cloc/<int:project_id>', views.clean_syn_cloc, name='clean_syn_cloc'),
    path('squality/project/cleaning/remove/usage/<int:project_id>/<str:type>', views.clean_remove_usage, name='clean_remove_usage'),
    path('squality/project/cleaning/sync/s101/<int:project_id>', views.clean_syn_s101, name='clean_syn_s101'),

    # Cluster Metric
    path('squality/project/cluster/<int:project_id>/raw_migration', views.migrate_raw_normalize, name='raw_migration'),
    path('squality/project/cluster/<int:project_id>/metric', views.clustering_metric, name='clustering_metric'),
    path('squality/project/cluster/<int:project_id>/normalize', views.clustering_normalize, name='clustering_normalize'),

    # path('squality/project/cluster/normalize', views.clustering_normalize, name='clustering_normalize_test'),

    # Cluster Network
    path('squality/project/cluster/<int:project_id>/network', views.clustering_network, name='clustering_network'),

    # Cluster Combo
    path('squality/project/cluster/<int:project_id>/combo', views.clustering_combo, name='clustering_combo'),

    # Scoring
    path('squality/project/scoring/<int:project_id>/initialize', views.scoring_initialize, name='scoring_initialize'),
    path('squality/project/scoring/<int:project_id>', views.scoring, name='scoring'),

    # Summary
    path('squality/project/summary/<int:project_id>', views.summary, name='summary'),
    path('squality/project/summary_remarks', views.summary_remarks, name='summary_remarks'),
    
]