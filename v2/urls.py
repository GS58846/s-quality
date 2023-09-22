from unicodedata import name
from django.urls import path

from . import views

urlpatterns = [

    path('v2/', views.index, name='v2_index'),

    path('v2/project/create', views.project_create, name='v2_project_create'),
    path('v2/project/delete/<int:id>', views.project_delete, name='v2_project_delete'),
    path('v2/project/import/<int:id>', views.project_import, name='v2_project_import'),
    path('v2/project/import/<int:id>/cupload', views.corpus_upload, name='v2_corpus_upload'),
    path('v2/project/import/<int:id>/s101upload', views.s101_upload, name='v2_s101_upload'),
    path('v2/project/import/<int:id>/comupload', views.complete_upload, name='v2_complete_upload'),

    path('v2/project/cleaning/<int:id>', views.project_clean, name='v2_project_clean'),
    path('v2/project/cleaning/delete/<int:class_id>', views.clean_delete, name='v2_clean_delete'),
    path('v2/project/cleaning/rename/<int:project_id>/<str:type>', views.clean_rename, name='v2_clean_rename'),
    path('v2/project/cleaning/remove/pkg`/<int:project_id>/<str:type>', views.clean_remove_pkg, name='v2_clean_remove_pkg'),
    path('v2/project/cleaning/remove/usage/<int:project_id>/<str:type>', views.clean_remove_usage, name='v2_clean_remove_usage'),
    path('v2/project/cleaning/sync/s101/<int:project_id>', views.clean_syn_s101, name='v2_clean_syn_s101'),

    path('v2/project/cluster/<int:project_id>/raw_migration', views.migrate_raw_normalize, name='v2_raw_migration'),
    path('v2/project/cluster/<int:project_id>/metric', views.view_cluster_metric, name='v2_cluster_metric'),
    path('v2/project/cluster/<int:project_id>/normalize', views.clustering_normalize, name='v2_clustering_normalize'),

    path('v2/project/cluster/<int:project_id>/kmeans', views.clustering_kmeans, name='v2_clustering_kmeans'),
    path('v2/project/generate/diagram/<int:project_id>/<str:algo>', views.generate_ms_diagram, name='v2_generate_ms_diagram'),
    path('v2/project/generate/radar/<int:project_id>/<str:algo>', views.generate_ms_radar_chart, name='v2_generate_ms_radar_chart'),

    path('v2/project/cluster/<int:project_id>/mean_shift', views.clustering_mean_shift, name='v2_clustering_mean_shift'),
    path('v2/project/cluster/<int:project_id>/agglomerative', views.clustering_agglomerative, name='v2_clustering_agglomerative'),
    path('v2/project/cluster/<int:project_id>/gaussian', views.clustering_gaussian, name='v2_clustering_gaussian'),

    path('v2/project/cluster/<int:project_id>/network', views.clustering_network, name='v2_clustering_network'),
    path('v2/project/cluster/<int:project_id>/network_run', views.clustering_network_run, name='v2_clustering_network_run'),

    path('v2/project/scoring/<int:project_id>/initialize', views.scoring_initialize, name='v2_scoring_initialize'),
    path('v2/project/scoring/<int:project_id>', views.scoring, name='v2_scoring'),
    # path('v2/project/scoring_topsis/<int:project_id>', views.scoring_topsis, name='v2_scoring_topsis'),

    path('v2/project/summary/<int:project_id>', views.summary_median, name='v2_summary'),
    path('v2/project/export_project_summary', views.export_project_summary, name='v2_export_project_summary'),

    path('v2/export/overall', views.export_overall_scoring, name='v2_export_overall_scoring'),
    path('v2/export/overall_normalize', views.export_overall_normalize, name='v2_export_overall_normalize'),

    path('v2/export/metric', views.export_metric_scoring, name='v2_export_metric_scoring'),
    path('v2/export/metric_normalize', views.export_metric_normalize, name='v2_export_metric_normalize'),

    path('v2/export/network', views.export_network_scoring, name='v2_export_network_scoring'),
    path('v2/export/network_normalize', views.export_network_normalize, name='v2_export_network_normalize'),

    path('v2/export/ms_metric/<int:project_id>', views.export_ms_metric, name='v2_export_ms_metric'),
]