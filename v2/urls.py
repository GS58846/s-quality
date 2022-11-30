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
    path('v2/project/cluster/<int:project_id>/kmeans_timer', views.clustering_kmeans_timer, name='v2_clustering_kmeans_timer'),
    path('v2/project/cluster/<int:project_id>/mean_shift', views.clustering_mean_shift, name='v2_clustering_mean_shift'),
    path('v2/project/cluster/<int:project_id>/agglomerative', views.clustering_agglomerative, name='v2_clustering_agglomerative'),
    path('v2/project/cluster/<int:project_id>/gaussian', views.clustering_gaussian, name='v2_clustering_gaussian'),

    path('v2/project/cluster/<int:project_id>/network', views.clustering_network, name='v2_clustering_network'),
    path('v2/project/cluster/<int:project_id>/network_run', views.clustering_network_run, name='v2_clustering_network_run'),

    path('v2/project/scoring/<int:project_id>/initialize', views.scoring_initialize, name='v2_scoring_initialize'),
    path('v2/project/scoring/<int:project_id>', views.scoring, name='v2_scoring'),

    path('v2/project/summary/<int:project_id>', views.summary, name='v2_summary'),
]