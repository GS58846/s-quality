from collections import defaultdict
from locale import normalize
from django.db.models import Q
from csv import DictReader, reader
from fileinput import filename
from functools import reduce
import io
from os import rename
import os
from platform import node
import random
import re
from statistics import mode
import string
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from webbrowser import get
from django.http import HttpRequest
from django.shortcuts import redirect, render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
import networkx as nx
import igraph
from igraph import *
from sklearn.metrics import silhouette_score


from squality.models import ClocMetric, ClocMetricRaw, ClusteringNormalize, Clustering, ClusteringMetric, GraphImages, MetricNormalize, Project, S101Metric, S101MetricRaw, ScoringAverage, ScoringFinale, SdMetric, SdMetricRaw


# Create your views here.

def index(request):
    data = {
        'projects': Project.objects.all()
    }

    return render(request, 'squality/index.html', data)

def about(request):
    return render(request, 'squality/about.html')

# PROJECT

def project_create(request: HttpRequest):
    project = Project( name = request.POST['name'])
    project.save()
    return redirect('/squality')

def project_delete(request, id):
    Project.objects.filter(id=id).delete()
    ClocMetric.objects.filter(project_id=id).delete()
    ClocMetricRaw.objects.filter(project_id=id).delete()
    Clustering.objects.filter(project_id=id).delete()
    ClusteringMetric.objects.filter(project_id=id).delete()
    ClusteringNormalize.objects.filter(project_id=id).delete()
    GraphImages.objects.filter(project_id=id).delete()
    MetricNormalize.objects.filter(project_id=id).delete()
    S101Metric.objects.filter(project_id=id).delete()
    S101MetricRaw.objects.filter(project_id=id).delete()
    ScoringAverage.objects.filter(project_id=id).delete()
    ScoringFinale.objects.filter(project_id=id).delete()
    SdMetric.objects.filter(project_id=id).delete()
    SdMetricRaw.objects.filter(project_id=id).delete()

    return redirect('/squality')

def project_details(request, id):
    try:
        project = Project.objects.get(id=id)

        sdo = SdMetric.objects.filter(project=project)
        s101o = S101Metric.objects.filter(project=project)
        cloco = ClocMetric.objects.filter(project=project)

        if sdo.count() > 0:
            sdmetric_file = sdo.get()
        else:
            sdmetric_file = False

        if s101o.count() > 0:
            s101_file = s101o.get()
        else:
            s101_file = False

        if cloco.count() > 0:
            cloco_file = cloco.get()
        else:
            cloco_file = False

        data = {
            'project': project,
            'sdmetric_file': sdmetric_file,
            's101_file': s101_file,
            'cloc_file': cloco_file
        }

        return render(request, 'squality/project_details.html', data)
    except Exception as exc:
        return redirect('/squality')


# EXTRACT

def sdmetrics_upload(request, id):
    if request.method == 'POST' and request.FILES['sdupload']:

        # upload csv file
        upload = request.FILES['sdupload']
        fss = FileSystemStorage()
        new_name = 'SD-'+''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) + '.csv'
        file = fss.save(new_name, upload)
        file_url = fss.url(file)

        project = Project.objects.get(id=id)

        if SdMetric.objects.filter(project=project).count() > 0:
            p = SdMetric.objects.filter(project=project).get()
            p.filename = new_name
            p.fileurl = file_url
            p.save()

            if SdMetricRaw.objects.filter(project_id=id).count() > 0:
                SdMetricRaw.objects.filter(project_id=id).delete()

        else:
            sdmetric = SdMetric(
                filename = new_name,
                fileurl = file_url,
                project = Project.objects.get(id=id)
            )
            sdmetric.save()
        
        # read saved csv file
        # base_dir = settings.BASE_DIR
        csv_folder = os.path.join(settings.BASE_DIR, 'uploads/csv/' + new_name)
        local_csv = csv_folder

        with open(local_csv, mode='r', encoding="utf-8-sig") as csv_file:
            csv_reader = DictReader(csv_file)
            for row in csv_reader:
                sdmetric_raw = SdMetricRaw()
                sdmetric_raw.class_name = row['Name']
                # sdmetric_raw.cbo = row['CBO']
                # sdmetric_raw.ic = row['IC']
                # sdmetric_raw.oc = row['OC']
                sdmetric_raw.cam = row['CAM']
                sdmetric_raw.nco = row['NumOps']
                sdmetric_raw.dit = row['DIT']
                # sdmetric_raw.rfc = row['RFC']
                # sdmetric_raw.loc = row['LOC']
                sdmetric_raw.nca = row['NumAttr']
                sdmetric_raw.project_id = id
                sdmetric_raw.save()

        return redirect('project_details', id=id)
    return redirect('/squality')


def s101_upload(request, id):
    if request.method == 'POST' and request.FILES['s101upload']:

        upload = request.FILES['s101upload']
        fss = FileSystemStorage()
        new_name = 'S101-'+''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) + '.csv'
        file = fss.save(new_name, upload)
        file_url = fss.url(file)

        project = Project.objects.get(id=id)

        if(S101Metric.objects.filter(project=project).count() > 0):
            p = S101Metric.objects.filter(project=project).get()
            p.filename = new_name
            p.fileurl = file_url
            p.save()

            if S101MetricRaw.objects.filter(project_id=id).count() > 0:
                S101MetricRaw.objects.filter(project_id=id).delete()
                
        else:
            s101 = S101Metric(
                filename = new_name,
                fileurl = file_url,
                project = Project.objects.get(id=id)
            )
            s101.save()

        # read saved csv file
        # base_dir = settings.BASE_DIR
        csv_folder = os.path.join(settings.BASE_DIR, 'uploads/csv/' + new_name)
        local_csv = csv_folder

        with open(local_csv, mode='r', encoding="utf-8-sig") as csv_file:
            csv_reader = reader(csv_file)
            for row in csv_reader:
                print(row[3])
                s101_raw = S101MetricRaw()
                s101_raw.class_from = row[0]
                s101_raw.usage = row[1]
                s101_raw.class_to = row[2]

                s101_raw.weight = re.search(r"(?<=\[)[^][]*(?=])", row[3]).group(0)

                s101_raw.project_id = id
                s101_raw.save()

        return redirect('project_details', id=id)
    return redirect('/squality')


def cloc_upload(request, id):
    if request.method == 'POST' and request.FILES['clocupload']:

        upload = request.FILES['clocupload']
        fss = FileSystemStorage()
        new_name = 'CLOC-'+''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) + '.csv'
        file = fss.save(new_name, upload)
        file_url = fss.url(file)

        project = Project.objects.get(id=id)

        if ClocMetric.objects.filter(project=project).count() > 0:
            p = ClocMetric.objects.filter(project=project).get()
            p.filename = new_name
            p.fileurl = file_url
            p.save()

            if ClocMetricRaw.objects.filter(project_id=project.id).count() > 0:
                ClocMetricRaw.objects.filter(project_id=project.id).delete()

        else:
            cloc = ClocMetric(
                filename = new_name,
                fileurl = file_url,
                project = Project.objects.get(id=id)
            )
            cloc.save()

        # read saved csv file
        # base_dir = settings.BASE_DIR
        csv_folder = os.path.join(settings.BASE_DIR, 'uploads/csv/' + new_name)
        local_csv = csv_folder

        with open(local_csv, mode='r', encoding="utf-8-sig") as csv_file:
            csv_reader = DictReader(csv_file)
            for row in csv_reader:
                if row['language']=='Java':
                    cloc_raw = ClocMetricRaw()
                    cloc_raw.class_name = row['filename'].replace('/','.')
                    cloc_raw.comment = row['comment']
                    cloc_raw.code = row['code']
                    cloc_raw.project_id = id
                    cloc_raw.save()

        return redirect('project_details', id=id)
    return redirect('/squality')


# CLEANING

def project_clean(request, id):
    try:
        project = Project.objects.get(id=id)

        sdmetric_data = SdMetricRaw.objects.order_by('class_name').all().filter(project_id=id)
        
        cloc_data = ClocMetricRaw.objects.order_by('class_name').all().filter(project_id=id)
        
        s101_data_from = S101MetricRaw.objects.order_by('class_from').all().filter(project_id=id).distinct('class_from')
        s101_data_to = S101MetricRaw.objects.order_by('class_to').all().filter(project_id=id).distinct('class_to')
        s101_usages = S101MetricRaw.objects.order_by('usage').all().filter(project_id=id).distinct('usage')

        data = {
            'project': project,
            'sdmetrics': sdmetric_data,
            'clocs': cloc_data,
            's101s_from': s101_data_from,
            's101s_to': s101_data_to,
            's101_usages': s101_usages
        }

        return render(request, 'squality/project_clean.html', data)
    except Exception as exc:
        return redirect('/squality')

def clean_delete(request, class_id):
    sdmetric = SdMetricRaw.objects.get(id=class_id)
    project_id = sdmetric.project_id
    sdmetric.delete()
    return redirect('project_clean', id=project_id)

def clean_rename(request, project_id, type):
    
    if type=='sdmetric':
        remove_string = request.POST['str_sdmetric']
        sdmetrics = SdMetricRaw.objects.filter(project_id=project_id).all()
    
        for sd in sdmetrics:
            sd.class_name = sd.class_name.replace(remove_string,'')
            sd.save()

    elif type=='cloc':
        remove_string = request.POST['str_cloc']
        clocs = ClocMetricRaw.objects.filter(project_id=project_id).all()
    
        for cloc in clocs:
            cloc.class_name = cloc.class_name.replace(remove_string,'')
            cloc.save()

    elif type=='s101':
        remove_string = request.POST['str_s101']
        s101 = S101MetricRaw.objects.filter(project_id=project_id).all()

        for s in s101:
            s.class_from = s.class_from.replace(remove_string,'')
            s.class_to = s.class_to.replace(remove_string,'')
            s.save()

    return redirect('project_clean', id=project_id)

def clean_syn_cloc(request, project_id):

    sd_classes = SdMetricRaw.objects.filter(project_id=project_id).all()
    # cloc_list = ClocMetricRaw.objects.filter(project_id=project_id).all()
    
    for sd in sd_classes:
        if ClocMetricRaw.objects.filter(class_name=sd.class_name,project_id=project_id):
            cloc = ClocMetricRaw.objects.filter(class_name=sd.class_name,project_id=project_id).get()
            sd.loc = cloc.code
            sd.save()
            cloc.ok = 1
            cloc.save()

    cloc_metric = ClocMetricRaw.objects.filter(ok=0).all()
    cloc_metric.delete()

    return redirect('project_clean', id=project_id)

def clean_remove_usage(request, project_id, type):
    usage_list = S101MetricRaw.objects.filter(project_id=project_id, usage=type).all()
    usage_list.delete()
    return redirect('project_clean', id=project_id)

def clean_syn_s101(request, project_id):
    # clean from
    sd_classes = SdMetricRaw.objects.filter(project_id=project_id).all()
    # sync with sdmetric class list
    for sd in sd_classes:
        if S101MetricRaw.objects.filter(class_from=sd.class_name):
            s101from_list = S101MetricRaw.objects.filter(class_from=sd.class_name).all()
            for sfl in s101from_list:
                sfl.ok_from = 1
                sfl.save()
    remove_metric = S101MetricRaw.objects.filter(ok_from=0).all()
    remove_metric.delete()
    # clean to
    sd_classes = SdMetricRaw.objects.filter(project_id=project_id).all()
    for sd in sd_classes:    
        if S101MetricRaw.objects.filter(class_to=sd.class_name):
            s101to_list = S101MetricRaw.objects.filter(class_to=sd.class_name).all()
            for sft in s101to_list:
                sft.ok_to = 1
                sft.save()
    remove_metric = S101MetricRaw.objects.filter(ok_to=0).all()
    remove_metric.delete()

    sd_classes = SdMetricRaw.objects.filter(project_id=project_id).all()
    for sd in sd_classes:
        # rfc
        if S101MetricRaw.objects.filter(project_id=project_id, usage='returns', class_from=sd.class_name).count() > 0:
            rfc_list = S101MetricRaw.objects.filter(project_id=project_id, usage='returns', class_from=sd.class_name).all()
            rfc_value = 0
            for rfc in rfc_list:
                rfc_value += rfc.weight
        else:
            rfc_value = 0
        sd.rfc = rfc_value
        # ic
        if S101MetricRaw.objects.filter(project_id=project_id, class_to=sd.class_name).count() > 0:
            ic_list = S101MetricRaw.objects.filter(project_id=project_id, class_to=sd.class_name).all()
            ic_value = 0
            for ic in ic_list:
                ic_value += ic.weight
        else:
            ic_value = 0
        sd.ic = ic_value
        # oc
        if S101MetricRaw.objects.filter(project_id=project_id, class_from=sd.class_name).count() > 0:
            oc_list = S101MetricRaw.objects.filter(project_id=project_id, class_from=sd.class_name).all()
            oc_value = 0
            for oc in oc_list:
                oc_value += oc.weight
        else:
            oc_value = 0
        sd.oc = oc_value
        # cbo
        if S101MetricRaw.objects.filter(project_id=project_id, class_from=sd.class_name).count() > 0:
            cbo = S101MetricRaw.objects.filter(project_id=project_id, class_from=sd.class_name).all().distinct('class_to').count()
            cbo_value = cbo
        else:
            cbo_value = 0
        sd.cbo = cbo_value 

        sd.save()

    return redirect('project_clean', id=project_id)

# METRIC CLUSTERING

def migrate_raw_normalize(request, project_id):
    if MetricNormalize.objects.filter(project_id=project_id).count() > 0:
        MetricNormalize.objects.filter(project_id=project_id).delete()

    raw_data = SdMetricRaw.objects.filter(project_id=project_id).all()
    for row in raw_data:
        normalize_data = MetricNormalize(
            class_name = row.class_name,
            cbo = row.cbo,
            ic = row.ic,
            oc = row.oc,
            cam = row.cam,
            nco = row.nco,
            dit = row.dit,
            rfc = row.rfc,
            loc = row.loc,
            nca = row.nca,
            project_id = project_id
        )
        normalize_data.save()
    return redirect('clustering_metric', project_id=project_id)

def clustering_metric(request, project_id):
    project = Project.objects.get(id=project_id)
    sdmetric_data = MetricNormalize.objects.order_by('class_name').all().filter(project_id=project_id)

    if MetricNormalize.objects.order_by('class_name').filter(project_id=project_id, normalized=1).count() > 0:
        state = 'disabled'
    else:
        state = ''

    # k-mean

    raw_data = MetricNormalize.objects.filter(project_id=project_id).all().values()
    df = pd.DataFrame(raw_data)
    df_metric = df.iloc[:,2:-2]

    class_count = MetricNormalize.objects.order_by('class_name').filter(project_id=project_id).count()

    # the elbow method
    kmeans_args = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42
    }

    sse = []
    sse_list = []

    for k in range(1,class_count): #rows
        kmeans = KMeans(n_clusters=k, **kmeans_args)
        kmeans.fit(df_metric)
        sse.append([k, kmeans.inertia_])
        sse_list.append(kmeans.inertia_)

    k_value = KneeLocator(range(1,class_count), sse_list, curve="convex", direction="decreasing")
    k_value.elbow  
    
    kmeans_minmax = KMeans(k_value.elbow).fit(df_metric)
    kmeans_clusters = kmeans_minmax.fit_predict(df_metric)

    df_kmeans = df.iloc[:,1:2].copy()
    df_kmeans['kmeans'] = kmeans_clusters

    # save into db
    if Clustering.objects.filter(project_id=project_id,algo='kmeans').count() > 0:
        Clustering.objects.filter(project_id=project_id,algo='kmeans').delete()

    for k in df_kmeans.index:
        c = Clustering(
            class_name = df_kmeans['class_name'][k],
            cluster = df_kmeans['kmeans'][k],
            type = 'metric',
            algo = 'kmeans',
            project_id = project_id
        )
        c.save()
    
    kmeans_group = Clustering.objects.filter(project_id=project_id,algo='kmeans').order_by('cluster').all()

    # kmeans summary
    # TODO: separate as re-usable function? start ---------------------------------------------

    if ClusteringMetric.objects.filter(project_id=project_id,algo='kmeans').count() > 0:
        ClusteringMetric.objects.filter(project_id=project_id,algo='kmeans').delete()

    ms_grp = defaultdict(list)
    ms_len = Clustering.objects.filter(project_id=project_id,algo='kmeans').distinct('cluster').count()
    for i in range(ms_len):
        mloc = 0
        mnoc = 0
        ncam = 0
        imc = 0
        nmo = 0
        cluster_grp = []
        cls = Clustering.objects.filter(project_id=project_id,algo='kmeans',cluster=i).all()
        for c in cls:
            cm = SdMetricRaw.objects.filter(project_id=project_id,class_name=c.class_name).get()
            mloc += cm.loc
            mnoc += 1 
            nmo += cm.nco
            ncam += cm.cam
            cluster_grp.append(c.class_name)
            ms_grp[i].append(c.class_name)
        # imc
        for cl in cluster_grp:
            imc_list = S101MetricRaw.objects.filter(project_id=project_id,class_from=cl).all()
            for il in imc_list:
                # if il.class_to != cl:
                if ((il.class_to in cluster_grp) and (il.class_to != cl)):
                    imc += il.weight


        ncam = ncam / mnoc
        imc = imc       
        
        fms = ClusteringMetric(
            algo = 'kmeans',
            type = 'metric',
            microservice = i,
            mloc = mloc,
            mnoc = mnoc,
            ncam = ncam,
            imc = imc,
            nmo = nmo,
            project_id = project_id
        )
        fms.save()

    # wcbm

    for key, val in ms_grp.items():
        ms_wcbm = 0
        ms_trm = 0
        if S101MetricRaw.objects.filter(class_from__in=val, project_id=project_id).count() > 0:
            cf = S101MetricRaw.objects.filter(class_from__in=val, project_id=project_id).all()
            for cc in cf:
                if cc.class_to not in val:
                    ms_wcbm += cc.weight
                    if cc.usage == 'returns':
                        ms_trm += cc.weight
        ms_x = ClusteringMetric.objects.filter(project_id=project_id, microservice=key, algo='kmeans').get()
        ms_x.wcbm = ms_wcbm
        ms_x.trm = ms_trm
        ms_x.save()

    # cbm
    
    for key, val in ms_grp.items():
        ms_cbm = 0
        ms_acbm = 0
        # print('MS'+str(key)+' ===========================================')
        # print(val)
        for i in range(ms_len):
            if key != i:
                if S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_grp[i], project_id=project_id):
                    ms_cbm += 1
                   
                    if S101MetricRaw.objects.filter(class_from__in=ms_grp[i], class_to__in=val, project_id=project_id):
                        # print('     MS'+str(i)) 
                        # print(ms_grp[i])
                        # print('---------------')
                        ms_from = S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_grp[i], project_id=project_id).all()
                        for mf in ms_from:
                            ms_acbm += mf.weight
                            # print(mf.class_from + '-' + str(mf.weight))
                        # print('...............')
                        ms_to = S101MetricRaw.objects.filter(class_from__in=ms_grp[i], class_to__in=val, project_id=project_id).all()
                        for mt in ms_to:
                            ms_acbm += mt.weight
                            # print(mt.class_from + '-' + str(mt.weight))
                # if (S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_grp[i]) and S101MetricRaw.objects.filter(class_from__in=ms_grp[i], class_to__in=val)):
                #     # ms_acbm += 1
                #     acbm_from_list = S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_grp[i]).all()
                #     for afl in acbm_from_list:
                #         ms_acbm += afl.weight
                #     acbm_to_list = S101MetricRaw.objects.filter(class_to__in=val, class_from__in=ms_grp[i]).all()
                #     for atl in acbm_to_list:
                #         ms_acbm += atl.weight


        ms_x = ClusteringMetric.objects.filter(project_id=project_id, microservice=key, algo='kmeans').get()
        ms_x.cbm = ms_cbm
        ms_x.acbm = ms_acbm
        ms_x.save()

    # TODO: separate as re-usable function? end ---------------------------------------------
 
    # print(ms_grp[0])

    # mean-shift

    mshift = MeanShift()
    mshift_cluster = mshift.fit_predict(df_metric)
    df_mshift = df.iloc[:,1:2].copy()
    df_mshift['mean_shift'] = mshift_cluster
    
    # save into db
    if Clustering.objects.filter(project_id=project_id,algo='mean_shift').count() > 0:
        Clustering.objects.filter(project_id=project_id,algo='mean_shift').delete()

    for k in df_mshift.index:
        c = Clustering(
            class_name = df_mshift['class_name'][k],
            cluster = df_mshift['mean_shift'][k],
            type = 'metric',
            algo = 'mean_shift',
            project_id = project_id
        )
        c.save()
    
    mshift_group = Clustering.objects.filter(project_id=project_id,algo='mean_shift').order_by('cluster').all()

    # mean-shift summary
    # TODO: separate as re-usable function? start ---------------------------------------------

    if ClusteringMetric.objects.filter(project_id=project_id,algo='mean_shift').count() > 0:
        ClusteringMetric.objects.filter(project_id=project_id,algo='mean_shift').delete()

    ms_ms_grp = defaultdict(list)
    ms_ms_len = Clustering.objects.filter(project_id=project_id,algo='mean_shift').distinct('cluster').count()
    for i in range(ms_ms_len):
        mloc = 0
        mnoc = 0
        ncam = 0
        imc = 0
        nmo = 0
        cluster_grp = []
        cls = Clustering.objects.filter(project_id=project_id,algo='mean_shift',cluster=i).all()
        for c in cls:
            cm = SdMetricRaw.objects.filter(project_id=project_id,class_name=c.class_name).get()
            mloc += cm.loc
            mnoc += 1 
            nmo += cm.nco
            ncam += cm.cam
            cluster_grp.append(c.class_name)
            ms_ms_grp[i].append(c.class_name)
        # imc
        for cl in cluster_grp:
            imc_list = S101MetricRaw.objects.filter(project_id=project_id,class_from=cl).all()
            for il in imc_list:
                # if il.class_to != cl:
                if ((il.class_to in cluster_grp) and (il.class_to != cl)):
                    imc += il.weight


        ncam = ncam / mnoc
        imc = imc       
        
        fms = ClusteringMetric(
            algo = 'mean_shift',
            type = 'metric',
            microservice = i,
            mloc = mloc,
            mnoc = mnoc,
            ncam = ncam,
            imc = imc,
            nmo = nmo,
            project_id = project_id
        )
        fms.save()

    # wcbm

    for key, val in ms_ms_grp.items():
        ms_wcbm = 0
        ms_trm = 0
        if S101MetricRaw.objects.filter(class_from__in=val, project_id=project_id).count() > 0:
            cf = S101MetricRaw.objects.filter(class_from__in=val, project_id=project_id).all()
            for cc in cf:
                if cc.class_to not in val:
                    ms_wcbm += cc.weight
                    if cc.usage == 'returns':
                        ms_trm += cc.weight
        ms_x = ClusteringMetric.objects.filter(project_id=project_id, microservice=key, algo='mean_shift').get()
        ms_x.wcbm = ms_wcbm
        ms_x.trm = ms_trm
        ms_x.save()

    # cbm
    
    for key, val in ms_ms_grp.items():
        ms_cbm = 0
        ms_acbm = 0
 
        for i in range(ms_ms_len):
            if key != i:
                if S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_ms_grp[i], project_id=project_id):
                    ms_cbm += 1
                   
                    if S101MetricRaw.objects.filter(class_from__in=ms_ms_grp[i], class_to__in=val, project_id=project_id):
                        ms_from = S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_ms_grp[i], project_id=project_id).all()
                        for mf in ms_from:
                            ms_acbm += mf.weight
                        ms_to = S101MetricRaw.objects.filter(class_from__in=ms_ms_grp[i], class_to__in=val, project_id=project_id).all()
                        for mt in ms_to:
                            ms_acbm += mt.weight

        ms_x = ClusteringMetric.objects.filter(project_id=project_id, microservice=key, algo='mean_shift').get()
        ms_x.cbm = ms_cbm
        ms_x.acbm = ms_acbm
        ms_x.save()

    # TODO: separate as re-usable function? end ---------------------------------------------

    # agglomerative
    agglomerative = AgglomerativeClustering(k_value.elbow)
    agglomerative_cluster = agglomerative.fit_predict(df_metric)
    df_agglomerative = df.iloc[:,1:2].copy()
    df_agglomerative['agglomerative'] = agglomerative_cluster
    
    # save into db
    if Clustering.objects.filter(project_id=project_id,algo='agglomerative').count() > 0:
        Clustering.objects.filter(project_id=project_id,algo='agglomerative').delete()

    for k in df_agglomerative.index:
        c = Clustering(
            class_name = df_agglomerative['class_name'][k],
            cluster = df_agglomerative['agglomerative'][k],
            type = 'metric',
            algo = 'agglomerative',
            project_id = project_id
        )
        c.save()
    
    agglomerative_group = Clustering.objects.filter(project_id=project_id,algo='agglomerative').order_by('cluster').all()

    # agglomerative summary
    # TODO: separate as re-usable function? start ---------------------------------------------

    if ClusteringMetric.objects.filter(project_id=project_id,algo='agglomerative').count() > 0:
        ClusteringMetric.objects.filter(project_id=project_id,algo='agglomerative').delete()

    ms_ms_grp = defaultdict(list)
    ms_ms_len = Clustering.objects.filter(project_id=project_id,algo='agglomerative').distinct('cluster').count()
    for i in range(ms_ms_len):
        mloc = 0
        mnoc = 0
        ncam = 0
        imc = 0
        nmo = 0
        cluster_grp = []
        cls = Clustering.objects.filter(project_id=project_id,algo='agglomerative',cluster=i).all()
        for c in cls:
            cm = SdMetricRaw.objects.filter(project_id=project_id,class_name=c.class_name).get()
            mloc += cm.loc
            mnoc += 1 
            nmo += cm.nco
            ncam += cm.cam
            cluster_grp.append(c.class_name)
            ms_ms_grp[i].append(c.class_name)
        # imc
        for cl in cluster_grp:
            imc_list = S101MetricRaw.objects.filter(project_id=project_id,class_from=cl).all()
            for il in imc_list:
                # if il.class_to != cl:
                if ((il.class_to in cluster_grp) and (il.class_to != cl)):
                    imc += il.weight


        ncam = ncam / mnoc
        imc = imc       
        
        fms = ClusteringMetric(
            algo = 'agglomerative',
            type = 'metric',
            microservice = i,
            mloc = mloc,
            mnoc = mnoc,
            ncam = ncam,
            imc = imc,
            nmo = nmo,
            project_id = project_id
        )
        fms.save()

    # wcbm

    for key, val in ms_ms_grp.items():
        ms_wcbm = 0
        ms_trm = 0
        if S101MetricRaw.objects.filter(class_from__in=val, project_id=project_id).count() > 0:
            cf = S101MetricRaw.objects.filter(class_from__in=val, project_id=project_id).all()
            for cc in cf:
                if cc.class_to not in val:
                    ms_wcbm += cc.weight
                    if cc.usage == 'returns':
                        ms_trm += cc.weight
        ms_x = ClusteringMetric.objects.filter(project_id=project_id, microservice=key, algo='agglomerative').get()
        ms_x.wcbm = ms_wcbm
        ms_x.trm = ms_trm
        ms_x.save()

    # cbm
    
    for key, val in ms_ms_grp.items():
        ms_cbm = 0
        ms_acbm = 0
 
        for i in range(ms_ms_len):
            if key != i:
                if S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_ms_grp[i], project_id=project_id):
                    ms_cbm += 1
                   
                    if S101MetricRaw.objects.filter(class_from__in=ms_ms_grp[i], class_to__in=val, project_id=project_id):
                        ms_from = S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_ms_grp[i], project_id=project_id).all()
                        for mf in ms_from:
                            ms_acbm += mf.weight
                        ms_to = S101MetricRaw.objects.filter(class_from__in=ms_ms_grp[i], class_to__in=val, project_id=project_id).all()
                        for mt in ms_to:
                            ms_acbm += mt.weight

        ms_x = ClusteringMetric.objects.filter(project_id=project_id, microservice=key, algo='agglomerative').get()
        ms_x.cbm = ms_cbm
        ms_x.acbm = ms_acbm
        ms_x.save()

    # TODO: separate as re-usable function? end ---------------------------------------------

    # gaussian mixture

    gaussian = GaussianMixture(k_value.elbow)
    gaussian_cluster = gaussian.fit_predict(df_metric)
    df_gaussian = df.iloc[:,1:2].copy()
    df_gaussian['gaussian'] = gaussian_cluster
    
    # save into db
    if Clustering.objects.filter(project_id=project_id,algo='gaussian').count() > 0:
        Clustering.objects.filter(project_id=project_id,algo='gaussian').delete()

    for k in df_gaussian.index:
        c = Clustering(
            class_name = df_gaussian['class_name'][k],
            cluster = df_gaussian['gaussian'][k],
            type = 'metric',
            algo = 'gaussian',
            project_id = project_id
        )
        c.save()
    
    gaussian_group = Clustering.objects.filter(project_id=project_id,algo='gaussian').order_by('cluster').all()

    # gaussian summary
    # TODO: separate as re-usable function? start ---------------------------------------------

    if ClusteringMetric.objects.filter(project_id=project_id,algo='gaussian').count() > 0:
        ClusteringMetric.objects.filter(project_id=project_id,algo='gaussian').delete()

    ms_ms_grp = defaultdict(list)
    ms_ms_len = Clustering.objects.filter(project_id=project_id,algo='gaussian').distinct('cluster').count()
    for i in range(ms_ms_len):
        mloc = 0
        mnoc = 0
        ncam = 0
        imc = 0
        nmo = 0
        cluster_grp = []
        cls = Clustering.objects.filter(project_id=project_id,algo='gaussian',cluster=i).all()
        for c in cls:
            cm = SdMetricRaw.objects.filter(project_id=project_id,class_name=c.class_name).get()
            mloc += cm.loc
            mnoc += 1 
            nmo += cm.nco
            ncam += cm.cam
            cluster_grp.append(c.class_name)
            ms_ms_grp[i].append(c.class_name)
        # imc
        for cl in cluster_grp:
            imc_list = S101MetricRaw.objects.filter(project_id=project_id,class_from=cl).all()
            for il in imc_list:
                # if il.class_to != cl:
                if ((il.class_to in cluster_grp) and (il.class_to != cl)):
                    imc += il.weight


        ncam = ncam / mnoc
        imc = imc       
        
        fms = ClusteringMetric(
            algo = 'gaussian',
            type = 'metric',
            microservice = i,
            mloc = mloc,
            mnoc = mnoc,
            ncam = ncam,
            imc = imc,
            nmo = nmo,
            project_id = project_id
        )
        fms.save()

    # wcbm

    for key, val in ms_ms_grp.items():
        ms_wcbm = 0
        ms_trm = 0
        if S101MetricRaw.objects.filter(class_from__in=val, project_id=project_id).count() > 0:
            cf = S101MetricRaw.objects.filter(class_from__in=val, project_id=project_id).all()
            for cc in cf:
                if cc.class_to not in val:
                    ms_wcbm += cc.weight
                    if cc.usage == 'returns':
                        ms_trm += cc.weight
        ms_x = ClusteringMetric.objects.filter(project_id=project_id, microservice=key, algo='gaussian').get()
        ms_x.wcbm = ms_wcbm
        ms_x.trm = ms_trm
        ms_x.save()

    # cbm
    
    for key, val in ms_ms_grp.items():
        ms_cbm = 0
        ms_acbm = 0
 
        for i in range(ms_ms_len):
            if key != i:
                if S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_ms_grp[i], project_id=project_id):
                    ms_cbm += 1
                   
                    if S101MetricRaw.objects.filter(class_from__in=ms_ms_grp[i], class_to__in=val, project_id=project_id):
                        ms_from = S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_ms_grp[i], project_id=project_id).all()
                        for mf in ms_from:
                            ms_acbm += mf.weight
                        ms_to = S101MetricRaw.objects.filter(class_from__in=ms_ms_grp[i], class_to__in=val, project_id=project_id).all()
                        for mt in ms_to:
                            ms_acbm += mt.weight

        ms_x = ClusteringMetric.objects.filter(project_id=project_id, microservice=key, algo='gaussian').get()
        ms_x.cbm = ms_cbm
        ms_x.acbm = ms_acbm
        ms_x.save()

    # TODO: separate as re-usable function? end ---------------------------------------------

    # ---------------

    ms_kmeans = ClusteringMetric.objects.filter(project_id=project_id, algo='kmeans').order_by('microservice').all()
    ms_mean_shift = ClusteringMetric.objects.filter(project_id=project_id, algo='mean_shift').order_by('microservice').all()
    ms_agglomerative = ClusteringMetric.objects.filter(project_id=project_id, algo='agglomerative').order_by('microservice').all()
    ms_gaussian = ClusteringMetric.objects.filter(project_id=project_id, algo='gaussian').order_by('microservice').all()

    # display page

    data = {
        'project': project,
        'sdmetrics': sdmetric_data,
        'state': state,
        'k': k_value.elbow,
        'kmeans_group': kmeans_group,
        'mshift_group': mshift_group,
        'ms_kmeans': ms_kmeans,
        'ms_mean_shift': ms_mean_shift,
        'ms_agglomerative': ms_agglomerative,
        'agglomerative_group': agglomerative_group,
        'gaussian_group': gaussian_group,
        'ms_gaussian': ms_gaussian,
    }

    return render(request, 'squality/project_cluster_metric.html', data)

def clustering_normalize(request, project_id):
    raw_data = MetricNormalize.objects.filter(project_id=project_id).all().values()
    df = pd.DataFrame(raw_data)
    df_metric = df.iloc[:,2:-2]
    # normalize
    scaler = MinMaxScaler() 
    scaler_feature = scaler.fit_transform(df_metric)
    df_normalize_id = df.iloc[:,0:1].copy()
    df_normalize_metric = pd.DataFrame(scaler_feature)
    df_normalize = pd.concat([df_normalize_id, df_normalize_metric], axis=1)
    df_normalize.columns = ['id','cbo','ic','oc','cam','nco','dit','rfc','loc','nca']
    
    # mydict = {
    #     'df': df.to_html(),
    #     'df_metric': df_metric.to_html(),
    #     'df_normalize': df_normalize.to_html()
    # }

    # update db
    for df_row in df_normalize.index:
        normalize = MetricNormalize.objects.filter(project_id=project_id, id=df_normalize['id'][df_row]).get()
        normalize.cbo = df_normalize['cbo'][df_row]
        normalize.ic = df_normalize['ic'][df_row]
        normalize.oc = df_normalize['oc'][df_row]
        normalize.cam = df_normalize['cam'][df_row]
        normalize.nco = df_normalize['nco'][df_row]
        normalize.dit = df_normalize['dit'][df_row]
        normalize.rfc = df_normalize['rfc'][df_row]
        normalize.loc = df_normalize['loc'][df_row]
        normalize.nca = df_normalize['nca'][df_row]
        normalize.normalized = 1
        normalize.save()

    return redirect('clustering_metric', project_id=project_id)
    # return render(request, 'squality/project_test.html',context=mydict)


def clustering_network(request, project_id):

    GraphImages.objects.filter(project_id=project_id).delete()

    # fix single node
    singles = [] # defaultdict(list)
    sdm = SdMetricRaw.objects.filter(project_id=project_id).all()
    # i = 0
    for sd in sdm :
        # singles[i].append(sd.class_name)
        singles.append(sd.class_name)
        # i += 1

    # print(singles)

    elist = S101MetricRaw.objects.filter(project_id=project_id).all()
    uses_list = set([])
    for el in elist:
        uses_list.add(el.class_from)
        uses_list.add(el.class_to)

    single_nodes = list(set(singles) - uses_list)
    if len(single_nodes) > 0:
        for sn in single_nodes:
            x = S101MetricRaw(
                class_from = sn,
                usage = '-',
                class_to = sn,
                weight = 0,
                project_id = project_id,
                ok_from = 1,
                ok_to = 1
            )
            x.save()


    # init mono network
    df_ref = pd.DataFrame.from_records(SdMetricRaw.objects.filter(project_id=project_id).order_by('class_name').all().values())
    df_ref['node_id'] = range(0, len(df_ref))
    df_s101 = pd.DataFrame.from_records(S101MetricRaw.objects.filter(project_id=project_id).all().values())

    df_raw = df_s101[['class_from','class_to']].copy()
    df_raw['class_from'] = df_raw['class_from'].map(df_ref.set_index('class_name')['node_id'])
    df_raw['class_to'] = df_raw['class_to'].map(df_ref.set_index('class_name')['node_id'])

    np.savetxt(r'uploads/edges/edge_list.txt', df_raw.values, fmt='%d')
    edgelist = Graph.Read_Edgelist("uploads/edges/edge_list.txt", directed=False)

    df_nr = df_ref[['id','node_id','class_name']].copy()
    
    # edgelist = Graph.TupleList(df_raw.itertuples(index=False), directed=False, weights=True)
    # print(edgelist_tmp)

    visual_style = {}
    visual_style['vertex_label'] = list(df_ref['class_name'])
    visual_style['vertex_label_dist'] = 1
    visual_style['bbox'] = (800,800)
    visual_style['margin'] = 50
    visual_style['edge_curved'] = False

    # fast greedy

    try:
        fg_clusters = edgelist.community_fastgreedy().as_clustering()
        # print(fg_clusters)
        # print(fg_clusters_tmp)
        
        fg_pal = igraph.drawing.colors.ClusterColoringPalette(len(fg_clusters))
        edgelist.vs["color"] = fg_pal.get_many(fg_clusters.membership)
        # edgelist.es["curved"] = False
        # edgelist.es['weight'] = list(df_s101['weight'])
        # edgelist.es["label"] = list(df_s101['weight'])
        
        igraph.plot(edgelist, "uploads/csv/fast_greedy.png", **visual_style)

        gi = GraphImages(
            fullname = 'Fast-greedy Community Detection',
            algo = 'fast_greedy',
            fileurl = '/files/fast_greedy.png',
            project_id = project_id
        )
        gi.save()

        # save into db
        if Clustering.objects.filter(project_id=project_id,algo='fast_greedy').count() > 0:
            Clustering.objects.filter(project_id=project_id,algo='fast_greedy').delete()

        i = 0
        for nodes in list(fg_clusters):
            for n in nodes:
                nn = Clustering(
                    class_name = df_nr['class_name'].values[n],
                    cluster = i,
                    type = 'network',
                    algo = 'fast_greedy',
                    project_id = project_id
                )
                nn.save()
            i += 1
        
        # fg_group = Clustering.objects.filter(project_id=project_id,algo='fast_greedy').order_by('cluster').all()

        # TODO: separate as re-usable function? start ---------------------------------------------

        if ClusteringMetric.objects.filter(project_id=project_id,algo='fast_greedy').count() > 0:
            ClusteringMetric.objects.filter(project_id=project_id,algo='fast_greedy').delete()

        ms_ms_grp = defaultdict(list)
        ms_ms_len = Clustering.objects.filter(project_id=project_id,algo='fast_greedy').distinct('cluster').count()
        for i in range(ms_ms_len):
            mloc = 0
            mnoc = 0
            ncam = 0
            imc = 0
            nmo = 0
            cluster_grp = []
            cls = Clustering.objects.filter(project_id=project_id,algo='fast_greedy',cluster=i).all()
            for c in cls:
                cm = SdMetricRaw.objects.filter(project_id=project_id,class_name=c.class_name).get()
                mloc += cm.loc
                mnoc += 1 
                nmo += cm.nco
                ncam += cm.cam
                cluster_grp.append(c.class_name)
                ms_ms_grp[i].append(c.class_name)
            # imc
            for cl in cluster_grp:
                imc_list = S101MetricRaw.objects.filter(project_id=project_id,class_from=cl).all()
                for il in imc_list:
                    # if il.class_to != cl:
                    if ((il.class_to in cluster_grp) and (il.class_to != cl)):
                        imc += il.weight


            ncam = ncam / mnoc
            imc = imc       
            
            fms = ClusteringMetric(
                algo = 'fast_greedy',
                type = 'network',
                microservice = i,
                mloc = mloc,
                mnoc = mnoc,
                ncam = ncam,
                imc = imc,
                nmo = nmo,
                project_id = project_id
            )
            fms.save()

        # wcbm

        for key, val in ms_ms_grp.items():
            ms_wcbm = 0
            ms_trm = 0
            if S101MetricRaw.objects.filter(class_from__in=val, project_id=project_id).count() > 0:
                cf = S101MetricRaw.objects.filter(class_from__in=val, project_id=project_id).all()
                for cc in cf:
                    if cc.class_to not in val:
                        ms_wcbm += cc.weight
                        if cc.usage == 'returns':
                            ms_trm += cc.weight
            ms_x = ClusteringMetric.objects.filter(project_id=project_id, microservice=key, algo='fast_greedy').get()
            ms_x.wcbm = ms_wcbm
            ms_x.trm = ms_trm
            ms_x.save()

        # cbm
        
        for key, val in ms_ms_grp.items():
            ms_cbm = 0
            ms_acbm = 0

            for i in range(ms_ms_len):
                if key != i:
                    if S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_ms_grp[i], project_id=project_id):
                        ms_cbm += 1
                    
                        if S101MetricRaw.objects.filter(class_from__in=ms_ms_grp[i], class_to__in=val, project_id=project_id):
                            ms_from = S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_ms_grp[i], project_id=project_id).all()
                            for mf in ms_from:
                                ms_acbm += mf.weight
                            ms_to = S101MetricRaw.objects.filter(class_from__in=ms_ms_grp[i], class_to__in=val, project_id=project_id).all()
                            for mt in ms_to:
                                ms_acbm += mt.weight

            ms_x = ClusteringMetric.objects.filter(project_id=project_id, microservice=key, algo='fast_greedy').get()
            ms_x.cbm = ms_cbm
            ms_x.acbm = ms_acbm
            ms_x.save()

        # TODO: separate as re-usable function? end ---------------------------------------------
    except Exception:
        pass

    # louvain

    lv_clusters = edgelist.community_multilevel()
    # print(lv_clusters)
    lv_pal = igraph.drawing.colors.ClusterColoringPalette(len(lv_clusters))
    edgelist.vs["color"] = lv_pal.get_many(lv_clusters.membership)
    igraph.plot(edgelist, "uploads/csv/louvain.png", **visual_style)

    gi = GraphImages(
        fullname = 'Louvain Method',
        algo = 'louvain',
        fileurl = '/files/louvain.png',
        project_id = project_id
    )
    gi.save()

    # save into db
    if Clustering.objects.filter(project_id=project_id,algo='louvain').count() > 0:
        Clustering.objects.filter(project_id=project_id,algo='louvain').delete()

    i = 0
    for nodes in list(lv_clusters):
        for n in nodes:
            nn = Clustering(
                class_name = df_nr['class_name'].values[n],
                cluster = i,
                type = 'network',
                algo = 'louvain',
                project_id = project_id
            )
            nn.save()
        i += 1

    # TODO: separate as re-usable function? start ---------------------------------------------

    if ClusteringMetric.objects.filter(project_id=project_id,algo='louvain').count() > 0:
        ClusteringMetric.objects.filter(project_id=project_id,algo='louvain').delete()

    ms_ms_grp = defaultdict(list)
    ms_ms_len = Clustering.objects.filter(project_id=project_id,algo='louvain').distinct('cluster').count()
    for i in range(ms_ms_len):
        mloc = 0
        mnoc = 0
        ncam = 0
        imc = 0
        nmo = 0
        cluster_grp = []
        cls = Clustering.objects.filter(project_id=project_id,algo='louvain',cluster=i).all()
        for c in cls:
            cm = SdMetricRaw.objects.filter(project_id=project_id,class_name=c.class_name).get()
            mloc += cm.loc
            mnoc += 1 
            nmo += cm.nco
            ncam += cm.cam
            cluster_grp.append(c.class_name)
            ms_ms_grp[i].append(c.class_name)
        # imc
        for cl in cluster_grp:
            imc_list = S101MetricRaw.objects.filter(project_id=project_id,class_from=cl).all()
            for il in imc_list:
                # if il.class_to != cl:
                if ((il.class_to in cluster_grp) and (il.class_to != cl)):
                    imc += il.weight


        ncam = ncam / mnoc
        imc = imc       
        
        fms = ClusteringMetric(
            algo = 'louvain',
            type = 'network',
            microservice = i,
            mloc = mloc,
            mnoc = mnoc,
            ncam = ncam,
            imc = imc,
            nmo = nmo,
            project_id = project_id
        )
        fms.save()

    # wcbm

    for key, val in ms_ms_grp.items():
        ms_wcbm = 0
        ms_trm = 0
        if S101MetricRaw.objects.filter(class_from__in=val, project_id=project_id).count() > 0:
            cf = S101MetricRaw.objects.filter(class_from__in=val, project_id=project_id).all()
            for cc in cf:
                if cc.class_to not in val:
                    ms_wcbm += cc.weight
                    if cc.usage == 'returns':
                        ms_trm += cc.weight
        ms_x = ClusteringMetric.objects.filter(project_id=project_id, microservice=key, algo='louvain').get()
        ms_x.wcbm = ms_wcbm
        ms_x.trm = ms_trm
        ms_x.save()

    # cbm
    
    for key, val in ms_ms_grp.items():
        ms_cbm = 0
        ms_acbm = 0
 
        for i in range(ms_ms_len):
            if key != i:
                if S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_ms_grp[i], project_id=project_id):
                    ms_cbm += 1
                   
                    if S101MetricRaw.objects.filter(class_from__in=ms_ms_grp[i], class_to__in=val, project_id=project_id):
                        ms_from = S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_ms_grp[i], project_id=project_id).all()
                        for mf in ms_from:
                            ms_acbm += mf.weight
                        ms_to = S101MetricRaw.objects.filter(class_from__in=ms_ms_grp[i], class_to__in=val, project_id=project_id).all()
                        for mt in ms_to:
                            ms_acbm += mt.weight

        ms_x = ClusteringMetric.objects.filter(project_id=project_id, microservice=key, algo='louvain').get()
        ms_x.cbm = ms_cbm
        ms_x.acbm = ms_acbm
        ms_x.save()

    # TODO: separate as re-usable function? end ---------------------------------------------

    # leiden

    le_clusters = edgelist.community_leiden(objective_function="modularity")
    le_pal = igraph.drawing.colors.ClusterColoringPalette(len(le_clusters))
    edgelist.vs["color"] = le_pal.get_many(le_clusters.membership)
    igraph.plot(edgelist, "uploads/csv/leiden.png", **visual_style)

    gi = GraphImages(
        fullname = 'Leiden Method',
        algo = 'leiden',
        fileurl = '/files/leiden.png',
        project_id = project_id
    )
    gi.save()

    # save into db
    if Clustering.objects.filter(project_id=project_id,algo='leiden').count() > 0:
        Clustering.objects.filter(project_id=project_id,algo='leiden').delete()

    i = 0
    for nodes in list(lv_clusters):
        for n in nodes:
            nn = Clustering(
                class_name = df_nr['class_name'].values[n],
                cluster = i,
                type = 'network',
                algo = 'leiden',
                project_id = project_id
            )
            nn.save()
        i += 1

    # TODO: separate as re-usable function? start ---------------------------------------------

    if ClusteringMetric.objects.filter(project_id=project_id,algo='leiden').count() > 0:
        ClusteringMetric.objects.filter(project_id=project_id,algo='leiden').delete()

    ms_ms_grp = defaultdict(list)
    ms_ms_len = Clustering.objects.filter(project_id=project_id,algo='leiden').distinct('cluster').count()
    for i in range(ms_ms_len):
        mloc = 0
        mnoc = 0
        ncam = 0
        imc = 0
        nmo = 0
        cluster_grp = []
        cls = Clustering.objects.filter(project_id=project_id,algo='leiden',cluster=i).all()
        for c in cls:
            cm = SdMetricRaw.objects.filter(project_id=project_id,class_name=c.class_name).get()
            mloc += cm.loc
            mnoc += 1 
            nmo += cm.nco
            ncam += cm.cam
            cluster_grp.append(c.class_name)
            ms_ms_grp[i].append(c.class_name)
        # imc
        for cl in cluster_grp:
            imc_list = S101MetricRaw.objects.filter(project_id=project_id,class_from=cl).all()
            for il in imc_list:
                # if il.class_to != cl:
                if ((il.class_to in cluster_grp) and (il.class_to != cl)):
                    imc += il.weight


        ncam = ncam / mnoc
        imc = imc       
        
        fms = ClusteringMetric(
            algo = 'leiden',
            type = 'network',
            microservice = i,
            mloc = mloc,
            mnoc = mnoc,
            ncam = ncam,
            imc = imc,
            nmo = nmo,
            project_id = project_id
        )
        fms.save()

    # wcbm

    for key, val in ms_ms_grp.items():
        ms_wcbm = 0
        ms_trm = 0
        if S101MetricRaw.objects.filter(class_from__in=val, project_id=project_id).count() > 0:
            cf = S101MetricRaw.objects.filter(class_from__in=val, project_id=project_id).all()
            for cc in cf:
                if cc.class_to not in val:
                    ms_wcbm += cc.weight
                    if cc.usage == 'returns':
                        ms_trm += cc.weight
        ms_x = ClusteringMetric.objects.filter(project_id=project_id, microservice=key, algo='leiden').get()
        ms_x.wcbm = ms_wcbm
        ms_x.trm = ms_trm
        ms_x.save()

    # cbm
    
    for key, val in ms_ms_grp.items():
        ms_cbm = 0
        ms_acbm = 0
 
        for i in range(ms_ms_len):
            if key != i:
                if S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_ms_grp[i], project_id=project_id):
                    ms_cbm += 1
                   
                    if S101MetricRaw.objects.filter(class_from__in=ms_ms_grp[i], class_to__in=val, project_id=project_id):
                        ms_from = S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_ms_grp[i], project_id=project_id).all()
                        for mf in ms_from:
                            ms_acbm += mf.weight
                        ms_to = S101MetricRaw.objects.filter(class_from__in=ms_ms_grp[i], class_to__in=val, project_id=project_id).all()
                        for mt in ms_to:
                            ms_acbm += mt.weight

        ms_x = ClusteringMetric.objects.filter(project_id=project_id, microservice=key, algo='leiden').get()
        ms_x.cbm = ms_cbm
        ms_x.acbm = ms_acbm
        ms_x.save()

    # TODO: separate as re-usable function? end ---------------------------------------------

    # girvan-newman
 
    try:
        gn_clusters = edgelist.community_edge_betweenness().as_clustering()
        # print(gn_clusters)
        gn_pal = igraph.drawing.colors.ClusterColoringPalette(len(gn_clusters))
        edgelist.vs["color"] = gn_pal.get_many(gn_clusters.membership)
        igraph.plot(edgelist, "uploads/csv/gnewman.png", **visual_style)

        gi = GraphImages(
            fullname = 'Girvan-Newman Betweenness',
            algo = 'gnewman',
            fileurl = '/files/gnewman.png',
            project_id = project_id
        )
        gi.save()

        # save into db
        if Clustering.objects.filter(project_id=project_id,algo='gnewman').count() > 0:
            Clustering.objects.filter(project_id=project_id,algo='gnewman').delete()

        i = 0
        for nodes in list(gn_clusters):
            for n in nodes:
                nn = Clustering(
                    class_name = df_nr['class_name'].values[n],
                    cluster = i,
                    type = 'network',
                    algo = 'gnewman',
                    project_id = project_id
                )
                nn.save()
            i += 1

        # TODO: separate as re-usable function? start ---------------------------------------------

        if ClusteringMetric.objects.filter(project_id=project_id,algo='gnewman').count() > 0:
            ClusteringMetric.objects.filter(project_id=project_id,algo='gnewman').delete()

        ms_ms_grp = defaultdict(list)
        ms_ms_len = Clustering.objects.filter(project_id=project_id,algo='gnewman').distinct('cluster').count()
        for i in range(ms_ms_len):
            mloc = 0
            mnoc = 0
            ncam = 0
            imc = 0
            nmo = 0
            cluster_grp = []
            cls = Clustering.objects.filter(project_id=project_id,algo='gnewman',cluster=i).all()
            for c in cls:
                cm = SdMetricRaw.objects.filter(project_id=project_id,class_name=c.class_name).get()
                mloc += cm.loc
                mnoc += 1 
                nmo += cm.nco
                ncam += cm.cam
                cluster_grp.append(c.class_name)
                ms_ms_grp[i].append(c.class_name)
            # imc
            for cl in cluster_grp:
                imc_list = S101MetricRaw.objects.filter(project_id=project_id,class_from=cl).all()
                for il in imc_list:
                    # if il.class_to != cl:
                    if ((il.class_to in cluster_grp) and (il.class_to != cl)):
                        imc += il.weight


            ncam = ncam / mnoc
            imc = imc       
            
            fms = ClusteringMetric(
                algo = 'gnewman',
                type = 'network',
                microservice = i,
                mloc = mloc,
                mnoc = mnoc,
                ncam = ncam,
                imc = imc,
                nmo = nmo,
                project_id = project_id
            )
            fms.save()

        # wcbm

        for key, val in ms_ms_grp.items():
            ms_wcbm = 0
            ms_trm = 0
            if S101MetricRaw.objects.filter(class_from__in=val, project_id=project_id).count() > 0:
                cf = S101MetricRaw.objects.filter(class_from__in=val, project_id=project_id).all()
                for cc in cf:
                    if cc.class_to not in val:
                        ms_wcbm += cc.weight
                        if cc.usage == 'returns':
                            ms_trm += cc.weight
            ms_x = ClusteringMetric.objects.filter(project_id=project_id, microservice=key, algo='gnewman').get()
            ms_x.wcbm = ms_wcbm
            ms_x.trm = ms_trm
            ms_x.save()

        # cbm
        
        for key, val in ms_ms_grp.items():
            ms_cbm = 0
            ms_acbm = 0
    
            for i in range(ms_ms_len):
                if key != i:
                    if S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_ms_grp[i], project_id=project_id):
                        ms_cbm += 1
                    
                        if S101MetricRaw.objects.filter(class_from__in=ms_ms_grp[i], class_to__in=val, project_id=project_id):
                            ms_from = S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_ms_grp[i], project_id=project_id).all()
                            for mf in ms_from:
                                ms_acbm += mf.weight
                            ms_to = S101MetricRaw.objects.filter(class_from__in=ms_ms_grp[i], class_to__in=val, project_id=project_id).all()
                            for mt in ms_to:
                                ms_acbm += mt.weight

            ms_x = ClusteringMetric.objects.filter(project_id=project_id, microservice=key, algo='gnewman').get()
            ms_x.cbm = ms_cbm
            ms_x.acbm = ms_acbm
            ms_x.save()

        # TODO: separate as re-usable function? end ---------------------------------------------
    except Exception:
        pass

    # main

    project = Project.objects.get(id=project_id)
    graph_images = GraphImages.objects.filter(project_id=project_id).all()
    fastgreedy = ClusteringMetric.objects.filter(project_id=project_id,algo='fast_greedy').order_by('microservice').all()
    louvain = ClusteringMetric.objects.filter(project_id=project_id,algo='louvain').order_by('microservice').all()
    leiden = ClusteringMetric.objects.filter(project_id=project_id,algo='leiden').order_by('microservice').all()
    gnewman = ClusteringMetric.objects.filter(project_id=project_id,algo='gnewman').order_by('microservice').all()

    data = {
        'project': project,
        'graph_images': graph_images,
        'fastgreedy': fastgreedy,
        'louvain': louvain,
        'leiden': leiden,
        'gnewman': gnewman
        # 'df': df_nr.to_html()
    }
    return render(request, 'squality/project_cluster_network.html', data)

def scoring_initialize(request, project_id):
    project = Project.objects.get(id=project_id)

    # k-means

    raw_data = ClusteringMetric.objects.filter(project_id=project_id, algo='kmeans').order_by('microservice').all().values()
    df = pd.DataFrame(raw_data)
    df_metric = df.iloc[:,4:-1]
    # normalize
    scaler = MinMaxScaler() 
    scaler_feature = scaler.fit_transform(df_metric)
    df_normalize_id = df.iloc[:,0:1].copy()
    df_normalize_metric = pd.DataFrame(scaler_feature)
    df_normalize = pd.concat([df_normalize_id, df_normalize_metric], axis=1)
    df_normalize.columns = ['id','cbm','wcbm','acbm','ncam','imc','nmo','trm','mloc','mnoc']

    # update db
    if ClusteringNormalize.objects.filter(project_id=project_id,algo='kmeans').all().count() > 0:
        ClusteringNormalize.objects.filter(project_id=project_id,algo='kmeans').delete()
    
    for df_row in df_normalize.index:
        normalize = ClusteringNormalize(
            microservice = df_row,
            cbm = df_normalize['cbm'][df_row],
            wcbm = df_normalize['wcbm'][df_row],
            acbm = df_normalize['acbm'][df_row],
            ncam = df_normalize['ncam'][df_row],
            imc = df_normalize['imc'][df_row],
            nmo = df_normalize['nmo'][df_row],
            trm = df_normalize['trm'][df_row],
            mloc = df_normalize['mloc'][df_row],
            mnoc = df_normalize['mnoc'][df_row],
            algo = 'kmeans',
            type = 'metric',
            project_id = project_id
        )
        normalize.save()

    # mean shift

    raw_data_ms = ClusteringMetric.objects.filter(project_id=project_id, algo='mean_shift').order_by('microservice').all().values()
    df_ms = pd.DataFrame(raw_data_ms)
    df_metric_ms = df_ms.iloc[:,4:-1]
    # normalize
    scaler = MinMaxScaler() 
    scaler_feature = scaler.fit_transform(df_metric_ms)
    df_normalize_id = df_ms.iloc[:,0:1].copy()
    df_normalize_metric = pd.DataFrame(scaler_feature)
    df_normalize = pd.concat([df_normalize_id, df_normalize_metric], axis=1)
    df_normalize.columns = ['id','cbm','wcbm','acbm','ncam','imc','nmo','trm','mloc','mnoc']

    # update db
    if ClusteringNormalize.objects.filter(project_id=project_id,algo='mean_shift').all().count() > 0:
        ClusteringNormalize.objects.filter(project_id=project_id,algo='mean_shift').delete()
    
    for df_row in df_normalize.index:
        normalize = ClusteringNormalize(
            microservice = df_row,
            cbm = df_normalize['cbm'][df_row],
            wcbm = df_normalize['wcbm'][df_row],
            acbm = df_normalize['acbm'][df_row],
            ncam = df_normalize['ncam'][df_row],
            imc = df_normalize['imc'][df_row],
            nmo = df_normalize['nmo'][df_row],
            trm = df_normalize['trm'][df_row],
            mloc = df_normalize['mloc'][df_row],
            mnoc = df_normalize['mnoc'][df_row],
            algo = 'mean_shift',
            type = 'metric',
            project_id = project_id
        )
        normalize.save()

    # agglomerative

    raw_data_ms = ClusteringMetric.objects.filter(project_id=project_id, algo='agglomerative').order_by('microservice').all().values()
    df_ms = pd.DataFrame(raw_data_ms)
    df_metric_ms = df_ms.iloc[:,4:-1]
    # normalize
    scaler = MinMaxScaler() 
    scaler_feature = scaler.fit_transform(df_metric_ms)
    df_normalize_id = df_ms.iloc[:,0:1].copy()
    df_normalize_metric = pd.DataFrame(scaler_feature)
    df_normalize = pd.concat([df_normalize_id, df_normalize_metric], axis=1)
    df_normalize.columns = ['id','cbm','wcbm','acbm','ncam','imc','nmo','trm','mloc','mnoc']

    # update db
    if ClusteringNormalize.objects.filter(project_id=project_id,algo='agglomerative').all().count() > 0:
        ClusteringNormalize.objects.filter(project_id=project_id,algo='agglomerative').delete()
    
    for df_row in df_normalize.index:
        normalize = ClusteringNormalize(
            microservice = df_row,
            cbm = df_normalize['cbm'][df_row],
            wcbm = df_normalize['wcbm'][df_row],
            acbm = df_normalize['acbm'][df_row],
            ncam = df_normalize['ncam'][df_row],
            imc = df_normalize['imc'][df_row],
            nmo = df_normalize['nmo'][df_row],
            trm = df_normalize['trm'][df_row],
            mloc = df_normalize['mloc'][df_row],
            mnoc = df_normalize['mnoc'][df_row],
            algo = 'agglomerative',
            type = 'metric',
            project_id = project_id
        )
        normalize.save()

    # gaussian mixture

    raw_data_ms = ClusteringMetric.objects.filter(project_id=project_id, algo='gaussian').order_by('microservice').all().values()
    df_ms = pd.DataFrame(raw_data_ms)
    df_metric_ms = df_ms.iloc[:,4:-1]
    # normalize
    scaler = MinMaxScaler() 
    scaler_feature = scaler.fit_transform(df_metric_ms)
    df_normalize_id = df_ms.iloc[:,0:1].copy()
    df_normalize_metric = pd.DataFrame(scaler_feature)
    df_normalize = pd.concat([df_normalize_id, df_normalize_metric], axis=1)
    df_normalize.columns = ['id','cbm','wcbm','acbm','ncam','imc','nmo','trm','mloc','mnoc']

    # update db
    if ClusteringNormalize.objects.filter(project_id=project_id,algo='gaussian').all().count() > 0:
        ClusteringNormalize.objects.filter(project_id=project_id,algo='gaussian').delete()
    
    for df_row in df_normalize.index:
        normalize = ClusteringNormalize(
            microservice = df_row,
            cbm = df_normalize['cbm'][df_row],
            wcbm = df_normalize['wcbm'][df_row],
            acbm = df_normalize['acbm'][df_row],
            ncam = df_normalize['ncam'][df_row],
            imc = df_normalize['imc'][df_row],
            nmo = df_normalize['nmo'][df_row],
            trm = df_normalize['trm'][df_row],
            mloc = df_normalize['mloc'][df_row],
            mnoc = df_normalize['mnoc'][df_row],
            algo = 'gaussian',
            type = 'metric',
            project_id = project_id
        )
        normalize.save()

    # fast-greedy
    if ClusteringMetric.objects.filter(project_id=project_id, algo='fast_greedy').order_by('microservice').all().count() > 0:
        raw_data_ms = ClusteringMetric.objects.filter(project_id=project_id, algo='fast_greedy').order_by('microservice').all().values()
        df_ms = pd.DataFrame(raw_data_ms)
        df_metric_ms = df_ms.iloc[:,4:-1]
        # normalize
        scaler = MinMaxScaler() 
        scaler_feature = scaler.fit_transform(df_metric_ms)
        df_normalize_id = df_ms.iloc[:,0:1].copy()
        df_normalize_metric = pd.DataFrame(scaler_feature)
        df_normalize = pd.concat([df_normalize_id, df_normalize_metric], axis=1)
        df_normalize.columns = ['id','cbm','wcbm','acbm','ncam','imc','nmo','trm','mloc','mnoc']

        # update db
        if ClusteringNormalize.objects.filter(project_id=project_id,algo='fast_greedy').all().count() > 0:
            ClusteringNormalize.objects.filter(project_id=project_id,algo='fast_greedy').delete()
        
        for df_row in df_normalize.index:
            normalize = ClusteringNormalize(
                microservice = df_row,
                cbm = df_normalize['cbm'][df_row],
                wcbm = df_normalize['wcbm'][df_row],
                acbm = df_normalize['acbm'][df_row],
                ncam = df_normalize['ncam'][df_row],
                imc = df_normalize['imc'][df_row],
                nmo = df_normalize['nmo'][df_row],
                trm = df_normalize['trm'][df_row],
                mloc = df_normalize['mloc'][df_row],
                mnoc = df_normalize['mnoc'][df_row],
                algo = 'fast_greedy',
                type = 'network',
                project_id = project_id
            )
            normalize.save()


    # louvain

    raw_data_ms = ClusteringMetric.objects.filter(project_id=project_id, algo='louvain').order_by('microservice').all().values()
    df_ms = pd.DataFrame(raw_data_ms)
    df_metric_ms = df_ms.iloc[:,4:-1]
    # normalize
    scaler = MinMaxScaler() 
    scaler_feature = scaler.fit_transform(df_metric_ms)
    df_normalize_id = df_ms.iloc[:,0:1].copy()
    df_normalize_metric = pd.DataFrame(scaler_feature)
    df_normalize = pd.concat([df_normalize_id, df_normalize_metric], axis=1)
    df_normalize.columns = ['id','cbm','wcbm','acbm','ncam','imc','nmo','trm','mloc','mnoc']

    # update db
    if ClusteringNormalize.objects.filter(project_id=project_id,algo='louvain').all().count() > 0:
        ClusteringNormalize.objects.filter(project_id=project_id,algo='louvain').delete()
    
    for df_row in df_normalize.index:
        normalize = ClusteringNormalize(
            microservice = df_row,
            cbm = df_normalize['cbm'][df_row],
            wcbm = df_normalize['wcbm'][df_row],
            acbm = df_normalize['acbm'][df_row],
            ncam = df_normalize['ncam'][df_row],
            imc = df_normalize['imc'][df_row],
            nmo = df_normalize['nmo'][df_row],
            trm = df_normalize['trm'][df_row],
            mloc = df_normalize['mloc'][df_row],
            mnoc = df_normalize['mnoc'][df_row],
            algo = 'louvain',
            type = 'network',
            project_id = project_id
        )
        normalize.save()

    # leiden

    raw_data_ms = ClusteringMetric.objects.filter(project_id=project_id, algo='leiden').order_by('microservice').all().values()
    df_ms = pd.DataFrame(raw_data_ms)
    df_metric_ms = df_ms.iloc[:,4:-1]
    # normalize
    scaler = MinMaxScaler() 
    scaler_feature = scaler.fit_transform(df_metric_ms)
    df_normalize_id = df_ms.iloc[:,0:1].copy()
    df_normalize_metric = pd.DataFrame(scaler_feature)
    df_normalize = pd.concat([df_normalize_id, df_normalize_metric], axis=1)
    df_normalize.columns = ['id','cbm','wcbm','acbm','ncam','imc','nmo','trm','mloc','mnoc']

    # update db
    if ClusteringNormalize.objects.filter(project_id=project_id,algo='leiden').all().count() > 0:
        ClusteringNormalize.objects.filter(project_id=project_id,algo='leiden').delete()
    
    for df_row in df_normalize.index:
        normalize = ClusteringNormalize(
            microservice = df_row,
            cbm = df_normalize['cbm'][df_row],
            wcbm = df_normalize['wcbm'][df_row],
            acbm = df_normalize['acbm'][df_row],
            ncam = df_normalize['ncam'][df_row],
            imc = df_normalize['imc'][df_row],
            nmo = df_normalize['nmo'][df_row],
            trm = df_normalize['trm'][df_row],
            mloc = df_normalize['mloc'][df_row],
            mnoc = df_normalize['mnoc'][df_row],
            algo = 'leiden',
            type = 'network',
            project_id = project_id
        )
        normalize.save()

    # girvan-newman

    if ClusteringMetric.objects.filter(project_id=project_id, algo='gnewman').order_by('microservice').all().count() > 0:
        raw_data_ms = ClusteringMetric.objects.filter(project_id=project_id, algo='gnewman').order_by('microservice').all().values()
        df_ms = pd.DataFrame(raw_data_ms)
        df_metric_ms = df_ms.iloc[:,4:-1]
        # normalize
        scaler = MinMaxScaler() 
        scaler_feature = scaler.fit_transform(df_metric_ms)
        df_normalize_id = df_ms.iloc[:,0:1].copy()
        df_normalize_metric = pd.DataFrame(scaler_feature)
        df_normalize = pd.concat([df_normalize_id, df_normalize_metric], axis=1)
        df_normalize.columns = ['id','cbm','wcbm','acbm','ncam','imc','nmo','trm','mloc','mnoc']

        # update db
        if ClusteringNormalize.objects.filter(project_id=project_id,algo='gnewman').all().count() > 0:
            ClusteringNormalize.objects.filter(project_id=project_id,algo='gnewman').delete()
        
        for df_row in df_normalize.index:
            normalize = ClusteringNormalize(
                microservice = df_row,
                cbm = df_normalize['cbm'][df_row],
                wcbm = df_normalize['wcbm'][df_row],
                acbm = df_normalize['acbm'][df_row],
                ncam = df_normalize['ncam'][df_row],
                imc = df_normalize['imc'][df_row],
                nmo = df_normalize['nmo'][df_row],
                trm = df_normalize['trm'][df_row],
                mloc = df_normalize['mloc'][df_row],
                mnoc = df_normalize['mnoc'][df_row],
                algo = 'gnewman',
                type = 'network',
                project_id = project_id
            )
            normalize.save()

    return redirect('scoring', project_id=project_id)

def scoring(request, project_id):
    project = Project.objects.get(id=project_id)

    # average scoring

    if ScoringAverage.objects.filter(project_id=project_id).all().count() > 0:
        ScoringAverage.objects.filter(project_id=project_id).delete()

    # k-mean

    ms_kmeans_normalize = ClusteringNormalize.objects.filter(project_id=project_id,algo='kmeans').order_by('microservice').all()

    avg_cbm=0
    avg_wcbm=0
    avg_acbm=0
    avg_ncam=0
    avg_imc=0
    avg_nmo=0
    avg_trm=0
    avg_mloc=0
    avg_mnoc=0
    algo = ''
    type = ''

    for ms in ms_kmeans_normalize:
        avg_cbm += ms.cbm
        avg_wcbm += ms.wcbm
        avg_acbm += ms.acbm
        avg_ncam += ms.ncam
        avg_imc += ms.imc
        avg_nmo += ms.nmo
        avg_trm += ms.trm
        avg_mloc += ms.mloc
        avg_mnoc += ms.mnoc
        algo = ms.algo
        type = ms.type

    avg_ms = ScoringAverage(
        cbm = avg_cbm/len(ms_kmeans_normalize),
        wcbm = avg_wcbm/len(ms_kmeans_normalize),
        acbm = avg_acbm/len(ms_kmeans_normalize),
        ncam = avg_ncam/len(ms_kmeans_normalize),
        imc = avg_imc/len(ms_kmeans_normalize),
        nmo = avg_nmo/len(ms_kmeans_normalize),
        trm = avg_trm/len(ms_kmeans_normalize),
        mloc = avg_mloc/len(ms_kmeans_normalize),
        mnoc = avg_mnoc/len(ms_kmeans_normalize),
        algo = algo,
        type = type,
        project_id = project_id
    )
    avg_ms.save()

    # mean-shift

    ms_mean_shift_normalize = ClusteringNormalize.objects.filter(project_id=project_id,algo='mean_shift').order_by('microservice').all()

    avg_cbm=0
    avg_wcbm=0
    avg_acbm=0
    avg_ncam=0
    avg_imc=0
    avg_nmo=0
    avg_trm=0
    avg_mloc=0
    avg_mnoc=0
    algo = ''
    type = ''

    for ms in ms_mean_shift_normalize:
        avg_cbm += ms.cbm
        avg_wcbm += ms.wcbm
        avg_acbm += ms.acbm
        avg_ncam += ms.ncam
        avg_imc += ms.imc
        avg_nmo += ms.nmo
        avg_trm += ms.trm
        avg_mloc += ms.mloc
        avg_mnoc += ms.mnoc
        algo = ms.algo
        type = ms.type

    avg_ms = ScoringAverage(
        cbm = avg_cbm/len(ms_mean_shift_normalize),
        wcbm = avg_wcbm/len(ms_mean_shift_normalize),
        acbm = avg_acbm/len(ms_mean_shift_normalize),
        ncam = avg_ncam/len(ms_mean_shift_normalize),
        imc = avg_imc/len(ms_mean_shift_normalize),
        nmo = avg_nmo/len(ms_mean_shift_normalize),
        trm = avg_trm/len(ms_mean_shift_normalize),
        mloc = avg_mloc/len(ms_mean_shift_normalize),
        mnoc = avg_mnoc/len(ms_mean_shift_normalize),
        algo = algo,
        type = type,
        project_id = project_id
    )
    avg_ms.save()

    # agglomerative

    ms_agglomerative_normalize = ClusteringNormalize.objects.filter(project_id=project_id,algo='agglomerative').order_by('microservice').all()

    avg_cbm=0
    avg_wcbm=0
    avg_acbm=0
    avg_ncam=0
    avg_imc=0
    avg_nmo=0
    avg_trm=0
    avg_mloc=0
    avg_mnoc=0
    algo = ''
    type = ''

    for ms in ms_agglomerative_normalize:
        avg_cbm += ms.cbm
        avg_wcbm += ms.wcbm
        avg_acbm += ms.acbm
        avg_ncam += ms.ncam
        avg_imc += ms.imc
        avg_nmo += ms.nmo
        avg_trm += ms.trm
        avg_mloc += ms.mloc
        avg_mnoc += ms.mnoc
        algo = ms.algo
        type = ms.type

    avg_ms = ScoringAverage(
        cbm = avg_cbm/len(ms_agglomerative_normalize),
        wcbm = avg_wcbm/len(ms_agglomerative_normalize),
        acbm = avg_acbm/len(ms_agglomerative_normalize),
        ncam = avg_ncam/len(ms_agglomerative_normalize),
        imc = avg_imc/len(ms_agglomerative_normalize),
        nmo = avg_nmo/len(ms_agglomerative_normalize),
        trm = avg_trm/len(ms_agglomerative_normalize),
        mloc = avg_mloc/len(ms_agglomerative_normalize),
        mnoc = avg_mnoc/len(ms_agglomerative_normalize),
        algo = algo,
        type = type,
        project_id = project_id
    )
    avg_ms.save()

    # gaussian-mixture

    ms_gaussian_normalize = ClusteringNormalize.objects.filter(project_id=project_id,algo='gaussian').order_by('microservice').all()

    avg_cbm=0
    avg_wcbm=0
    avg_acbm=0
    avg_ncam=0
    avg_imc=0
    avg_nmo=0
    avg_trm=0
    avg_mloc=0
    avg_mnoc=0
    algo = ''
    type = ''

    for ms in ms_gaussian_normalize:
        avg_cbm += ms.cbm
        avg_wcbm += ms.wcbm
        avg_acbm += ms.acbm
        avg_ncam += ms.ncam
        avg_imc += ms.imc
        avg_nmo += ms.nmo
        avg_trm += ms.trm
        avg_mloc += ms.mloc
        avg_mnoc += ms.mnoc
        algo = ms.algo
        type = ms.type

    avg_ms = ScoringAverage(
        cbm = avg_cbm/len(ms_gaussian_normalize),
        wcbm = avg_wcbm/len(ms_gaussian_normalize),
        acbm = avg_acbm/len(ms_gaussian_normalize),
        ncam = avg_ncam/len(ms_gaussian_normalize),
        imc = avg_imc/len(ms_gaussian_normalize),
        nmo = avg_nmo/len(ms_gaussian_normalize),
        trm = avg_trm/len(ms_gaussian_normalize),
        mloc = avg_mloc/len(ms_gaussian_normalize),
        mnoc = avg_mnoc/len(ms_gaussian_normalize),
        algo = algo,
        type = type,
        project_id = project_id
    )
    avg_ms.save()

    # fast-greedy

    if ClusteringNormalize.objects.filter(project_id=project_id,algo='fast_greedy').order_by('microservice').all().count() > 0:
        ms_fast_greedy_normalize = ClusteringNormalize.objects.filter(project_id=project_id,algo='fast_greedy').order_by('microservice').all()

        avg_cbm=0
        avg_wcbm=0
        avg_acbm=0
        avg_ncam=0
        avg_imc=0
        avg_nmo=0
        avg_trm=0
        avg_mloc=0
        avg_mnoc=0
        algo = ''
        type = ''

        for ms in ms_fast_greedy_normalize:
            avg_cbm += ms.cbm
            avg_wcbm += ms.wcbm
            avg_acbm += ms.acbm
            avg_ncam += ms.ncam
            avg_imc += ms.imc
            avg_nmo += ms.nmo
            avg_trm += ms.trm
            avg_mloc += ms.mloc
            avg_mnoc += ms.mnoc
            algo = ms.algo
            type = ms.type

        avg_ms = ScoringAverage(
            cbm = avg_cbm/len(ms_fast_greedy_normalize),
            wcbm = avg_wcbm/len(ms_fast_greedy_normalize),
            acbm = avg_acbm/len(ms_fast_greedy_normalize),
            ncam = avg_ncam/len(ms_fast_greedy_normalize),
            imc = avg_imc/len(ms_fast_greedy_normalize),
            nmo = avg_nmo/len(ms_fast_greedy_normalize),
            trm = avg_trm/len(ms_fast_greedy_normalize),
            mloc = avg_mloc/len(ms_fast_greedy_normalize),
            mnoc = avg_mnoc/len(ms_fast_greedy_normalize),
            algo = algo,
            type = type,
            project_id = project_id
        )
        avg_ms.save()
    else:
        ms_fast_greedy_normalize = {}

    ms_louvain_normalize = ClusteringNormalize.objects.filter(project_id=project_id,algo='louvain').order_by('microservice').all()

    avg_cbm=0
    avg_wcbm=0
    avg_acbm=0
    avg_ncam=0
    avg_imc=0
    avg_nmo=0
    avg_trm=0
    avg_mloc=0
    avg_mnoc=0
    algo = ''
    type = ''

    for ms in ms_louvain_normalize:
        avg_cbm += ms.cbm
        avg_wcbm += ms.wcbm
        avg_acbm += ms.acbm
        avg_ncam += ms.ncam
        avg_imc += ms.imc
        avg_nmo += ms.nmo
        avg_trm += ms.trm
        avg_mloc += ms.mloc
        avg_mnoc += ms.mnoc
        algo = ms.algo
        type = ms.type

    avg_ms = ScoringAverage(
        cbm = avg_cbm/len(ms_louvain_normalize),
        wcbm = avg_wcbm/len(ms_louvain_normalize),
        acbm = avg_acbm/len(ms_louvain_normalize),
        ncam = avg_ncam/len(ms_louvain_normalize),
        imc = avg_imc/len(ms_louvain_normalize),
        nmo = avg_nmo/len(ms_louvain_normalize),
        trm = avg_trm/len(ms_louvain_normalize),
        mloc = avg_mloc/len(ms_louvain_normalize),
        mnoc = avg_mnoc/len(ms_louvain_normalize),
        algo = algo,
        type = type,
        project_id = project_id
    )
    avg_ms.save()

    ms_leiden_normalize = ClusteringNormalize.objects.filter(project_id=project_id,algo='leiden').order_by('microservice').all()

    avg_cbm=0
    avg_wcbm=0
    avg_acbm=0
    avg_ncam=0
    avg_imc=0
    avg_nmo=0
    avg_trm=0
    avg_mloc=0
    avg_mnoc=0
    algo = ''
    type = ''

    for ms in ms_leiden_normalize:
        avg_cbm += ms.cbm
        avg_wcbm += ms.wcbm
        avg_acbm += ms.acbm
        avg_ncam += ms.ncam
        avg_imc += ms.imc
        avg_nmo += ms.nmo
        avg_trm += ms.trm
        avg_mloc += ms.mloc
        avg_mnoc += ms.mnoc
        algo = ms.algo
        type = ms.type

    avg_ms = ScoringAverage(
        cbm = avg_cbm/len(ms_leiden_normalize),
        wcbm = avg_wcbm/len(ms_leiden_normalize),
        acbm = avg_acbm/len(ms_leiden_normalize),
        ncam = avg_ncam/len(ms_leiden_normalize),
        imc = avg_imc/len(ms_leiden_normalize),
        nmo = avg_nmo/len(ms_leiden_normalize),
        trm = avg_trm/len(ms_leiden_normalize),
        mloc = avg_mloc/len(ms_leiden_normalize),
        mnoc = avg_mnoc/len(ms_leiden_normalize),
        algo = algo,
        type = type,
        project_id = project_id
    )
    avg_ms.save()

    if ClusteringNormalize.objects.filter(project_id=project_id,algo='gnewman').order_by('microservice').all().count() > 0:
        ms_girvan_newman_normalize = ClusteringNormalize.objects.filter(project_id=project_id,algo='gnewman').order_by('microservice').all()

        avg_cbm=0
        avg_wcbm=0
        avg_acbm=0
        avg_ncam=0
        avg_imc=0
        avg_nmo=0
        avg_trm=0
        avg_mloc=0
        avg_mnoc=0
        algo = ''
        type = ''

        for ms in ms_girvan_newman_normalize:
            avg_cbm += ms.cbm
            avg_wcbm += ms.wcbm
            avg_acbm += ms.acbm
            avg_ncam += ms.ncam
            avg_imc += ms.imc
            avg_nmo += ms.nmo
            avg_trm += ms.trm
            avg_mloc += ms.mloc
            avg_mnoc += ms.mnoc
            algo = ms.algo
            type = ms.type

        avg_ms = ScoringAverage(
            cbm = avg_cbm/len(ms_girvan_newman_normalize),
            wcbm = avg_wcbm/len(ms_girvan_newman_normalize),
            acbm = avg_acbm/len(ms_girvan_newman_normalize),
            ncam = avg_ncam/len(ms_girvan_newman_normalize),
            imc = avg_imc/len(ms_girvan_newman_normalize),
            nmo = avg_nmo/len(ms_girvan_newman_normalize),
            trm = avg_trm/len(ms_girvan_newman_normalize),
            mloc = avg_mloc/len(ms_girvan_newman_normalize),
            mnoc = avg_mnoc/len(ms_girvan_newman_normalize),
            algo = algo,
            type = type,
            project_id = project_id
        )
        avg_ms.save()
    else:
        ms_girvan_newman_normalize = {}

    # scoring for metric

    if ScoringFinale.objects.filter(project_id=project_id).all().count() > 0:
        ScoringFinale.objects.filter(project_id=project_id).delete()

    df_metric = pd.DataFrame(ScoringAverage.objects.filter(project_id=project_id,type='metric').all().values())
    df_metric['rank_cbm'] = df_metric['cbm'].rank(ascending=False)
    df_metric['rank_wcbm'] = df_metric['wcbm'].rank(ascending=False)
    df_metric['rank_acbm'] = df_metric['acbm'].rank(ascending=False)
    df_metric['rank_ncam'] = df_metric['ncam'].rank()
    df_metric['rank_imc'] = df_metric['imc'].rank()
    df_metric['rank_nmo'] = df_metric['nmo'].rank(ascending=False)
    df_metric['rank_trm'] = df_metric['trm'].rank(ascending=False)
    df_metric['rank_mloc'] = df_metric['mloc'].rank(ascending=False)
    df_metric['rank_mnoc'] = df_metric['mnoc'].rank(ascending=False)

    df_metric_ranked = df_metric[['algo','type','rank_cbm','rank_wcbm','rank_acbm','rank_ncam','rank_imc','rank_nmo','rank_trm','rank_mloc','rank_mnoc']].copy()

    for df_row in df_metric_ranked.index:
        scoring_finale = ScoringFinale(
            cbm = df_metric_ranked['rank_cbm'][df_row],
            wcbm = df_metric_ranked['rank_wcbm'][df_row],
            acbm = df_metric_ranked['rank_acbm'][df_row],
            ncam = df_metric_ranked['rank_ncam'][df_row],
            imc = df_metric_ranked['rank_imc'][df_row],
            nmo = df_metric_ranked['rank_nmo'][df_row],
            trm = df_metric_ranked['rank_trm'][df_row],
            mloc = df_metric_ranked['rank_mloc'][df_row],
            mnoc = df_metric_ranked['rank_mnoc'][df_row],
            algo = df_metric_ranked['algo'][df_row],
            type = df_metric_ranked['type'][df_row],
            total = df_metric_ranked['rank_cbm'][df_row] + df_metric_ranked['rank_wcbm'][df_row] + df_metric_ranked['rank_acbm'][df_row] + df_metric_ranked['rank_ncam'][df_row]
                        + df_metric_ranked['rank_imc'][df_row] + df_metric_ranked['rank_nmo'][df_row] + df_metric_ranked['rank_trm'][df_row] + df_metric_ranked['rank_mloc'][df_row]
                        + df_metric_ranked['rank_mnoc'][df_row],
            project_id = project_id
        )
        scoring_finale.save()

    # scoring network

    df_network = pd.DataFrame(ScoringAverage.objects.filter(project_id=project_id,type='network').all().values())
    df_network['rank_cbm'] = df_network['cbm'].rank(ascending=False)
    df_network['rank_wcbm'] = df_network['wcbm'].rank(ascending=False)
    df_network['rank_acbm'] = df_network['acbm'].rank(ascending=False)
    df_network['rank_ncam'] = df_network['ncam'].rank()
    df_network['rank_imc'] = df_network['imc'].rank()
    df_network['rank_nmo'] = df_network['nmo'].rank(ascending=False)
    df_network['rank_trm'] = df_network['trm'].rank(ascending=False)
    df_network['rank_mloc'] = df_network['mloc'].rank(ascending=False)
    df_network['rank_mnoc'] = df_network['mnoc'].rank(ascending=False)

    df_network_ranked = df_network[['algo','type','rank_cbm','rank_wcbm','rank_acbm','rank_ncam','rank_imc','rank_nmo','rank_trm','rank_mloc','rank_mnoc']].copy()

    for df_row in df_network_ranked.index:
        scoring_finale = ScoringFinale(
            cbm = df_network_ranked['rank_cbm'][df_row],
            wcbm = df_network_ranked['rank_wcbm'][df_row],
            acbm = df_network_ranked['rank_acbm'][df_row],
            ncam = df_network_ranked['rank_ncam'][df_row],
            imc = df_network_ranked['rank_imc'][df_row],
            nmo = df_network_ranked['rank_nmo'][df_row],
            trm = df_network_ranked['rank_trm'][df_row],
            mloc = df_network_ranked['rank_mloc'][df_row],
            mnoc = df_network_ranked['rank_mnoc'][df_row],
            algo = df_network_ranked['algo'][df_row],
            type = df_network_ranked['type'][df_row],
            total = df_network_ranked['rank_cbm'][df_row] + df_network_ranked['rank_wcbm'][df_row] + df_network_ranked['rank_acbm'][df_row] + df_network_ranked['rank_ncam'][df_row]
                        + df_network_ranked['rank_imc'][df_row] + df_network_ranked['rank_nmo'][df_row] + df_network_ranked['rank_trm'][df_row] + df_network_ranked['rank_mloc'][df_row]
                        + df_network_ranked['rank_mnoc'][df_row],
            project_id = project_id
        )
        scoring_finale.save()

    # get scoring rank
    scoring_metric = ScoringFinale.objects.filter(project_id=project_id,type='metric').order_by('-total').all()
    scoring_network = ScoringFinale.objects.filter(project_id=project_id,type='network').order_by('-total').all()

    data = {
        'project': project,
        'scoring_metric': scoring_metric,
        'scoring_network': scoring_network,
        'ms_kmeans_normalize': ms_kmeans_normalize,
        'ms_mean_shift_normalize': ms_mean_shift_normalize,
        'ms_agglomerative_normalize': ms_agglomerative_normalize,
        'ms_gaussian_normalize': ms_gaussian_normalize,
        'ms_fast_greedy_normalize': ms_fast_greedy_normalize,
        'ms_louvain_normalize': ms_louvain_normalize,
        'ms_leiden_normalize': ms_leiden_normalize,
        'ms_girvan_newman_normalize': ms_girvan_newman_normalize,
        # 'df': df_metric_ranked.to_html()
    }
    return render(request, 'squality/project_scoring.html', data)


def summary(request, project_id):
    project = Project.objects.get(id=project_id)

    # overall scoring

    if ScoringFinale.objects.filter(project_id=project_id, type='overall').all().count() > 0:
        ScoringFinale.objects.filter(project_id=project_id, type='overall').delete()

    df_overall = pd.DataFrame(ScoringAverage.objects.filter(project_id=project_id).all().values())
    df_overall['rank_cbm'] = df_overall['cbm'].rank(ascending=False)
    df_overall['rank_wcbm'] = df_overall['wcbm'].rank(ascending=False)
    df_overall['rank_acbm'] = df_overall['acbm'].rank(ascending=False)
    df_overall['rank_ncam'] = df_overall['ncam'].rank()
    df_overall['rank_imc'] = df_overall['imc'].rank()
    df_overall['rank_nmo'] = df_overall['nmo'].rank(ascending=False)
    df_overall['rank_trm'] = df_overall['trm'].rank(ascending=False)
    df_overall['rank_mloc'] = df_overall['mloc'].rank(ascending=False)
    df_overall['rank_mnoc'] = df_overall['mnoc'].rank(ascending=False)

    df_overall_ranked = df_overall[['algo','type','rank_cbm','rank_wcbm','rank_acbm','rank_ncam','rank_imc','rank_nmo','rank_trm','rank_mloc','rank_mnoc']].copy()

    for df_row in df_overall_ranked.index:
        scoring_finale = ScoringFinale(
            cbm = df_overall_ranked['rank_cbm'][df_row],
            wcbm = df_overall_ranked['rank_wcbm'][df_row],
            acbm = df_overall_ranked['rank_acbm'][df_row],
            ncam = df_overall_ranked['rank_ncam'][df_row],
            imc = df_overall_ranked['rank_imc'][df_row],
            nmo = df_overall_ranked['rank_nmo'][df_row],
            trm = df_overall_ranked['rank_trm'][df_row],
            mloc = df_overall_ranked['rank_mloc'][df_row],
            mnoc = df_overall_ranked['rank_mnoc'][df_row],
            algo = df_overall_ranked['algo'][df_row],
            total = df_overall_ranked['rank_cbm'][df_row] + df_overall_ranked['rank_wcbm'][df_row] + df_overall_ranked['rank_acbm'][df_row] + df_overall_ranked['rank_ncam'][df_row]
                        + df_overall_ranked['rank_imc'][df_row] + df_overall_ranked['rank_nmo'][df_row] + df_overall_ranked['rank_trm'][df_row] + df_overall_ranked['rank_mloc'][df_row]
                        + df_overall_ranked['rank_mnoc'][df_row],
            type = 'overall',
            project_id = project_id
        )
        scoring_finale.save()

    # get scoring rank
    scoring_metric = ScoringFinale.objects.filter(project_id=project_id,type='metric').order_by('-total').all()
    scoring_network = ScoringFinale.objects.filter(project_id=project_id,type='network').order_by('-total').all()
    scoring_overall = ScoringFinale.objects.filter(project_id=project_id,type='overall').order_by('-total').all()

    data = {
        'project': project,
        'scoring_metric': scoring_metric,
        'scoring_network': scoring_network,
        'scoring_overall': scoring_overall,
    }
    return render(request, 'squality/project_summary.html', data)
