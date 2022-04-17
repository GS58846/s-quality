from collections import defaultdict
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


from squality.models import ClocMetric, ClocMetricRaw, Clustering, ClusteringMetric, GraphImages, MetricNormalize, NetworkMetric, Project, S101Metric, S101MetricRaw, SdMetric, SdMetricRaw


# Create your views here.

def index(request):
    data = {
        'projects': Project.objects.all()
    }

    return render(request, 'squality/index.html', data)


# PROJECT

def project_create(request: HttpRequest):
    project = Project( name = request.POST['name'])
    project.save()
    return redirect('/squality')

def project_delete(request, id):
    project = Project.objects.get(id=id)
    project.delete()
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
        if ClocMetricRaw.objects.filter(class_name=sd.class_name):
            cloc = ClocMetricRaw.objects.filter(class_name=sd.class_name).get()
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

    ms_kmeans = ClusteringMetric.objects.filter(project_id=project_id, algo='kmeans').order_by('microservice').all()
    ms_mean_shift = ClusteringMetric.objects.filter(project_id=project_id, algo='mean_shift').order_by('microservice').all()

    # display page

    data = {
        'project': project,
        'sdmetrics': sdmetric_data,
        'state': state,
        # 'df': df_kmeans.to_html(),
        'k': k_value.elbow,
        'kmeans_group': kmeans_group,
        'mshift_group': mshift_group,
        'ms_kmeans': ms_kmeans,
        'ms_mean_shift': ms_mean_shift
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

    # fast greedy

    fg_clusters = edgelist.community_fastgreedy().as_clustering()
    # print(fg_clusters)
    # print(fg_clusters_tmp)
    
    fg_pal = igraph.drawing.colors.ClusterColoringPalette(len(fg_clusters))
    edgelist.vs["color"] = fg_pal.get_many(fg_clusters.membership)
    
    visual_style = {}
    visual_style['vertex_label'] = list(df_ref['class_name'])
    visual_style['vertex_label_dist'] = 1
    visual_style['bbox'] = (800,800)
    visual_style['margin'] = 50
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

    if NetworkMetric.objects.filter(project_id=project_id,algo='fast_greedy').count() > 0:
        NetworkMetric.objects.filter(project_id=project_id,algo='fast_greedy').delete()

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
        
        fms = NetworkMetric(
            algo = 'fast_greedy',
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
        ms_x = NetworkMetric.objects.filter(project_id=project_id, microservice=key, algo='fast_greedy').get()
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

        ms_x = NetworkMetric.objects.filter(project_id=project_id, microservice=key, algo='fast_greedy').get()
        ms_x.cbm = ms_cbm
        ms_x.acbm = ms_acbm
        ms_x.save()

    # TODO: separate as re-usable function? end ---------------------------------------------

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

    if NetworkMetric.objects.filter(project_id=project_id,algo='louvain').count() > 0:
        NetworkMetric.objects.filter(project_id=project_id,algo='louvain').delete()

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
        
        fms = NetworkMetric(
            algo = 'louvain',
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
        ms_x = NetworkMetric.objects.filter(project_id=project_id, microservice=key, algo='louvain').get()
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

        ms_x = NetworkMetric.objects.filter(project_id=project_id, microservice=key, algo='louvain').get()
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

    if NetworkMetric.objects.filter(project_id=project_id,algo='leiden').count() > 0:
        NetworkMetric.objects.filter(project_id=project_id,algo='leiden').delete()

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
        
        fms = NetworkMetric(
            algo = 'leiden',
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
        ms_x = NetworkMetric.objects.filter(project_id=project_id, microservice=key, algo='leiden').get()
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

        ms_x = NetworkMetric.objects.filter(project_id=project_id, microservice=key, algo='leiden').get()
        ms_x.cbm = ms_cbm
        ms_x.acbm = ms_acbm
        ms_x.save()

    # TODO: separate as re-usable function? end ---------------------------------------------

    # main

    project = Project.objects.get(id=project_id)
    graph_images = GraphImages.objects.filter(project_id=project_id).all()
    fastgreedy = NetworkMetric.objects.filter(project_id=project_id,algo='fast_greedy').order_by('microservice').all()
    louvain = NetworkMetric.objects.filter(project_id=project_id,algo='louvain').order_by('microservice').all()
    leiden = NetworkMetric.objects.filter(project_id=project_id,algo='leiden').order_by('microservice').all()

    data = {
        'project': project,
        'graph_images': graph_images,
        'fastgreedy': fastgreedy,
        'louvain': louvain,
        'leiden': leiden
        # 'df': df_nr.to_html()
    }
    return render(request, 'squality/project_cluster_network.html', data)
    
