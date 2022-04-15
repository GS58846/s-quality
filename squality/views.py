import base64, urllib
from csv import DictReader, reader
from fileinput import filename
from functools import reduce
import io
from os import rename
import os
import random
import re
from statistics import mode
import string
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
from sklearn.metrics import silhouette_score


from squality.models import ClocMetric, ClocMetricRaw, Clustering, MetricNormalize, Project, S101Metric, S101MetricRaw, SdMetric, SdMetricRaw


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

    data = {
        'project': project,
        'sdmetrics': sdmetric_data,
        'state': state,
        # 'df': df_kmeans.to_html(),
        'k': k_value.elbow,
        'kmeans_group': kmeans_group,
        'mshift_group': mshift_group
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
    
    mydict = {
        'df': df.to_html(),
        'df_metric': df_metric.to_html(),
        'df_normalize': df_normalize.to_html()
    }

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


