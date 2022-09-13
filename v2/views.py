import math
import random
import time
import string
from os import rename
import os
import re
import csv
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
from csv import DictReader, reader, writer
from django.http import HttpRequest
from django.shortcuts import redirect, render
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from bs4 import BeautifulSoup
from v2.models import ClassMetricRaw, Clustering, ClusteringMetric, ClusteringTime, CorpusFile, MetricNormalize, Project, S101File, S101MetricRaw

def index(request):
    projects = Project.objects.all()
    data = {
        'projects': projects,
    }
    return render(request, 'v2/index.html', data)

def project_create(request: HttpRequest):
    project = Project( name = request.POST['name'])
    project.save()
    return redirect('/v2')

def project_import(request, id):
    try:
        project = Project.objects.get(id=id)
        corpus = CorpusFile.objects.filter(project=project)
        s101 = S101File.objects.filter(project=project)

        completed_file = 0

        if corpus.count() > 0:
            corpus_metric_file = corpus.get()
            completed_file += 1
        else:
            corpus_metric_file = False

        if s101.count() > 0:
            s101_metric_file = s101.get()
            completed_file += 1
        else:
            s101_metric_file = False


        if completed_file != 2:
            btn_state = 'disabled'
        else:
            btn_state = ''


        data = {
            'project': project,
            'corpus_metric_file': corpus_metric_file,
            's101_metric_file': s101_metric_file,
            'btn_state': btn_state
        }

        return render(request, 'v2/project_import.html', data)
    except Exception as exc:
        return redirect('/v2')

def corpus_upload(request, id):
    if request.method == 'POST' and request.FILES['cupload']:

        # upload xml file
        upload = request.FILES['cupload']
        fss = FileSystemStorage()
        new_name = 'V2-C-'+''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) + '.xml'
        file = fss.save(new_name, upload)
        file_url = fss.url(file)

        project = Project.objects.get(id=id)

        if CorpusFile.objects.filter(project=project).count() > 0:
            p = CorpusFile.objects.filter(project=project).get()
            p.filename = new_name
            p.fileurl = file_url
            p.save()

            if ClassMetricRaw.objects.filter(project_id=id).count() > 0:
                ClassMetricRaw.objects.filter(project_id=id).delete()

        else:
            corpus_metric = CorpusFile(
                filename = new_name,
                fileurl = file_url,
                project = Project.objects.get(id=id)
            )
            corpus_metric.save()

        # start timer
        st = time.time()
        
        # read saved xml file
        csv_folder = os.path.join(settings.BASE_DIR, 'uploads/csv/' + new_name)
        local_xml = csv_folder

        with open(local_xml, 'r') as f:
            xml_file = f.read()
        
        corpus_file = BeautifulSoup(xml_file, 'xml')
        javas = corpus_file.find_all('Value')

        class_array = []
        for java in javas:
            java_source = java.get('source')

            if java_source is None:
                continue
            else:
                java_name = java_source.replace('.java','')
                java_pkg = java.get('package')
                if java_pkg is None:
                    java_pkg = ''
                java_combine = java_pkg + '.' + java_name
                class_array.append(java_combine)
            
        # print(len(class_array))

        uniq_class = list(dict.fromkeys(class_array))
        # print(len(uniq_class))

        for uc in uniq_class:

            pkg_name = uc.rsplit('.',1)[0]
            cls_name = uc.rsplit('.',1)[1]
            cls_name = cls_name + '.java'

            # line of code

            mloc_array = corpus_file.find_all('Metric', {'id':'MLOC'})
            total_mloc = 0
            for mloc in mloc_array:
                vs = mloc.find_all('Value', {'source':cls_name, 'package':pkg_name})
                for v in vs:
                    total_mloc = total_mloc + int(v.get('value'))

            # no of attributes

            nof_array = corpus_file.find_all('Metric', {'id':'NOF'})
            total_nof = 0
            for nof in nof_array:
                vs = nof.find_all('Value', {'source':cls_name, 'package':pkg_name})
                for v in vs:
                    total_nof = total_nof + int(v.get('value'))

            nsf_array = corpus_file.find_all('Metric', {'id':'NSF'})
            total_nsf = 0
            for nsf in nsf_array:
                vs = nsf.find_all('Value', {'source':cls_name, 'package':pkg_name})
                for v in vs:
                    total_nsf = total_nsf + int(v.get('value'))

            total_noa = total_nof + total_nsf

            # no of class operations

            nom_array = corpus_file.find_all('Metric', {'id':'NOM'})
            total_nom = 0
            for nom in nom_array:
                vs = nom.find_all('Value', {'source':cls_name, 'package':pkg_name})
                for v in vs:
                    total_nom = total_nom + int(v.get('value'))

            # depth of inheritance

            dit_array = corpus_file.find_all('Metric', {'id':'DIT'})
            total_dit = 0
            for dit in dit_array:
                vs = dit.find_all('Value', {'source':cls_name, 'package':pkg_name})
                for v in vs:
                    total_dit = total_dit + int(v.get('value'))

            # afferent coupling / incoming

            # ca_array = corpus_file.find_all('Metric', {'id':'CA'})
            # total_ca = 0
            # for ca in ca_array:
            #     vs = ca.find_all('Value', {'source':cls_name, 'package':pkg_name})
            #     for v in vs:
            #         total_ca = total_ca + int(v.get('value'))

            # efferent coupling / outgoing

            # ce_array = corpus_file.find_all('Metric', {'id':'CE'})
            # total_ce = 0
            # for ce in ce_array:
            #     vs = ce.find_all('Value', {'source':cls_name, 'package':pkg_name})
            #     for v in vs:
            #         total_ce = total_ce + int(v.get('value'))

            # cohesion among method

            lcom_array = corpus_file.find_all('Metric', {'id':'LCOM'})
            total_lcom = 0
            count_lcom = 0
            for lcom in lcom_array:
                vs = lcom.find_all('Value', {'source':cls_name, 'package':pkg_name})
                for v in vs:
                    total_lcom = total_lcom + (1- float(v.get('value')))
                    count_lcom = count_lcom + 1

            avg_cam = total_lcom if count_lcom == 0 else total_lcom / count_lcom

            # save into db

            cls = ClassMetricRaw()
            cls.class_name = uc
            cls.loc = total_mloc
            cls.nca = total_noa
            cls.dit = total_dit
            cls.nco = total_nom
            # cls.ic = total_ca
            # cls.oc = total_ce
            cls.cam = avg_cam
            cls.project_id = id
            cls.save()

        et = time.time()

        if CorpusFile.objects.filter(project=project).count() > 0:
            p = CorpusFile.objects.filter(project=project).get()
            p.processing_time = et - st
            p.save()

        return redirect('v2_project_import', id=id)
    return redirect('/v2')

def s101_upload(request, id):
    if request.method == 'POST' and request.FILES['s101upload']:
        # upload xml file
        upload = request.FILES['s101upload']
        fss = FileSystemStorage()
        new_name = 'V2-S101-'+''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) + '.csv'
        file = fss.save(new_name, upload)
        file_url = fss.url(file)

        project = Project.objects.get(id=id)

        if S101File.objects.filter(project=project).count() > 0:
            p = S101File.objects.filter(project=project).get()
            p.filename = new_name
            p.fileurl = file_url
            p.save()

            if S101MetricRaw.objects.filter(project_id=id).count() > 0:
                S101MetricRaw.objects.filter(project_id=id).delete()

        else:
            corpus_metric = S101File(
                filename = new_name,
                fileurl = file_url,
                project = Project.objects.get(id=id)
            )
            corpus_metric.save()
        
        # read saved csv file
        # base_dir = settings.BASE_DIR
        csv_folder = os.path.join(settings.BASE_DIR, 'uploads/csv/' + new_name)
        local_csv = csv_folder

        st = time.time()

        with open(local_csv, mode='r', encoding="utf-8-sig") as csv_file:
            csv_reader = reader(csv_file)
            for row in csv_reader:
                s101_raw = S101MetricRaw()
                s101_raw.class_from = row[0]
                s101_raw.usage = row[1]
                s101_raw.class_to = row[2]

                s101_raw.weight = re.search(r"(?<=\[)[^][]*(?=])", row[3]).group(0)

                s101_raw.project_id = id
                s101_raw.save()

        et = time.time()

        if S101File.objects.filter(project=project).count() > 0:
            p = S101File.objects.filter(project=project).get()
            p.processing_time = et - st
            p.save()

        return redirect('v2_project_import', id=id)
    return redirect('/v2')

def project_clean(request, id):
    try:
        project = Project.objects.get(id=id)

        class_metric = ClassMetricRaw.objects.order_by('class_name').all().filter(project_id=id)
        
        s101_data_from = S101MetricRaw.objects.order_by('class_from').all().filter(project_id=id).distinct('class_from')
        s101_data_to = S101MetricRaw.objects.order_by('class_to').all().filter(project_id=id).distinct('class_to')
        s101_usages = S101MetricRaw.objects.order_by('usage').all().filter(project_id=id).distinct('usage')

        data = {
            'project': project,
            'class_metric': class_metric,
            's101s_from': s101_data_from,
            's101s_to': s101_data_to,
            's101_usages': s101_usages
        }

        return render(request, 'v2/project_clean.html', data)
    except Exception as exc:
        return redirect('/v2')

def clean_delete(request, class_id):
    sdmetric = ClassMetricRaw.objects.get(id=class_id)
    project_id = sdmetric.project_id
    sdmetric.delete()
    return redirect('v2_project_clean', id=project_id)

def clean_rename(request, project_id, type):
    if type=='class_metric':
        remove_string = request.POST['str_class_metric']
        sdmetrics = ClassMetricRaw.objects.filter(project_id=project_id).all()
    
        for sd in sdmetrics:
            sd.class_name = sd.class_name.replace(remove_string,'')
            sd.save()

    elif type=='s101':
        remove_string = request.POST['str_s101']
        s101 = S101MetricRaw.objects.filter(project_id=project_id).all()

        for s in s101:
            s.class_from = s.class_from.replace(remove_string,'')
            s.class_to = s.class_to.replace(remove_string,'')
            s.save()

    return redirect('v2_project_clean', id=project_id)

def clean_remove_usage(request, project_id, type):
    usage_list = S101MetricRaw.objects.filter(project_id=project_id, usage=type).all()
    usage_list.delete()
    return redirect('v2_project_clean', id=project_id)

def clean_syn_s101(request, project_id):

    # reset as 0 back
    S101MetricRaw.objects.filter(project_id=project_id).update(ok_from=0, ok_to=0)

    # clean from
    sd_classes = ClassMetricRaw.objects.filter(project_id=project_id).all()
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
    sd_classes = ClassMetricRaw.objects.filter(project_id=project_id).all()
    for sd in sd_classes:    
        if S101MetricRaw.objects.filter(class_to=sd.class_name):
            s101to_list = S101MetricRaw.objects.filter(class_to=sd.class_name).all()
            for sft in s101to_list:
                sft.ok_to = 1
                sft.save()
    remove_metric = S101MetricRaw.objects.filter(ok_to=0).all()
    remove_metric.delete()

    sd_classes = ClassMetricRaw.objects.filter(project_id=project_id).all()
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
        # # ic
        # if S101MetricRaw.objects.filter(project_id=project_id, class_to=sd.class_name).count() > 0:
        #     ic_list = S101MetricRaw.objects.filter(project_id=project_id, class_to=sd.class_name).all()
        #     ic_value = 0
        #     for ic in ic_list:
        #         ic_value += ic.weight
        # else:
        #     ic_value = 0
        # sd.ic = ic_value
        # # oc
        # if S101MetricRaw.objects.filter(project_id=project_id, class_from=sd.class_name).count() > 0:
        #     oc_list = S101MetricRaw.objects.filter(project_id=project_id, class_from=sd.class_name).all()
        #     oc_value = 0
        #     for oc in oc_list:
        #         oc_value += oc.weight
        # else:
        #     oc_value = 0
        # sd.oc = oc_value
        # cbo
        if S101MetricRaw.objects.filter(project_id=project_id, class_from=sd.class_name).count() > 0:
            cbo = S101MetricRaw.objects.filter(project_id=project_id, class_from=sd.class_name).all().distinct('class_to').count()
            cbo_value = cbo
        else:
            cbo_value = 0
        sd.cbo = cbo_value 

        sd.save()

    return redirect('v2_project_clean', id=project_id)

def migrate_raw_normalize(request, project_id):
    if MetricNormalize.objects.filter(project_id=project_id).count() > 0:
        MetricNormalize.objects.filter(project_id=project_id).delete()

    raw_data = ClassMetricRaw.objects.filter(project_id=project_id).all()
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
            project_id = project_id,
        )
        normalize_data.save()

    # extract methods for selected classes
    # extract_methods(project_id)

    return redirect('v2_cluster_metric', project_id=project_id)

def view_cluster_metric(request, project_id):

    project = Project.objects.get(id=project_id)
    class_data = MetricNormalize.objects.order_by('class_name').all().filter(project_id=project_id)

    ms_kmeans = ClusteringMetric.objects.filter(project_id=project_id, algo='kmeans').order_by('microservice').all()
    if ClusteringTime.objects.filter(project_id=project_id, algo='kmeans').count() > 0:
        time_kmeans = ClusteringTime.objects.get(project_id=project_id, algo='kmeans')
    else:
        time_kmeans = 0


    data = {
        'project': project,
        'class_metric': class_data,
        'ms_kmeans': ms_kmeans,
        'time_kmeans': time_kmeans,
        'k': len(ms_kmeans)
    }
    
    return render(request, 'v2/project_cluster_metric.html', data)

def clustering_kmeans(request, project_id):

    ######################
    # k-mean
    ######################

    st = time.time()

    raw_data = MetricNormalize.objects.filter(project_id=project_id).all().values()
    df = pd.DataFrame(raw_data)
    df_metric = df.iloc[:,2:-2]
    # df_metric = df[['cbo','ic','oc','cam','nco','dit','rfc','loc','nca','xmi_id']]

    class_count = MetricNormalize.objects.order_by('class_name').filter(project_id=project_id).count()

    print('class count = ' + str(class_count))

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

    # sample mean / average
    sample_mean = class_count / k_value.elbow
    print('sample mean = ' + str(sample_mean))
    
    kmeans_minmax = KMeans(k_value.elbow).fit(df_metric)
    kmeans_clusters = kmeans_minmax.fit_predict(df_metric)

    # df_kmeans = df.iloc[:,1:2].copy()
    df_kmeans = df[['class_name']]
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
            project_id = project_id,
            # xmi_id = df_kmeans['xmi_id'][k]
        )
        c.save()
    
    kmeans_group = Clustering.objects.filter(project_id=project_id,algo='kmeans').order_by('cluster').all()

    # kmeans summary
    # TODO: separate as re-usable function? start ---------------------------------------------

    if ClusteringMetric.objects.filter(project_id=project_id,algo='kmeans').count() > 0:
        ClusteringMetric.objects.filter(project_id=project_id,algo='kmeans').delete()

    sample_sum = 0

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
            cm = ClassMetricRaw.objects.filter(project_id=project_id,class_name=c.class_name).get()
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

        # mcd
        sample_sum += (mnoc - sample_mean)**2
        # print(str(i) + ' sample sum ' + str((mnoc - sample_mean)**2))

    # print('sample sum ' + str(sample_sum))
    sample_variance = sample_sum / (class_count - 1)
    # print('sample_variance ' + str(sample_variance))
    sample_std_deviation = math.sqrt(sample_variance)
    # print('sample std deviation ' + str(sample_std_deviation))
    lower_bound = sample_mean - sample_std_deviation 
    higher_bound = sample_mean + sample_std_deviation
    # print('ned bound ' + str(lower_bound) + ',' + str(higher_bound))
    # print('-------------------------------------')

    # assigning is_ned based on calculated std_deviation
    ms_ned = ClusteringMetric.objects.filter(algo='kmeans', project_id=project_id).all()
    for mn in ms_ned:
        if mn.mnoc <= higher_bound and mn.mnoc >= lower_bound:
            # print('ms ' + str(mn.microservice) + ' is ned')
            mn.is_ned = 1
            mn.save()

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

        ms_x = ClusteringMetric.objects.filter(project_id=project_id, microservice=key, algo='kmeans').get()
        ms_x.cbm = ms_cbm
        ms_x.acbm = ms_acbm
        ms_x.save()

    et = time.time()

    if ClusteringTime.objects.filter(project_id=project_id).count() > 0:
        p = ClusteringTime.objects.filter(project_id=project_id).get()
        p.algo = 'kmeans'
        p.processing_time = et - st
        p.save()

    return redirect('v2_cluster_metric', project_id=project_id)

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

    return redirect('v2_cluster_metric', project_id=project_id)

def clustering_metric(request, project_id):
    project = Project.objects.get(id=project_id)
    sdmetric_data = MetricNormalize.objects.order_by('class_name').all().filter(project_id=project_id)

    # save for export data
    export_metric = 'V2-EXPORT-METRIC.csv'
    csv_folder = os.path.join(settings.BASE_DIR, 'uploads/csv/')
    local_csv = csv_folder
    with open(local_csv+export_metric, 'w', newline='') as f_handle:
        writer = csv.writer(f_handle)
        # Add the header/column names
        header = ['classname','CBO','IC','OC','CAM','NCO','DIT','RFC','LOC','NCA']
        writer.writerow(header)
        # Iterate over `data`  and  write to the csv file
        for sd in sdmetric_data:
            row = [sd.class_name,sd.cbo,sd.ic,sd.oc,sd.cam,sd.nco,sd.dit,sd.rfc,sd.loc,sd.nca]
            writer.writerow(row)

    if MetricNormalize.objects.order_by('class_name').filter(project_id=project_id, normalized=1).count() > 0:
        state = 'disabled'
    else:
        state = ''

    ######################
    # k-mean
    ######################

    raw_data = MetricNormalize.objects.filter(project_id=project_id).all().values()
    df = pd.DataFrame(raw_data)
    df_metric = df.iloc[:,2:-2]
    # df_metric = df[['cbo','ic','oc','cam','nco','dit','rfc','loc','nca','xmi_id']]

    class_count = MetricNormalize.objects.order_by('class_name').filter(project_id=project_id).count()

    # print('class count = ' + str(class_count))

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

    # sample mean / average
    sample_mean = class_count / k_value.elbow
    # print('sample mean = ' + str(sample_mean))
    
    kmeans_minmax = KMeans(k_value.elbow).fit(df_metric)
    kmeans_clusters = kmeans_minmax.fit_predict(df_metric)

    # df_kmeans = df.iloc[:,1:2].copy()
    df_kmeans = df[['class_name','xmi_id']]
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
            project_id = project_id,
        )
        c.save()
    
    kmeans_group = Clustering.objects.filter(project_id=project_id,algo='kmeans').order_by('cluster').all()

    # kmeans summary
    # TODO: separate as re-usable function? start ---------------------------------------------

    if ClusteringMetric.objects.filter(project_id=project_id,algo='kmeans').count() > 0:
        ClusteringMetric.objects.filter(project_id=project_id,algo='kmeans').delete()

    sample_sum = 0

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
            cm = ClassMetricRaw.objects.filter(project_id=project_id,class_name=c.class_name).get()
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

        # mcd
        sample_sum += (mnoc - sample_mean)**2
        # print(str(i) + ' sample sum ' + str((mnoc - sample_mean)**2))

    # print('sample sum ' + str(sample_sum))
    sample_variance = sample_sum / (class_count - 1)
    # print('sample_variance ' + str(sample_variance))
    sample_std_deviation = math.sqrt(sample_variance)
    # print('sample std deviation ' + str(sample_std_deviation))
    lower_bound = sample_mean - sample_std_deviation 
    higher_bound = sample_mean + sample_std_deviation
    # print('ned bound ' + str(lower_bound) + ',' + str(higher_bound))
    # print('-------------------------------------')

    # assigning is_ned based on calculated std_deviation
    ms_ned = ClusteringMetric.objects.filter(algo='kmeans', project_id=project_id).all()
    for mn in ms_ned:
        if mn.mnoc <= higher_bound and mn.mnoc >= lower_bound:
            # print('ms ' + str(mn.microservice) + ' is ned')
            mn.is_ned = 1
            mn.save()

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

    ######################
    # mean-shift
    ######################

    mshift = MeanShift()
    mshift_cluster = mshift.fit_predict(df_metric)
    # df_mshift = df.iloc[:,1:2].copy()
    df_mshift = df[['class_name','xmi_id']]
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
            project_id = project_id,
            xmi_id = df_mshift['xmi_id'][k]
        )
        c.save()
    
    mshift_group = Clustering.objects.filter(project_id=project_id,algo='mean_shift').order_by('cluster').all()

    # mean-shift summary
    # TODO: separate as re-usable function? start ---------------------------------------------

    if ClusteringMetric.objects.filter(project_id=project_id,algo='mean_shift').count() > 0:
        ClusteringMetric.objects.filter(project_id=project_id,algo='mean_shift').delete()

    ms_ms_grp = defaultdict(list)
    ms_ms_len = Clustering.objects.filter(project_id=project_id,algo='mean_shift').distinct('cluster').count()

    sample_nxsum = 0
    sample_mean_nx = class_count / ms_ms_len

    for i in range(ms_ms_len):
        mloc = 0
        mnoc = 0
        ncam = 0
        imc = 0
        nmo = 0
        cluster_grp = []
        cls = Clustering.objects.filter(project_id=project_id,algo='mean_shift',cluster=i).all()
        for c in cls:
            cm = ClassMetricRaw.objects.filter(project_id=project_id,class_name=c.class_name).get()
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

        # mcd
        sample_nxsum += (mnoc - sample_mean_nx)**2

    # print('cluster no ' + str(ms_ms_len))
    # print('sample mean nx ' + str(sample_mean_nx))    
    # print('sample sum nx ' + str(sample_nxsum))
    samplenx_variance = sample_nxsum / (class_count - 1)
    # print('sample_variance nx ' + str(samplenx_variance))
    samplenx_std_deviation = math.sqrt(samplenx_variance)
    # print('sample std deviation nx ' + str(samplenx_std_deviation))
    lower_bound_nx = sample_mean_nx - samplenx_std_deviation 
    higher_bound_nx = sample_mean_nx + samplenx_std_deviation
    # print('ned bound nx ' + str(lower_bound_nx) + ',' + str(higher_bound_nx))
    # print('-------------------------------------')

    # assigning is_ned based on calculated std_deviation
    msnx_ned = ClusteringMetric.objects.filter(algo='mean_shift', project_id=project_id).all()
    for mn in msnx_ned:
        if mn.mnoc <= higher_bound_nx and mn.mnoc >= lower_bound_nx:
            # print('ms ' + str(mn.microservice) + ' is ned')
            mn.is_ned = 1
            mn.save()

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

    ###################
    # agglomerative
    ###################

    agglomerative = AgglomerativeClustering(k_value.elbow)
    agglomerative_cluster = agglomerative.fit_predict(df_metric)
    # df_agglomerative = df.iloc[:,1:2].copy()
    df_agglomerative = df[['class_name','xmi_id']]
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
            project_id = project_id,
            xmi_id = df_agglomerative['xmi_id'][k]
        )
        c.save()
    
    agglomerative_group = Clustering.objects.filter(project_id=project_id,algo='agglomerative').order_by('cluster').all()

    # agglomerative summary
    # TODO: separate as re-usable function? start ---------------------------------------------

    sample_sum = 0

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
            cm = ClassMetricRaw.objects.filter(project_id=project_id,class_name=c.class_name).get()
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

        # mcd
        sample_sum += (mnoc - sample_mean)**2
        # print(str(i) + ' sample sum ' + str((mnoc - sample_mean)**2))

    # print('sample sum ' + str(sample_sum))
    sample_variance = sample_sum / (class_count - 1)
    # print('sample_variance ' + str(sample_variance))
    sample_std_deviation = math.sqrt(sample_variance)
    # print('sample std deviation ' + str(sample_std_deviation))
    lower_bound = sample_mean - sample_std_deviation 
    higher_bound = sample_mean + sample_std_deviation
    # print('ned bound ' + str(lower_bound) + ',' + str(higher_bound))
    # print('-------------------------------------')

    # assigning is_ned based on calculated std_deviation
    ms_ned = ClusteringMetric.objects.filter(algo='agglomerative', project_id=project_id).all()
    for mn in ms_ned:
        if mn.mnoc <= higher_bound and mn.mnoc >= lower_bound:
            # print('ms ' + str(mn.microservice) + ' is ned')
            mn.is_ned = 1
            mn.save()

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

    #######################
    # gaussian mixture
    #######################

    gaussian = GaussianMixture(k_value.elbow)
    gaussian_cluster = gaussian.fit_predict(df_metric)
    # df_gaussian = df.iloc[:,1:2].copy()
    df_gaussian = df[['class_name','xmi_id']]
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
            project_id = project_id,
            xmi_id = df_gaussian['xmi_id'][k]
        )
        c.save()
    
    gaussian_group = Clustering.objects.filter(project_id=project_id,algo='gaussian').order_by('cluster').all()

    # gaussian summary
    # TODO: separate as re-usable function? start ---------------------------------------------

    sample_sum = 0

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
            cm = ClassMetricRaw.objects.filter(project_id=project_id,class_name=c.class_name).get()
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

        # mcd
        sample_sum += (mnoc - sample_mean)**2
        # print(str(i) + ' sample sum ' + str((mnoc - sample_mean)**2))

    # print('sample sum ' + str(sample_sum))
    sample_variance = sample_sum / (class_count - 1)
    # print('sample_variance ' + str(sample_variance))
    sample_std_deviation = math.sqrt(sample_variance)
    # print('sample std deviation ' + str(sample_std_deviation))
    lower_bound = sample_mean - sample_std_deviation 
    higher_bound = sample_mean + sample_std_deviation
    # print('ned bound ' + str(lower_bound) + ',' + str(higher_bound))
    # print('-------------------------------------')

    # assigning is_ned based on calculated std_deviation
    ms_ned = ClusteringMetric.objects.filter(algo='gaussian', project_id=project_id).all()
    for mn in ms_ned:
        if mn.mnoc <= higher_bound and mn.mnoc >= lower_bound:
            # print('ms ' + str(mn.microservice) + ' is ned')
            mn.is_ned = 1
            mn.save()

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
        'export_metric': export_metric
    }

    return render(request, 'v2/project_cluster_metric.html', data)