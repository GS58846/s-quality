from decimal import Decimal
import math
import statistics
from django.conf import settings
from pyvis.network import Network
import networkx as nx
import pandas as pd
import plotly.express as px
from django.http import HttpRequest
from django.db.models import Q, Sum, Max, Min
from django.shortcuts import redirect, render
from sklearn.base import defaultdict
from sklearn.preprocessing import MinMaxScaler

from eval.models import ClassMetricRaw, Clustering, ClusteringMetric, ClusteringNormalize, MetricNormalize, MsInteractions, Project, S101MetricRaw, ScoringFinaleMedian, ScoringMedian

def index(request):
    projects = Project.objects.order_by('name').all()

    # START overall TOPSIS scoring for evaluation

    column_ids = Project.objects.values_list('id', flat=True)
    id_list = list(column_ids)

    for aid in id_list:

        # TOPSIS method

        # 1. Find Best Ideal Value & Worst Ideal Value for each quality metric (feature)

        ideal_ncam = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('ncam'))
        ideal_ncam = ideal_ncam['max_value']
        worst_ncam = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('ncam'))
        worst_ncam = worst_ncam['min_value']

        ideal_imc = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('imc'))
        ideal_imc = ideal_imc['max_value']
        worst_imc = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('imc'))
        worst_imc = worst_imc['min_value']

        ideal_cbm = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('cbm'))
        ideal_cbm = ideal_cbm['min_value']
        worst_cbm = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('cbm'))
        worst_cbm = worst_cbm['max_value']


        ideal_wcbm = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('wcbm'))
        ideal_wcbm = ideal_wcbm['min_value']
        worst_wcbm = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('wcbm'))
        worst_wcbm = worst_wcbm['max_value']

        ideal_acbm = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('acbm'))
        ideal_acbm = ideal_acbm['min_value']
        worst_acbm = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('acbm'))
        worst_acbm = worst_acbm['max_value']

        ideal_nmo = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('nmo'))
        ideal_nmo = ideal_nmo['min_value']
        worst_nmo = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('nmo'))
        worst_nmo = worst_nmo['max_value']

        ideal_trm = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('trm'))
        ideal_trm = ideal_trm['min_value']
        worst_trm = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('trm'))
        worst_trm = worst_trm['max_value']

        ideal_mloc = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('mloc'))
        ideal_mloc = ideal_mloc['min_value']
        worst_mloc = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('mloc'))
        worst_mloc = worst_mloc['max_value']

        ideal_mnoc = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('mnoc'))
        ideal_mnoc = ideal_mnoc['min_value']
        worst_mnoc = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('mnoc'))
        worst_mnoc = worst_mnoc['max_value']

        ideal_mcd = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('mcd'))
        ideal_mcd = ideal_mcd['min_value']
        worst_mcd = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('mcd'))
        worst_mcd = worst_mcd['max_value']

        # 2. Calculate Euclidean Distance for each algo
        # col_list = ['cbm','wcbm','acbm','ncam','imc','nmo','trm','mloc','mnoc','mcd']
        if ScoringMedian.objects.filter(project_id=aid).count() > 0:
            sm = ScoringMedian.objects.filter(project_id=aid).get()
            # for sm in scoring_median:
            algo_ideal_distance = 0.0
            algo_worst_distance = 0.0
            algo_ideal_distance = Decimal(algo_ideal_distance) + (sm.cbm - ideal_cbm) ** 2
            # print(sm.algo + ' cbm ' + str(sm.cbm) + '-' + str(ideal_cbm) + ' **2')
            algo_ideal_distance += (sm.wcbm - ideal_wcbm) ** 2
            algo_ideal_distance += Decimal((sm.acbm - ideal_acbm) ** 2)
            algo_ideal_distance += Decimal((sm.ncam - ideal_ncam) ** 2)
            algo_ideal_distance += Decimal((sm.imc - ideal_imc) ** 2)
            algo_ideal_distance += Decimal((sm.nmo - ideal_nmo) ** 2)
            algo_ideal_distance += Decimal((sm.trm - ideal_trm) ** 2)
            algo_ideal_distance += Decimal((sm.mloc - ideal_mloc) ** 2)
            algo_ideal_distance += Decimal((sm.mnoc - ideal_mnoc) ** 2)
            algo_ideal_distance += Decimal((sm.mcd - ideal_mcd) ** 2)

            algo_worst_distance = Decimal(algo_worst_distance) + Decimal((sm.cbm - worst_cbm) ** 2)
            algo_worst_distance += Decimal((sm.wcbm - worst_wcbm) ** 2)
            algo_worst_distance += Decimal((sm.acbm - worst_acbm) ** 2)
            algo_worst_distance += Decimal((sm.ncam - worst_ncam) ** 2)
            algo_worst_distance += Decimal((sm.imc - worst_imc) ** 2)
            algo_worst_distance += Decimal((sm.nmo - worst_nmo) ** 2)
            algo_worst_distance += Decimal((sm.trm - worst_trm) ** 2)
            algo_worst_distance += Decimal((sm.mloc - worst_mloc) ** 2)
            algo_worst_distance += Decimal((sm.mnoc - worst_mnoc) ** 2)
            algo_worst_distance += Decimal((sm.mcd - worst_mcd) ** 2)

            # print('algo_ideal_distance ' + str(algo_ideal_distance))
            algo_ideal_distance = math.sqrt(algo_ideal_distance)
            # print('algo_ideal_distance SQRT ' + str(algo_ideal_distance))

            # print('algo_worst_distance ' + str(algo_worst_distance))
            algo_worst_distance = math.sqrt(algo_worst_distance)
            # print('algo_worst_distance SQRT ' + str(algo_worst_distance))

            sm_update = ScoringMedian.objects.filter(project_id=sm.project_id).get()
            sm_update.ideal_d = algo_ideal_distance
            sm_update.worst_d = algo_worst_distance
            sm_update.save()
        

    # 3. Topsis score for each algo
    algos = ScoringMedian.objects.filter(project_id__in=id_list).all()
    for algo in algos:
        topsis_score = algo.worst_d / (algo.ideal_d + algo.worst_d)
        sm = ScoringMedian.objects.filter(project_id=algo.project_id).get()
        sm.topsis_score = topsis_score
        sm.save()

    # scoring_median = ScoringFinaleMedian.objects.filter(id__in=id_list).order_by('-total').all()
    scoring_median = ScoringMedian.objects.filter(project_id__in=id_list).order_by('-topsis_score').all()

    # data = {
    #     'project': project,
    #     'class_metric': cm_res,
    #     'clustering_metric': cluster_metric,
    #     'scoring_median': scoring_median
    # }

    data = {
        'projects': projects,
        'scoring_median': scoring_median
    }

    return render(request, 'eval/index.html', data)

def project_create(request: HttpRequest):
    project = Project( name = request.POST['name'])
    project.save()
    return redirect('/eval')

def project_delete(request, id):
    Project.objects.filter(id=id).delete()
    # ClassMetricRaw.objects.filter(project_id=id).delete()
    # S101MetricRaw.objects.filter(project_id=id).delete()
    # MetricNormalize.objects.filter(project_id=id).delete()
    Clustering.objects.filter(project_id=id).delete()
    ClusteringMetric.objects.filter(project_id=id).delete()
    ClusteringNormalize.objects.filter(project_id=id).delete()

    return redirect('/eval')

def project_assign(request, id):
    project = Project.objects.get(id=id)

    class_metric = MetricNormalize.objects.order_by('class_name').all()

    if Clustering.objects.filter(project_id=id).count() == 0: 
        # copy default classes as template
        for cm in class_metric:
            c = Clustering(
                class_name = cm.class_name,
                cbo = cm.cbo,
                ic = cm.ic,
                oc = cm.oc,
                cam = cm.cam,
                nco = cm.nco,
                dit = cm.dit,
                rfc = cm.rfc,
                loc = cm.loc,
                nca = cm.nca,
                cluster = 0,
                project_id = id
            )
            c.save()

    cm_res = Clustering.objects.filter(project_id=id).order_by('class_name').all()

    # if ClusteringMetric.objects.filter(project_id=id).count() > 0:
    cluster_metric = ClusteringMetric.objects.filter(project_id=id).order_by('microservice').all()

    id_list = [1,2,3,4,5,6,id]

    # scoring_median = ScoringFinaleMedian.objects.filter(project_id__in=id_list).order_by('-total').all()

    for aid in id_list:

        # TOPSIS method

        # 1. Find Best Ideal Value & Worst Ideal Value for each quality metric (feature)

        ideal_ncam = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('ncam'))
        ideal_ncam = ideal_ncam['max_value']
        worst_ncam = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('ncam'))
        worst_ncam = worst_ncam['min_value']

        ideal_imc = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('imc'))
        ideal_imc = ideal_imc['max_value']
        worst_imc = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('imc'))
        worst_imc = worst_imc['min_value']

        ideal_cbm = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('cbm'))
        ideal_cbm = ideal_cbm['min_value']
        worst_cbm = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('cbm'))
        worst_cbm = worst_cbm['max_value']


        ideal_wcbm = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('wcbm'))
        ideal_wcbm = ideal_wcbm['min_value']
        worst_wcbm = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('wcbm'))
        worst_wcbm = worst_wcbm['max_value']

        ideal_acbm = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('acbm'))
        ideal_acbm = ideal_acbm['min_value']
        worst_acbm = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('acbm'))
        worst_acbm = worst_acbm['max_value']

        ideal_nmo = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('nmo'))
        ideal_nmo = ideal_nmo['min_value']
        worst_nmo = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('nmo'))
        worst_nmo = worst_nmo['max_value']

        ideal_trm = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('trm'))
        ideal_trm = ideal_trm['min_value']
        worst_trm = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('trm'))
        worst_trm = worst_trm['max_value']

        ideal_mloc = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('mloc'))
        ideal_mloc = ideal_mloc['min_value']
        worst_mloc = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('mloc'))
        worst_mloc = worst_mloc['max_value']

        ideal_mnoc = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('mnoc'))
        ideal_mnoc = ideal_mnoc['min_value']
        worst_mnoc = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('mnoc'))
        worst_mnoc = worst_mnoc['max_value']

        ideal_mcd = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(min_value=Min('mcd'))
        ideal_mcd = ideal_mcd['min_value']
        worst_mcd = ScoringMedian.objects.filter(project_id__in=id_list).aggregate(max_value=Max('mcd'))
        worst_mcd = worst_mcd['max_value']

        # 2. Calculate Euclidean Distance for each algo
        # col_list = ['cbm','wcbm','acbm','ncam','imc','nmo','trm','mloc','mnoc','mcd']
        if ScoringMedian.objects.filter(project_id=aid).count() > 0:
            sm = ScoringMedian.objects.filter(project_id=aid).get()
            # for sm in scoring_median:
            algo_ideal_distance = 0.0
            algo_worst_distance = 0.0
            algo_ideal_distance = Decimal(algo_ideal_distance) + (sm.cbm - ideal_cbm) ** 2
            # print(sm.algo + ' cbm ' + str(sm.cbm) + '-' + str(ideal_cbm) + ' **2')
            algo_ideal_distance += (sm.wcbm - ideal_wcbm) ** 2
            algo_ideal_distance += Decimal((sm.acbm - ideal_acbm) ** 2)
            algo_ideal_distance += Decimal((sm.ncam - ideal_ncam) ** 2)
            algo_ideal_distance += Decimal((sm.imc - ideal_imc) ** 2)
            algo_ideal_distance += Decimal((sm.nmo - ideal_nmo) ** 2)
            algo_ideal_distance += Decimal((sm.trm - ideal_trm) ** 2)
            algo_ideal_distance += Decimal((sm.mloc - ideal_mloc) ** 2)
            algo_ideal_distance += Decimal((sm.mnoc - ideal_mnoc) ** 2)
            algo_ideal_distance += Decimal((sm.mcd - ideal_mcd) ** 2)

            algo_worst_distance = Decimal(algo_worst_distance) + Decimal((sm.cbm - worst_cbm) ** 2)
            algo_worst_distance += Decimal((sm.wcbm - worst_wcbm) ** 2)
            algo_worst_distance += Decimal((sm.acbm - worst_acbm) ** 2)
            algo_worst_distance += Decimal((sm.ncam - worst_ncam) ** 2)
            algo_worst_distance += Decimal((sm.imc - worst_imc) ** 2)
            algo_worst_distance += Decimal((sm.nmo - worst_nmo) ** 2)
            algo_worst_distance += Decimal((sm.trm - worst_trm) ** 2)
            algo_worst_distance += Decimal((sm.mloc - worst_mloc) ** 2)
            algo_worst_distance += Decimal((sm.mnoc - worst_mnoc) ** 2)
            algo_worst_distance += Decimal((sm.mcd - worst_mcd) ** 2)

            # print('algo_ideal_distance ' + str(algo_ideal_distance))
            algo_ideal_distance = math.sqrt(algo_ideal_distance)
            # print('algo_ideal_distance SQRT ' + str(algo_ideal_distance))

            # print('algo_worst_distance ' + str(algo_worst_distance))
            algo_worst_distance = math.sqrt(algo_worst_distance)
            # print('algo_worst_distance SQRT ' + str(algo_worst_distance))

            sm_update = ScoringMedian.objects.filter(project_id=sm.project_id).get()
            sm_update.ideal_d = algo_ideal_distance
            sm_update.worst_d = algo_worst_distance
            sm_update.save()
        

    # 3. Topsis score for each algo
    algos = ScoringMedian.objects.filter(project_id__in=id_list).all()
    for algo in algos:
        topsis_score = algo.worst_d / (algo.ideal_d + algo.worst_d)
        sm = ScoringMedian.objects.filter(project_id=algo.project_id).get()
        sm.topsis_score = topsis_score
        sm.save()

    # scoring_median = ScoringFinaleMedian.objects.filter(id__in=id_list).order_by('-total').all()
    scoring_median = ScoringMedian.objects.filter(project_id__in=id_list).order_by('-topsis_score').all()

    data = {
        'project': project,
        'class_metric': cm_res,
        'clustering_metric': cluster_metric,
        'scoring_median': scoring_median
    }
    return render(request, 'eval/project_assign.html', data)

def project_cluster(request, id):

    if request.method == 'POST':
        cm_res = Clustering.objects.filter(project_id=id).order_by('class_name').all()
        for cr in cm_res:
            new_val = request.POST.get(cr.class_name)
            # print(cr.class_name + " new value " + str(new_val))
            Clustering.objects.filter(project_id=id, class_name=cr.class_name).update(cluster=new_val)

        # calculation starts

        if ClusteringMetric.objects.filter(project_id=id).count() > 0:
            ClusteringMetric.objects.filter(project_id=id).delete()
        
        sample_sum = 0

        ms_grp = defaultdict(list)
        ms_len = Clustering.objects.filter(project_id=id).distinct('cluster').count()

        sample_mean = 17 / ms_len
        
        for i in range(ms_len):
            mloc = 0
            mnoc = 0
            ncam = 0
            imc = 0
            nmo = 0
            cluster_grp = []
            cls = Clustering.objects.filter(project_id=id,cluster=i).all()
            for c in cls:
                # print(str(i) + "Class Name: " + c.class_name)
                cm = ClassMetricRaw.objects.filter(class_name=c.class_name).get()
                mloc += cm.loc
                mnoc += 1 
                nmo += cm.nco
                ncam += cm.cam
                cluster_grp.append(c.class_name)
                ms_grp[i].append(c.class_name)
            # imc
            for cl in cluster_grp:
                imc_list = S101MetricRaw.objects.filter(class_from=cl).all()
                for il in imc_list:
                    # if il.class_to != cl:
                    if ((il.class_to in cluster_grp) and (il.class_to != cl)):
                        imc += il.weight

            ncam = ncam / mnoc
            imc = imc 
            
            fms = ClusteringMetric(
                microservice = i,
                mloc = mloc,
                mnoc = mnoc,
                ncam = ncam,
                imc = imc,
                nmo = nmo,
                project_id = id
            )
            fms.save()

            # mcd
            sample_sum += (mnoc - sample_mean)**2
           
        sample_variance = sample_sum / (17 - 1)

        sample_std_deviation = math.sqrt(sample_variance)
        
        lower_bound = sample_mean - sample_std_deviation 
        higher_bound = sample_mean + sample_std_deviation
        
        # assigning is_ned based on calculated std_deviation
        ms_ned = ClusteringMetric.objects.filter(project_id=id).all()
        for mn in ms_ned:
            if mn.mnoc <= higher_bound and mn.mnoc >= lower_bound:
                mn.is_ned = 1
                mn.save()

        # wcbm

        for key, val in ms_grp.items():
            ms_wcbm = 0
            ms_trm = 0
            if S101MetricRaw.objects.filter(class_from__in=val).count() > 0:
                cf = S101MetricRaw.objects.filter(class_from__in=val).all()
                for cc in cf:
                    if cc.class_to not in val:
                        ms_wcbm += cc.weight
                        if cc.usage == 'returns':
                            ms_trm += cc.weight
            ms_x = ClusteringMetric.objects.filter(project_id=id, microservice=key).get()
            ms_x.wcbm = ms_wcbm
            ms_x.trm = ms_trm
            ms_x.save()

        # cbm
        
        for key, val in ms_grp.items():
            ms_cbm = 0
            ms_acbm = 0
            
            for i in range(ms_len):
                if key != i:
                    if S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_grp[i]):
                        ms_cbm += 1

                        # inter ms coupling
                        curr_ms = S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_grp[i])
                    
                        if S101MetricRaw.objects.filter(class_from__in=ms_grp[i], class_to__in=val):
                           
                            ms_from = S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_grp[i]).all()
                            for mf in ms_from:
                                ms_acbm += mf.weight
                               
                            ms_to = S101MetricRaw.objects.filter(class_from__in=ms_grp[i], class_to__in=val).all()
                            for mt in ms_to:
                                ms_acbm += mt.weight

            ms_x = ClusteringMetric.objects.filter(project_id=id, microservice=key).get()
            ms_x.cbm = ms_cbm
            ms_x.acbm = ms_acbm
            ms_x.save()

        ##################
        # scoring
        ##################

        raw_data = ClusteringMetric.objects.filter(project_id=id).order_by('microservice').all().values()
        df = pd.DataFrame(raw_data)
        df_metric = df.iloc[:,2:-2]
        # normalize
        scaler = MinMaxScaler() 
        scaler_feature = scaler.fit_transform(df_metric)
        df_normalize_id = df.iloc[:,0:1].copy()
        df_normalize_metric = pd.DataFrame(scaler_feature)
        df_normalize = pd.concat([df_normalize_id, df_normalize_metric], axis=1)
        df_normalize.columns = ['id','cbm','wcbm','acbm','ncam','imc','nmo','trm','mloc','mnoc']

        # update db
        if ClusteringNormalize.objects.filter(project_id=id).all().count() > 0:
            ClusteringNormalize.objects.filter(project_id=id).delete()
        
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
                project_id = id
            )
            normalize.save()

        ##################
        # Scoring median
        ##################

        # calculate mcd ned

        ms_list = ClusteringMetric.objects.filter(project_id=id).all()
        ms_ned = 0
        ms_all = 0
        # print('ms count ' + str(ms_count))
        for msl in ms_list:
            if msl.is_ned == 1:
                ms_ned += msl.mnoc
                ms_all += msl.mnoc
            else:
                ms_all += msl.mnoc

        ned = ms_ned / ms_all
        
        mcd = 1 - ned

        if ScoringMedian.objects.filter(project_id=id).count() > 0:
            ScoringMedian.objects.filter(project_id=id).delete()

        median_cbm = ClusteringNormalize.objects.filter(project_id=id).values_list('cbm', flat=True)
        median_wcbm = ClusteringNormalize.objects.filter(project_id=id).values_list('wcbm', flat=True)
        median_acbm = ClusteringNormalize.objects.filter(project_id=id).values_list('acbm', flat=True)
        median_ncam = ClusteringNormalize.objects.filter(project_id=id).values_list('ncam', flat=True)
        median_imc = ClusteringNormalize.objects.filter(project_id=id).values_list('imc', flat=True)
        median_nmo = ClusteringNormalize.objects.filter(project_id=id).values_list('nmo', flat=True)
        median_trm = ClusteringNormalize.objects.filter(project_id=id).values_list('trm', flat=True)
        median_mloc = ClusteringNormalize.objects.filter(project_id=id).values_list('mloc', flat=True)
        median_mnoc = ClusteringNormalize.objects.filter(project_id=id).values_list('mnoc', flat=True)

        median_ms = ScoringMedian(
            cbm = statistics.median(list(median_cbm)), 
            wcbm = statistics.median(list(median_wcbm)),
            acbm = statistics.median(list(median_acbm)),
            ncam = statistics.median(list(median_ncam)),
            imc = statistics.median(list(median_imc)),
            nmo = statistics.median(list(median_nmo)),
            trm = statistics.median(list(median_trm)),
            mloc = statistics.median(list(median_mloc)),
            mnoc = statistics.median(list(median_mnoc)),
            mcd = mcd,
            project_id = id
        )
        median_ms.save()

        ##################
        # Scoring finale median
        ##################

        # if ScoringFinaleMedian.objects.filter(project_id=id).count() > 0:
        
        # reset rank scoring table
        ScoringFinaleMedian.objects.all().delete()

        # copy data to ScoringFinaleMedian table before process

        id_list = [1,2,3,4,5,6,id]
        # print(id_list)

        # if ScoringMedian.objects.filter(project_id=id).exists():
        data_copy = ScoringMedian.objects.filter(project_id__in=id_list).all()
        # print(data_copy)
        for dc in data_copy:
            # print('------> ' + str(dc.project_id) + " IMC " + str(dc.imc))
            copy_scoring_median = ScoringFinaleMedian(
                cbm = dc.cbm,
                wcbm = dc.wcbm,
                acbm = dc.acbm,
                ncam = dc.ncam,
                imc = dc.imc,
                nmo = dc.nmo,
                trm = dc.trm,
                mloc = dc.mloc,
                mnoc = dc.mnoc,
                mcd = dc.mcd,
                project_id = dc.project_id
            )
            copy_scoring_median.save()

        # all
        df_median = pd.DataFrame.from_records(ScoringFinaleMedian.objects.filter(project_id__in=id_list).all().values())

        # convert to numeric type for manipulation

        neg_metrics = ['cbm','wcbm','acbm','nmo','trm','mloc','mnoc','mcd']

        for nm in neg_metrics:
            df_median[nm] = pd.to_numeric(df_median[nm])
            tmp_metric = df_median.loc[df_median[nm].idxmin()]
            filter_gt = nm + '__gt'
            filter = nm
            search_string = tmp_metric[nm]
            search_none = None
            col_name = nm

            # update score = none
            ScoringFinaleMedian.objects.filter(**{filter_gt:search_string}).update(**{col_name:None})
            # update score =1 
            ScoringFinaleMedian.objects.filter(**{filter:search_string}).update(**{col_name:1})
            # fix None
            ScoringFinaleMedian.objects.filter(**{filter:search_none}).update(**{col_name:0})

        pos_metrics = ['ncam','imc']
        
        for nm in pos_metrics:
            df_median[nm] = pd.to_numeric(df_median[nm])
            tmp_metric = df_median.loc[df_median[nm].idxmax()]
            filter_lt = nm + '__lt'
            filter = nm
            search_string = tmp_metric[nm]
            search_none = None
            col_name = nm
            # update score = none
            ScoringFinaleMedian.objects.filter(**{filter_lt:search_string}).update(**{col_name:None})
            # update score =1 
            ScoringFinaleMedian.objects.filter(**{filter:search_string}).update(**{col_name:1})
            # fix None
            ScoringFinaleMedian.objects.filter(**{filter:search_none}).update(**{col_name:0})

        sfm = ScoringFinaleMedian.objects.filter(project_id__in=id_list).all()
        total = 0
        for x in sfm:
            total = (x.cbm + x.wcbm + x.acbm + x.ncam + x.imc + x.nmo + x.trm + x.mloc + x.mnoc + x.mcd)
            ScoringFinaleMedian.objects.filter(project_id=x.project_id).update(total=total)

    return redirect('eval_project_assign', id=id)

def generate_ms_radar_chart(request, project_id):

    ms_len = ClusteringNormalize.objects.filter(project_id=project_id).count()

    for i in range(ms_len):
        df_metric = pd.DataFrame(ClusteringNormalize.objects.filter(project_id=project_id,microservice=i).order_by('microservice').all().values())
        # print('MS-'+str(i)+' CBM '+ str(df_metric['cbm']))
        r = [
            # float(df_metric['cbm'].to_string(index=False)),
            float(df_metric['cbm']),
            float(df_metric['wcbm']),
            float(df_metric['acbm']),
            float(df_metric['ncam']),
            float(df_metric['imc']),
            float(df_metric['nmo']),
            float(df_metric['trm']),
            float(df_metric['mloc']),
            float(df_metric['mnoc'])]
        t = ['CBM','WCBM','ACBM','NCAM','IMC','NMO','TRM','MLOC','MNOC']

        fig = px.line_polar(df_metric,r=r,theta=t,line_close=True, title='MS-'+str(i))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    range=[0, 1]  # Specify the minimum and maximum values
                )
            )
        )

        fig.update_traces(fill='toself')
        filename = 'eval_ms_radar_' + str(project_id) + '_' + str(i)
        fig.write_image("uploads/csv/radar/" + filename + ".png")

        data = {
            'ms_len': range(ms_len),
            'project_id': project_id
        }

    return render(request, 'eval/ms_radar_chart.html', data)

def generate_ms_diagram(request, project_id):

    if MsInteractions.objects.filter(project_id=project_id).count() > 0:
        MsInteractions.objects.filter(project_id=project_id).delete()

    # query
    ms_grp = defaultdict(list)
    ms_len = Clustering.objects.filter(project_id=project_id).distinct('cluster').count()
    for i in range(ms_len):
        cluster_grp = []
        cls = Clustering.objects.filter(project_id=project_id,cluster=i).all()
        for c in cls:
            cluster_grp.append(c.class_name)
            ms_grp[i].append(c.class_name)
    
    for key, val in ms_grp.items():
        # print(val)
        for i in range(ms_len):
            if key != i:
                if S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_grp[i]):
                    
                    # inter ms coupling
                    sum_coupling = list(S101MetricRaw.objects.filter(class_from__in=val, class_to__in=ms_grp[i]).aggregate(Sum('weight')).values())[0]

                    ms2ms = MsInteractions(
                        ms_from = key,
                        ms_edge = sum_coupling,
                        ms_to = i,
                        project_id = project_id
                    )
                    ms2ms.save()

    # generate diagram
    g = Network(height='700px')

    # add nodes
    for x in range(ms_len):
        mnoc = ClusteringMetric.objects.filter(project_id=project_id, microservice=x).get().mnoc
        ms_name = 'MS-' +str(x);

        # list of classes
        class_list = Clustering.objects.filter(project_id=project_id,cluster=x).order_by('class_name').all()
        class_names = ''
        for cl in class_list:
            class_names = class_names + cl.class_name + '\n'

        g.add_node(x, label=ms_name, title=class_names, shape="dot", value=mnoc, size=mnoc)
    
    ms_interaction = MsInteractions.objects.filter(project_id=project_id).all()
    for msi in ms_interaction:
        g.add_edge(int(msi.ms_from), int(msi.ms_to), value=msi.ms_edge, title=msi.ms_edge, physics=False, smooth=False)

    g.layout = 'circle'

    # Set the layout option
    g.layout_options = {
        "randomSeed": 42,  # Set a random seed for layout consistency
        "hierarchical": True,  # Disable hierarchical layout
    }

    g.save_graph(str(settings.BASE_DIR)+'/v2/templates/v2/pvis_graph_file.html')  

    return render(request, 'v2/ms_diagram.html')