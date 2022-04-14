from csv import DictReader, reader
from fileinput import filename
from functools import reduce
from os import rename
import os
import random
import re
from statistics import mode
import string
from webbrowser import get
from django.http import HttpRequest
from django.shortcuts import redirect, render
from django.core.files.storage import FileSystemStorage
from django.conf import settings


from squality.models import ClocMetric, ClocMetricRaw, Project, S101Metric, S101MetricRaw, SdMetric, SdMetricRaw


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
                # sdmetric_raw.dit = row['DIT']
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

        if(ClocMetric.objects.filter(project=project).count() > 0):
            p = ClocMetric.objects.filter(project=project).get()
            p.filename = new_name
            p.fileurl = file_url
            p.save()

            if ClocMetricRaw.objects.filter(project_id=id).count() > 0:
                ClocMetricRaw.objects.filter(project_id=id).delete()

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
    project_id = sdmetric.id
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
            s.class_to = s.class_from.replace(remove_string,'')
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
