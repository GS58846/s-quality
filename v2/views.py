import random
import string
from os import rename
import os
from csv import DictReader, reader, writer
from django.http import HttpRequest
from django.shortcuts import redirect, render
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from bs4 import BeautifulSoup
from v2.models import Classes, CorpusMetric, Project

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

        corpus = CorpusMetric.objects.filter(project=project)

        completed_file = 0

        if corpus.count() > 0:
            corpus_metric_file = corpus.get()
            completed_file += 1
        else:
            corpus_metric_file = False


        if completed_file != 2:
            btn_state = 'disabled'
        else:
            btn_state = ''


        data = {
            'project': project,
            'corpus_metric_file': corpus_metric_file,
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

        if CorpusMetric.objects.filter(project=project).count() > 0:
            p = CorpusMetric.objects.filter(project=project).get()
            p.filename = new_name
            p.fileurl = file_url
            p.save()

            if Classes.objects.filter(project_id=id).count() > 0:
                Classes.objects.filter(project_id=id).delete()

        else:
            corpus_metric = CorpusMetric(
                filename = new_name,
                fileurl = file_url,
                project = Project.objects.get(id=id)
            )
            corpus_metric.save()
        
        # read saved xml file
        csv_folder = os.path.join(settings.BASE_DIR, 'uploads/csv/' + new_name)
        local_xml = csv_folder

        with open(local_xml, 'r') as f:
            xml_file = f.read()
        
        corpus_file = BeautifulSoup(xml_file, 'xml')
        javas = corpus_file.find_all('Value')

        class_array = []
        for java in javas:
            java_name = java.get('source')
            if java_name is None:
                java_name = ''
            else:
                java_name = java_name.replace('.java','')
            java_pkg = java.get('package')
            if java_pkg is None:
                java_pkg = ''
            java_combine = java_pkg + '.' + java_name
            class_array.append(java_combine)
            
        # print(len(class_array))

        uniq_class = list(dict.fromkeys(class_array))
        # print(len(uniq_class))

        for uc in uniq_class:
            cls = Classes()
            cls.class_name = uc
            cls.project_id = id
            cls.save()

        return redirect('v2_project_import', id=id)
    return redirect('/v2')
