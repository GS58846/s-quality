from django.db import models

# Create your models here.

class Project(models.Model):
    name = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'{self.name}'

# Input

class SdMetric(models.Model):
    filename = models.TextField(unique=True)
    fileurl = models.TextField()
    created_at = models.DateTimeField(auto_now=True)
    # csvfile = models.FileField(upload_to='csv')

    project = models.ForeignKey(Project, on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.filename}'

class ClocMetric(models.Model):
    filename = models.TextField(unique=True)
    fileurl = models.TextField()
    created_at = models.DateTimeField(auto_now=True)
    # csvfile = models.FileField(upload_to='csv')

    project = models.ForeignKey(Project, on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.filename}'


class S101Metric(models.Model):
    filename = models.TextField(unique=True)
    fileurl = models.TextField()
    created_at = models.DateTimeField(auto_now=True)
    # csvfile = models.FileField(upload_to='csv')

    project = models.ForeignKey(Project, on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.filename}'

# Collected metrics

class SdMetricRaw(models.Model):
    class_name = models.TextField()
    cbo = models.IntegerField(default=0, blank=True, null=True)
    ic = models.IntegerField(default=0, blank=True, null=True)
    oc = models.IntegerField(default=0, blank=True, null=True)
    cam = models.DecimalField(max_digits=12, decimal_places=10, default=0, blank=True, null=True)
    nco = models.IntegerField(default=0, blank=True, null=True)
    dit = models.IntegerField(default=0, blank=True, null=True)
    rfc = models.IntegerField(default=0, blank=True, null=True)
    loc = models.IntegerField(default=0, blank=True, null=True)
    nca = models.IntegerField(default=0, blank=True, null=True)
    project_id = models.IntegerField()

class MetricNormalize(models.Model):
    class_name = models.TextField()
    cbo = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    ic = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    oc = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    cam = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    nco = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    dit = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    rfc = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    loc = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    nca = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    normalized = models.IntegerField(default=0)
    project_id = models.IntegerField()

class S101MetricRaw(models.Model):
    class_from = models.TextField(null=True)
    usage = models.TextField(null=True)
    class_to = models.TextField(null=True)
    weight = models.IntegerField(default=0)
    ok_from = models.IntegerField(default=0, null=True)
    ok_to = models.IntegerField(default=0, null=True)
    project_id = models.IntegerField()

class ClocMetricRaw(models.Model):
    class_name = models.TextField()
    comment = models.IntegerField()
    code = models.IntegerField()
    ok = models.IntegerField(default=0, null=True)
    project_id = models.IntegerField()

# Cluster

class Clustering(models.Model):
    class_name = models.TextField()
    cluster = models.IntegerField()
    type = models.TextField()
    algo = models.TextField()
    project_id = models.IntegerField()

class ClusteringMetric(models.Model):
    algo = models.TextField()
    microservice = models.TextField()
    cbm = models.IntegerField(default=0)
    wcbm = models.IntegerField(default=0)
    acbm = models.IntegerField(default=0)
    ncam = models.DecimalField(max_digits=30, decimal_places=6, default=0)
    imc = models.IntegerField(default=0)
    nmo = models.IntegerField(default=0)
    trm = models.IntegerField(default=0)
    mloc = models.IntegerField(default=0)
    mnoc = models.IntegerField(default=0)
    project_id = models.IntegerField()

class NetworkMetric(models.Model):
    source = models.TextField(null=True)
    source_id = models.IntegerField(null=True)
    target = models.TextField(null=True)
    target_id = models.IntegerField(null=True)
    weight = models.IntegerField(null=True)
    algo = models.TextField(null=True)
    project_id = models.IntegerField(null=True)

# class NetworkMetric(models.Model):
#     algo = models.TextField()
#     microservice = models.TextField()
#     cbm = models.IntegerField(default=0)
#     wcbm = models.IntegerField(default=0)
#     acbm = models.IntegerField(default=0)
#     ncam = models.DecimalField(max_digits=30, decimal_places=6, default=0)
#     imc = models.IntegerField(default=0)
#     nmo = models.IntegerField(default=0)
#     trm = models.IntegerField(default=0)
#     mloc = models.IntegerField(default=0)
#     mnoc = models.IntegerField(default=0)
#     project_id = models.IntegerField()

class GraphImages(models.Model):
    algo = models.TextField()
    fileurl = models.TextField()
    updated_at = models.DateTimeField(auto_now=True)
    project_id = models.IntegerField()
