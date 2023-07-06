from django.db import models

class Project(models.Model):
    name = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'{self.name}'

class ClassMetricRaw(models.Model):
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

class S101MetricRaw(models.Model):
    class_from = models.TextField(null=True)
    usage = models.TextField(null=True)
    class_to = models.TextField(null=True)
    weight = models.IntegerField(default=0)
    ok_from = models.IntegerField(default=0, null=True)
    ok_to = models.IntegerField(default=0, null=True)

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

class Clustering(models.Model):
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
    cluster = models.IntegerField()
    project_id = models.IntegerField()

class ClusteringMetric(models.Model):
    microservice = models.IntegerField()
    cbm = models.IntegerField(default=0)
    wcbm = models.IntegerField(default=0)
    acbm = models.IntegerField(default=0)
    ncam = models.DecimalField(max_digits=30, decimal_places=6, default=0)
    imc = models.IntegerField(default=0)
    nmo = models.IntegerField(default=0)
    trm = models.IntegerField(default=0)
    mloc = models.IntegerField(default=0)
    mnoc = models.IntegerField(default=0)
    is_ned = models.IntegerField(default=0)
    project_id = models.IntegerField()

class ClusteringNormalize(models.Model):
    microservice = models.IntegerField()
    cbm = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    wcbm = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    acbm = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    ncam = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    imc = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    nmo = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    trm = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    mloc = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    mnoc = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    project_id = models.IntegerField()

class ScoringMedian(models.Model):
    cbm = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    wcbm = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    acbm = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    ncam = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    imc = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    nmo = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    trm = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    mloc = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    mnoc = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    mcd = models.DecimalField(max_digits=30, decimal_places=6, default=0, blank=True, null=True)
    project_id = models.IntegerField()

class ScoringFinaleMedian(models.Model):
    cbm = models.DecimalField(max_digits=9, decimal_places=6, default=0, blank=True, null=True)
    wcbm = models.DecimalField(max_digits=9, decimal_places=6, default=0, blank=True, null=True)
    acbm = models.DecimalField(max_digits=9, decimal_places=6, default=0, blank=True, null=True)
    ncam = models.DecimalField(max_digits=9, decimal_places=6, default=0, blank=True, null=True)
    imc = models.DecimalField(max_digits=9, decimal_places=6, default=0, blank=True, null=True)
    nmo = models.DecimalField(max_digits=9, decimal_places=6, default=0, blank=True, null=True)
    trm = models.DecimalField(max_digits=9, decimal_places=6, default=0, blank=True, null=True)
    mloc = models.DecimalField(max_digits=9, decimal_places=6, default=0, blank=True, null=True)
    mnoc = models.DecimalField(max_digits=9, decimal_places=6, default=0, blank=True, null=True)
    mcd = models.DecimalField(max_digits=9, decimal_places=6, default=0, blank=True, null=True)
    total = models.IntegerField(default=0, blank=True, null=True)
    project_id = models.IntegerField()

class MsInteractions(models.Model):
    ms_from = models.TextField()
    ms_edge = models.IntegerField()
    ms_to = models.TextField()
    project_id = models.IntegerField()