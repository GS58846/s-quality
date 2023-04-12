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
    project_id = models.IntegerField()

class CorpusFile(models.Model):
    filename = models.TextField(unique=True)
    fileurl = models.TextField()
    created_at = models.DateTimeField(auto_now=True)
    processing_time = models.DecimalField(max_digits=12, decimal_places=4, default=0, blank=True, null=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.filename}'

class S101File(models.Model):
    filename = models.TextField(unique=True)
    fileurl = models.TextField()
    created_at = models.DateTimeField(auto_now=True)
    processing_time = models.DecimalField(max_digits=12, decimal_places=4, default=0, blank=True, null=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.filename}'

class CompleteFile(models.Model):
    filename = models.TextField(unique=True)
    fileurl = models.TextField()
    created_at = models.DateTimeField(auto_now=True)
    processing_time = models.DecimalField(max_digits=12, decimal_places=4, default=0, blank=True, null=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.filename}'

class S101MetricRaw(models.Model):
    class_from = models.TextField(null=True)
    usage = models.TextField(null=True)
    class_to = models.TextField(null=True)
    weight = models.IntegerField(default=0)
    ok_from = models.IntegerField(default=0, null=True)
    ok_to = models.IntegerField(default=0, null=True)
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

class ClusteringTime(models.Model):
    project_id = models.IntegerField()
    algo = models.TextField()
    processing_time = models.DecimalField(max_digits=12, decimal_places=4, default=0, blank=True, null=True)
    clustering_time = models.DecimalField(max_digits=12, decimal_places=4, default=0, blank=True, null=True)
    created_at = models.DateTimeField(auto_now=True)

class Clustering(models.Model):
    class_name = models.TextField()
    cluster = models.IntegerField()
    type = models.TextField()
    algo = models.TextField()
    project_id = models.IntegerField()

class ClusteringMetric(models.Model):
    algo = models.TextField()
    type = models.TextField(null=True)
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

class GraphImages(models.Model):
    fullname = models.TextField(null=True)
    algo = models.TextField()
    fileurl = models.TextField()
    updated_at = models.DateTimeField(auto_now=True)
    project_id = models.IntegerField()
    processing_time = models.DecimalField(max_digits=12, decimal_places=4, default=0, blank=True, null=True)

class ScoringAverage(models.Model):
    algo = models.TextField()
    type = models.TextField(null=True)
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

class ScoringMedian(models.Model):
    algo = models.TextField()
    type = models.TextField(null=True)
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

class ScoringFinale(models.Model):
    algo = models.TextField()
    type = models.TextField(null=True)
    cbm = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    wcbm = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    acbm = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    ncam = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    imc = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    nmo = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    trm = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    mloc = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    mnoc = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    mcd = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    total = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    project_id = models.IntegerField()
    coupling = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    cohesion = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    complexity = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    size = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)

class ScoringFinaleMedian(models.Model):
    algo = models.TextField()
    type = models.TextField(null=True)
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
    # total = models.DecimalField(max_digits=9, decimal_places=6, default=0, blank=True, null=True)
    total = models.IntegerField(default=0, blank=True, null=True)
    project_id = models.IntegerField()
    # coupling = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    # cohesion = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    # complexity = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    # size = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)

class ClusteringNormalize(models.Model):
    algo = models.TextField()
    type = models.TextField(null=True)
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

class ScoringFinaleAll(models.Model):
    algo = models.TextField()
    type = models.TextField(null=True)
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
    total = models.DecimalField(max_digits=9, decimal_places=6, default=0, blank=True, null=True)
    project_id = models.IntegerField()
    coupling = models.DecimalField(max_digits=9, decimal_places=6, default=0, blank=True, null=True)
    cohesion = models.DecimalField(max_digits=9, decimal_places=6, default=0, blank=True, null=True)
    complexity = models.DecimalField(max_digits=9, decimal_places=6, default=0, blank=True, null=True)
    size = models.DecimalField(max_digits=9, decimal_places=6, default=0, blank=True, null=True)

class ScoringFinaleAllMedian(models.Model):
    algo = models.TextField()
    type = models.TextField(null=True)
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
    # total = models.DecimalField(max_digits=9, decimal_places=6, default=0, blank=True, null=True)
    total = models.IntegerField(default=0, blank=True, null=True)
    project_id = models.IntegerField()
    # coupling = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    # cohesion = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    # complexity = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)
    # size = models.DecimalField(max_digits=9, decimal_places=2, default=0, blank=True, null=True)