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

class S101MetricRaw(models.Model):
    class_from = models.TextField()
    usage = models.TextField()
    class_to = models.TextField()
    weight = models.IntegerField(default=0)
    ok = models.IntegerField(default=0, null=True)
    project_id = models.IntegerField()

class ClocMetricRaw(models.Model):
    class_name = models.TextField()
    comment = models.IntegerField()
    code = models.IntegerField()
    ok = models.IntegerField(default=0, null=True)
    project_id = models.IntegerField()
