# Generated by Django 4.0.4 on 2022-04-15 10:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('squality', '0014_metricnormalize'),
    ]

    operations = [
        migrations.AlterField(
            model_name='metricnormalize',
            name='cbo',
            field=models.DecimalField(blank=True, decimal_places=10, default=0, max_digits=12, null=True),
        ),
        migrations.AlterField(
            model_name='metricnormalize',
            name='dit',
            field=models.DecimalField(blank=True, decimal_places=10, default=0, max_digits=12, null=True),
        ),
        migrations.AlterField(
            model_name='metricnormalize',
            name='ic',
            field=models.DecimalField(blank=True, decimal_places=10, default=0, max_digits=12, null=True),
        ),
        migrations.AlterField(
            model_name='metricnormalize',
            name='loc',
            field=models.DecimalField(blank=True, decimal_places=10, default=0, max_digits=12, null=True),
        ),
        migrations.AlterField(
            model_name='metricnormalize',
            name='nca',
            field=models.DecimalField(blank=True, decimal_places=10, default=0, max_digits=12, null=True),
        ),
        migrations.AlterField(
            model_name='metricnormalize',
            name='nco',
            field=models.DecimalField(blank=True, decimal_places=10, default=0, max_digits=12, null=True),
        ),
        migrations.AlterField(
            model_name='metricnormalize',
            name='oc',
            field=models.DecimalField(blank=True, decimal_places=10, default=0, max_digits=12, null=True),
        ),
        migrations.AlterField(
            model_name='metricnormalize',
            name='rfc',
            field=models.DecimalField(blank=True, decimal_places=10, default=0, max_digits=12, null=True),
        ),
    ]
