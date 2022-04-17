# Generated by Django 4.0.4 on 2022-04-18 00:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('squality', '0007_clusternormalize'),
    ]

    operations = [
        migrations.CreateModel(
            name='ClusteringNormalize',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('algo', models.TextField()),
                ('type', models.TextField(null=True)),
                ('microservice', models.TextField()),
                ('cbm', models.DecimalField(blank=True, decimal_places=6, default=0, max_digits=30, null=True)),
                ('wcbm', models.DecimalField(blank=True, decimal_places=6, default=0, max_digits=30, null=True)),
                ('acbm', models.DecimalField(blank=True, decimal_places=6, default=0, max_digits=30, null=True)),
                ('ncam', models.DecimalField(blank=True, decimal_places=6, default=0, max_digits=30, null=True)),
                ('imc', models.DecimalField(blank=True, decimal_places=6, default=0, max_digits=30, null=True)),
                ('nmo', models.DecimalField(blank=True, decimal_places=6, default=0, max_digits=30, null=True)),
                ('trm', models.DecimalField(blank=True, decimal_places=6, default=0, max_digits=30, null=True)),
                ('mloc', models.DecimalField(blank=True, decimal_places=6, default=0, max_digits=30, null=True)),
                ('mnoc', models.DecimalField(blank=True, decimal_places=6, default=0, max_digits=30, null=True)),
                ('project_id', models.IntegerField()),
            ],
        ),
        migrations.DeleteModel(
            name='ClusterNormalize',
        ),
    ]
