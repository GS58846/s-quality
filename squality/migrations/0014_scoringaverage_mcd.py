# Generated by Django 4.0.4 on 2022-05-25 14:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('squality', '0013_clusteringmetric_is_ned'),
    ]

    operations = [
        migrations.AddField(
            model_name='scoringaverage',
            name='mcd',
            field=models.DecimalField(blank=True, decimal_places=6, default=0, max_digits=30, null=True),
        ),
    ]
