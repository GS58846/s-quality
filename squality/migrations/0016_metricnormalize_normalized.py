# Generated by Django 4.0.4 on 2022-04-15 10:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('squality', '0015_alter_metricnormalize_cbo_alter_metricnormalize_dit_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='metricnormalize',
            name='normalized',
            field=models.IntegerField(default=0),
        ),
    ]
