# Generated by Django 4.0.4 on 2022-04-14 16:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('squality', '0011_clocmetricraw_ok'),
    ]

    operations = [
        migrations.AddField(
            model_name='s101metricraw',
            name='ok',
            field=models.IntegerField(default=0, null=True),
        ),
    ]
