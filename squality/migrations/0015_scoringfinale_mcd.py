# Generated by Django 4.0.4 on 2022-05-25 14:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('squality', '0014_scoringaverage_mcd'),
    ]

    operations = [
        migrations.AddField(
            model_name='scoringfinale',
            name='mcd',
            field=models.IntegerField(blank=True, default=0, null=True),
        ),
    ]
