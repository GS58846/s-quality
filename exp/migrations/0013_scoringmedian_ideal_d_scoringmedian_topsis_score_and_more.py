# Generated by Django 4.0.4 on 2023-09-22 21:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('exp', '0012_msinteractions'),
    ]

    operations = [
        migrations.AddField(
            model_name='scoringmedian',
            name='ideal_d',
            field=models.DecimalField(blank=True, decimal_places=6, default=0, max_digits=30, null=True),
        ),
        migrations.AddField(
            model_name='scoringmedian',
            name='topsis_score',
            field=models.DecimalField(blank=True, decimal_places=6, default=0, max_digits=30, null=True),
        ),
        migrations.AddField(
            model_name='scoringmedian',
            name='worst_d',
            field=models.DecimalField(blank=True, decimal_places=6, default=0, max_digits=30, null=True),
        ),
    ]
