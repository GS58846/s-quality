# Generated by Django 4.0.4 on 2022-09-09 00:57

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('v2', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='CorpusMetric',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('filename', models.TextField(unique=True)),
                ('fileurl', models.TextField()),
                ('created_at', models.DateTimeField(auto_now=True)),
                ('project', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='v2.project')),
            ],
        ),
    ]
