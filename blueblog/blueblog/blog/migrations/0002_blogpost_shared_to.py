# Generated by Django 2.0 on 2018-01-29 22:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='blogpost',
            name='shared_to',
            field=models.ManyToManyField(related_name='shared_posts', to='blog.Blog'),
        ),
    ]
