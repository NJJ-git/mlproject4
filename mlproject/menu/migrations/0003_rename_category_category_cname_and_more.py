# Generated by Django 4.1.2 on 2022-10-21 07:59

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('menu', '0002_category_alter_menu_category'),
    ]

    operations = [
        migrations.RenameField(
            model_name='category',
            old_name='category',
            new_name='cName',
        ),
        migrations.RenameField(
            model_name='category',
            old_name='productNo',
            new_name='cateNo',
        ),
    ]
