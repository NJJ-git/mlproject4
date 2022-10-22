from django.db import models


class Category(models.Model):
    cateNo = models.IntegerField(default=0, primary_key=True)
    cName = models.CharField(max_length=20)


class Menu(models.Model):
    productNo = models.IntegerField(default=0, primary_key=True)  # 기본키 설정
    mName = models.CharField(max_length=20)
    mimg_path = models.CharField(max_length=100)
    price = models.IntegerField(default=0)
    explain = models.CharField(max_length=100)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
