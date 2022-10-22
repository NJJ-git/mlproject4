from django.db import models

# Create your models here.

class Category(models.Model):
    productNo = models.IntegerField(default=0,primary_key=True)
    category= models.CharField(max_length=20)
    

class Menu(models.Model) :
    productNo = models.IntegerField(default=0,primary_key=True)  #기본키 설정 
    mName = models.CharField(max_length=20)
    mimg_path = models.CharField(max_length=100)
    price = models.IntegerField(default=0)
    explain = models.CharField(max_length=100)
    category= models.ForeignKey(Category, on_delete = models.CASCADE)

    def __str__(self):
        return self.productNo, self.mName, self.mimg_path, self.price, self.explain, self.category