from django import forms
from .models import *

class ResgistrationForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = [
            'face_id',
            'name',
            'address',
            'job',
            'phone',
            'email'
        ]
        labels = {
            'face_id': '아이디(전화번호)',
            'name': '이름',
            'address': '주소',
            'job': '직업',
            'phone': '전화번호',
            'email': '이메일'
        }  