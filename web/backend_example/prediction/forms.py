from django import forms
from django.forms import ModelForm
from .models import PredictionAct, PredictionActMed, PredictionActBase

class PredictionForm(ModelForm):
    class Meta:
        model = PredictionAct
        fields = '__all__'



class PredictionFormMedium(ModelForm):
    class Meta:
        model = PredictionActMed
        fields = '__all__'



class PredictionFormBasic(ModelForm):
    class Meta:
        model = PredictionActBase
        fields = '__all__'



