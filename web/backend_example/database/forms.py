from .models import offer
from django.forms import ModelForm

class offerForm(ModelForm):
    class Meta:
        model = offer
        fields = '__all__'