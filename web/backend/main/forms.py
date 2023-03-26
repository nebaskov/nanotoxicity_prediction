from .models import Contact
from django.forms import ModelForm, TextInput


class ContactForm(ModelForm):
    class Meta:
        model = Contact
        fields = '__all__'
        widgets = {
            'Problem': TextInput(attrs={
                'class':'form-control',
            }),
            'Name': TextInput(attrs={
                'class': 'form-control',
            }),
            'Email': TextInput(attrs={
                'class': 'form-control',
            })
        }