from django.shortcuts import render, redirect
from .forms import ContactForm
from django.http import HttpResponseNotFound


def index(request):
    return render(request, 'main/index.html')


def contact(request):
    error = ' '
    if request.method == 'POST':
        form = ContactForm(request.POST or None)
        if form.is_valid():
            form.save()
            return redirect('home')
    form = ContactForm
    data = {
        'form': form,
        'error': error
    }
    return render(request, 'main/contact.html', data)


def info(request):
    return render(request, 'main/info.html')


def pageNotFound(request, exception):
    return HttpResponseNotFound('<h1>Page not found</h1>')

def serverError(request):
    return HttpResponseNotFound('<h1>In the database there are no element that is in the inputted formula '
                                'so the model cannot calculate some of descriptors. Please, offer your sample instead</h1>')