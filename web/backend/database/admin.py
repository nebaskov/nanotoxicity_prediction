from django.contrib import admin
from .models import nanoparticle_mri_r1, nanoparticle_mri_r2, nanoparticle_sar, offer

# Register your models here.

admin.site.register(nanoparticle_mri_r1)
admin.site.register(nanoparticle_mri_r2)
admin.site.register(nanoparticle_sar)
admin.site.register(offer)