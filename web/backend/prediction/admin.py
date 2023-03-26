from django.contrib import admin
from .models import nanoparticle_mri_r1_pred, nanoparticle_mri_r2_pred, nanoparticle_sar_pred
# Register your models here.

admin.site.register(nanoparticle_mri_r1_pred)
admin.site.register(nanoparticle_mri_r2_pred)
admin.site.register(nanoparticle_sar_pred)