from django.db import models


class PredictionAct(models.Model):
    core_composition = models.CharField('Chemical formula of core', max_length=50)
    coat_composition = models.CharField('Chemical formula of coating', max_length=50, null=True)
    av = models.CharField('Area/volume', max_length=250, default='')
    mm = models.CharField('Max(size)/min(size)', max_length=250, default='')
    magnetic_moment_core = models.CharField('Magnetic moment of core', max_length=50, default='')
    sum_surface_spins = models.CharField('Sum surface spins', max_length=50, default='')
    sat_magn = models.CharField('Saturation magnetization', max_length=50, default='')
    coerc = models.CharField('Coercitivity', max_length=50)
    rem_magn = models.CharField('Remanence magnetization', max_length=50)
    logp = models.CharField('LogP of organic coating', max_length=50)
    hacceptors = models.CharField('Number of H acceptors of organic coating', max_length=50)
    field_strenght = models.CharField('Strenth of field', max_length=50, default='')

    def __str__(self):
        return self.core_composition


class PredictionActMed(models.Model):
    core_composition = models.CharField('Chemical formula of core', max_length=50)
    av = models.CharField('Area/volume', max_length=250, default='')
    mm = models.CharField('Max(size)/min(size)', max_length=250, default='')
    magnetic_moment_core = models.CharField('Magnetic moment of core', max_length=50, default='')
    sum_surface_spins = models.CharField('Sum surface spins', max_length=50, default='')
    logp = models.CharField('LogP of organic coating', max_length=50)
    hacceptors = models.CharField('Number of H acceptors of organic coating', max_length=50)


    def __str__(self):
        return self.core_composition




class PredictionActBase(models.Model):
    core_composition = models.CharField('Chemical formula of core', max_length=50)


    def __str__(self):
        return self.core_composition

class nanoparticle_mri_r1_pred(models.Model):
    paper_doi = models.CharField('Link to article', max_length=250)
    np_core = models.CharField('Chemical formula of core', max_length=50)
    np_shell_1 = models.CharField('Chemical formula of coating', max_length=50, null=True)
    av = models.CharField('Area/volume', max_length=250, default='')
    mm = models.CharField('Max(size)/min(size)', max_length=250, default='')
    magnetic_moment = models.CharField('Magnetic moment of core', max_length=50, default='')
    sum_surface_spins = models.CharField('Sum surface spins', max_length=50, default='')
    squid_sat_mag = models.CharField('Saturation magnetization', max_length=50, default='')
    org_coating_LogP = models.CharField('LogP of organic coating', max_length=50, default='')
    org_coating_HAcceptors = models.CharField('Number of H acceptors of organic coating', max_length=50, default='')
    mri_h_val = models.CharField('Field strength', max_length=50, default='')
    mri_r1 = models.CharField('R1', max_length=50, default='')

    def __str__(self):
        return '{}[{}]'.format(self.np_core, self.mri_r1)

    class Meta:
        verbose_name = 'nanoparticle_mri_r1_pred'
        verbose_name_plural = 'nanoparticle_mri_r1_pred'


class nanoparticle_mri_r2_pred(models.Model):
    paper_doi = models.CharField('Link to article', max_length=250)
    np_core = models.CharField('Chemical formula of core', max_length=50)
    np_shell_1 = models.CharField('Chemical formula of coating', max_length=50, null=True)
    av = models.CharField('Area/volume', max_length=250, default='')
    mm = models.CharField('Max(size)/min(size)', max_length=250, default='')
    magnetic_moment = models.CharField('Magnetic moment of core', max_length=50, default='')
    sum_surface_spins = models.CharField('Sum surface spins', max_length=50, default='')
    squid_sat_mag = models.CharField('Saturation magnetization', max_length=50, default='')
    org_coating_LogP = models.CharField('LogP of organic coating', max_length=50, default='')
    org_coating_HAcceptors = models.CharField('Number of H acceptors of organic coating', max_length=50, default='')
    mri_h_val = models.CharField('Field strength', max_length=50, default='')
    mri_r2 = models.CharField('R2', max_length=50, default='')

    def __str__(self):
        return '{}[{}]'.format(self.np_core, self.mri_r2)

    class Meta:
        verbose_name = 'nanoparticle_mri_r2_pred'
        verbose_name_plural = 'nanoparticle_mri_r2_pred'



class nanoparticle_sar_pred(models.Model):
    paper_doi = models.CharField('Link to article', max_length=250)
    np_core = models.CharField('Chemical formula of core', max_length=50)
    np_shell_1 = models.CharField('Chemical formula of coating', max_length=50, null=True)
    conc = models.CharField('Particles concentration', max_length=250, default='')
    av = models.CharField('Area/volume', max_length=250, default='')
    mm = models.CharField('Max(size)/min(size)', max_length=250, default='')
    magnetic_moment = models.CharField('Magnetic moment of core', max_length=50, default='')
    squid_sat_mag = models.CharField('Saturation magnetization', max_length=50, default='')
    squid_coerc_f = models.CharField('Coercitivity', max_length=50, default='')
    squid_rem_mag = models.CharField('Remanence magnetization', max_length=50, default='')
    org_coating_LogP = models.CharField('LogP of organic coating', max_length=50, default='')
    org_coating_HAcceptors = models.CharField('Number of H acceptors of organic coating', max_length=50, default='')
    htherm_h_amp = models.CharField('Amplitude of field', max_length=50, default='')
    htherm_h_freq = models.CharField('Frequency of field', max_length=50, default='')
    sar = models.CharField('SAR', max_length=50, default='')

    def __str__(self):
        return '{}[{}]'.format(self.np_core, self.sar)

    class Meta:
        verbose_name = 'nanoparticle_sar_pred'
        verbose_name_plural = 'nanoparticle_sar_pred'