from django.db import models

class NpDb(models.Model):
    material = models.CharField('NP chemical formula', max_length=50)
    diameter = models.CharField('Diameter (nm)', max_length=50, null=True)
    zeta = models.CharField('Zeta-potential (mV)', max_length=250, default='')
    electroneg = models.CharField('Electronegativity', max_length=50, default='')
    ionic_rad = models.CharField('Ionic radius', max_length=50, default='')
    concentr = models.CharField('Concentration (g/L)', max_length=50, default='')
    mol_weight = models.CharField('Molecular weigth (g/mol)', max_length=50, default='')
    time = models.CharField('Time (h)', max_length=50, default='')
    
    cell_type = models.CharField('Cell type', max_length=250, default='')
    line_primary_cell = models.CharField('Line primary cell', max_length=50, default='')
    animal = models.CharField('Cell source', max_length=50, default='')
    cell_morph = models.CharField('Cell morphology', max_length=50, default='')
    cell_age = models.CharField('Cell age', max_length=50, default='')
    cell_organ = models.CharField('Cell organ', max_length=50, default='')
    
    test = models.CharField('Test name', max_length=50, default='')
    test_indicator = models.CharField('Test indicator', max_length=50, default='')
     
    viability = models.CharField('Cell viability (%)', max_length=50, default='')

    class Meta:
        verbose_name = 'nanoparticle_db'
        verbose_name_plural = 'nanoparticle_db'
