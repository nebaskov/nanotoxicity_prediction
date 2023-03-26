from django.db import models

class Contact(models.Model):
    Problem = models.CharField('Problem', max_length=200)
    Name = models.CharField('Name', max_length=200)
    Email = models.CharField('Email', max_length=200)

    def __str__(self):
        return self.Name