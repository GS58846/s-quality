from django.contrib import admin

from squality.models import Project

# Register your models here.

class ProjectAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_at')
    list_filter = ('name',)
    
    # prepopulated_fields = {'slug': ('name',)}


admin.site.register(Project, ProjectAdmin)