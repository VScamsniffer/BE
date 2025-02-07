
from django.contrib import admin
from django.urls import path
from django.conf.urls import include
from django.views.generic import TemplateView


urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('allauth.urls')),
    path('',TemplateView.as_view(template_name='index.html')),
    path('rollplaying/',include('rp.urls')),
]