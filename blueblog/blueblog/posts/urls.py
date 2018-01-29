# from django.conf.urls import url
# # from django.contrib.auth.views import login, logout
# #
# from . import views
# from django.views.generic import TemplateView
#
#
app_name = 'posts'
# urlpatterns = [
#     # Blog
#     url(r'^blog/', views.DiaryViewSet, name='blog'),
# ]

# coding: utf-8
from rest_framework import routers
from .views import DiaryViewSet

router = routers.DefaultRouter()
router.register(r'blog', DiaryViewSet)
