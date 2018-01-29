from django.shortcuts import render
# coding: utf-8
from rest_framework import viewsets
from .models import Diary
from .serializer import DiarySerializer
from rest_framework.views import APIView


class DiaryViewSet(viewsets.ModelViewSet):
    queryset = Diary.objects.all()
    serializer_class = DiarySerializer
