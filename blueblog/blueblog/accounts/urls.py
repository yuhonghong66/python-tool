from django.conf.urls import url
from django.contrib.auth.views import login, logout

from . import views


app_name = 'accounts'
urlpatterns = [
    # Sign Up
    url(r'^sign-up/', views.UserRegistrationView.as_view(), name='sign-up'),
    # Sign In
    url(r'^sign-in/', login, {'template_name': 'sign_in.html'}, name='sign-in'),
    # Sign Out
    url(r'^sign-out/', logout, {'next_page': '/account/sign-in/'}, name='sign-out'),
]