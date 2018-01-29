# Create your views here.

from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse
from django.views.generic import CreateView


class UserRegistrationView(CreateView):
    form_class = UserCreationForm
    template_name = 'sign_up.html'

    def get_success_url(self):
        return reverse('home')
