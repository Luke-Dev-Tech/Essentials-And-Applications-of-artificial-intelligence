from django.urls import path
from . import views 

urlpatterns = [
    path("home/", views.Home, name="Home"),
    path("about/", views.About, name="About"),
    path("team/", views.Team, name="Team"),
   
]
