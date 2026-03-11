from django.shortcuts import render,redirect,get_object_or_404

# Create your views here.



def Home(request):
    return render(request, 'index.html')



def About(request):
    return render(request, 'about.html')



def Team(request):
    return render(request, 'team.html')