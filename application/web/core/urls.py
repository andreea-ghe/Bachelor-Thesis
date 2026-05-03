from django.urls import path

from . import views

app_name = "core"

urlpatterns = [
    path("", views.index, name="index"),
    path("api/fracture/<str:fracture_id>/", views.fracture_pieces, name="fracture_pieces"),
    path("api/predict/", views.predict, name="predict"),
]
