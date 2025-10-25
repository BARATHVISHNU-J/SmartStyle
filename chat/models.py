from django.db import models

class FashionItem(models.Model):
    name = models.CharField(max_length=200)
    category = models.CharField(max_length=100)  # e.g., tops, bottoms, shoes
    description = models.TextField()
    season = models.CharField(max_length=50, blank=True)  # spring, summer, fall, winter
    occasion = models.CharField(max_length=100, blank=True)  # casual, formal, work, athletic
    gender = models.CharField(max_length=20, blank=True)  # male, female, unisex
    price_range = models.CharField(max_length=50, blank=True)  # budget, mid-range, premium
    brands = models.JSONField(default=list)  # list of suggested brands
    colors = models.JSONField(default=list)  # list of available colors
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class Trend(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    season = models.CharField(max_length=50)
    year = models.IntegerField()
    categories = models.JSONField(default=list)  # affected categories
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class UserPreference(models.Model):
    session_id = models.CharField(max_length=100, unique=True)
    name = models.CharField(max_length=100, blank=True)
    gender = models.CharField(max_length=20, blank=True)
    style_preferences = models.JSONField(default=list)
    color_preferences = models.JSONField(default=list)
    budget = models.CharField(max_length=50, blank=True)
    occasions = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Preferences for {self.session_id}"

class ChatSession(models.Model):
    session_id = models.CharField(max_length=100, unique=True)
    messages = models.JSONField(default=list)  # list of message dicts
    context = models.JSONField(default=dict)  # conversation context
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Chat session {self.session_id}"
