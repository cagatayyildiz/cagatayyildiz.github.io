---
layout: page
permalink: /notes/
title: ML Notes
description: some machine learning writings, mostly presentations.
nav: true
---

<!-- {% for post in site.posts %}
  {% if post.categories contains 'notes' %}
    <div class="post">
        <h3 class="title"><a href="{{ post.url }}">{{ post.title }}</a></h3>
        <p class="meta">Date: {{ post.date }}</p>
    </div>
  {% endif %}
{% endfor %} -->

<!-- 
{% for tag in site.tags %}
  <h4>{{ tag[0] }}</h4>
  <ul>
    {% for post in tag[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %} -->

{% for note in site.notes reversed %}
  <h2> <a href="{{ note.url }}">{{ note.title }}</a></h2>
  <p>[{{ note.date | date: "%Y-%m-%d" }}] {{ note.description }}</p>
{% endfor %}

