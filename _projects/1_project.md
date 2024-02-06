---
layout: widepost
title: project 1
description: a project with a background image
img: assets/img/12.jpg
importance: 1
category: work
related_publications: true
---

{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/bcpm.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/bcpm.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}



