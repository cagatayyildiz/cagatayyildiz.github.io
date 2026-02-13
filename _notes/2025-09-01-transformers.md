---
layout: widepost
title:  Transformers tutorial
date:   2025-09-01 08:57:00-0400
description: A gentle introduction to transformer architectures
img: assets/img/transformer.png
tags: tutorial
category: tutorial
giscus_comments: true
related_posts: false
---


{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/transformer_tutorial.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/transformer_tutorial.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}


