---
layout: widepost
title:  Bayesian Change Point Models
date:   2015-10-20 08:57:00-0400
description: Inference for HMMs with switch variables
tags: tutorial
categories: science
giscus_comments: true
related_posts: false

_styles: >
  .fake-img {
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
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


