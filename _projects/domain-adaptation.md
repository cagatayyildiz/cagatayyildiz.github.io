---
layout: mediumpost
title: Domain adapting LLMs
description: "A practitioner's guide: what the experiments actually say about perplexity, evaluation, and continual pretraining"
date: 2026-04-05
tags: llm continual-learning domain-adaptation evaluation
category: work
importance: 1
img: assets/img/domain-adapt/cover.png
---

Suppose you have a general-purpose LLM and a target domain — say, biomedical literature. The standard playbook is straightforward: collect domain text, keep training the model, watch the loss go down, ship it. This workflow is intuitive and widespread. It also has a quiet failure mode that most practitioners never notice until something downstream breaks.

Over the past two years, I have been running experiments on domain-adaptive and continual pretraining across a wide range of models, domains, and training setups. This post synthesizes what I found — specifically, three things that surprised me: why the metric everyone uses is broken, how to fix it, and what continual pretraining actually does to a model compared to standalone adaptation.

<div class="alert alert-info" role="alert">
  The goal is not to present our papers. It is to answer the question a practitioner actually asks: when should I adapt my model, how should I do it, and how do I know if it worked?
</div>


## 1. The metric you're trusting is broken

Let's start with a concrete experiment. Take GPT-2 and continue pretraining it on two groups of domains: Wikipedia articles (History, Culture, Technology, ...) and scientific papers from Semantic Scholar (Physics, Mathematics, Astrophysics, ...). In both cases, **the training loss goes down** — the model is learning, nothing looks wrong.

<div class="row justify-content-sm-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/domain-adapt/train_curves.png" title="Training curves" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Training loss curves for GPT-2 adapted to two different domains — one from scientific papers and one from Wikipedia. Training loss decreases in both cases.
</div>

Now look at the validation and test perplexity — the number everyone reports. For the scientific papers domain, it improves after adaptation, as expected. For the Wikipedia domain, it gets *worse*. The model was explicitly trained on the target domain, the training loss dropped, and yet the held-out test perplexity deteriorated.

<div class="row justify-content-sm-center">
    <div class="col-sm-11 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/domain-adapt/test_curves.png" title="Perplexity change after adaptation" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Validation and test loss on a scientific paper and a Wikipedia domain. Interestingly, both validation test perplexities increase on the Wikipedia domain while they are stable for the scientific papers domain.
</div>

Why? The key is distributional similarity. Wikipedia-style text is already well-represented in GPT-2's pretraining corpus (OpenWebText) and likewise that of other LLMs. When you continue training on data that closely resembles what the model has already seen, you are not teaching it something new but nudging it toward a slightly different local optimum in a part of the loss landscape it has already visited. The model overfits to the narrow distribution of your adaptation set at the expense of the broader representation it previously had. The scientific paper domains are genuinely out-of-distribution. There the model has something real to learn, and the perplexity improvement reflects that.

But there is a deeper problem. Even when perplexity improves, it may not be telling you what you think it is.

<div class="row justify-content-sm-center">
    <div class="col-sm-9 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/domain-adapt/token_histogram.png" title="Token-level perplexity changes" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Per-token perplexity change after adaptation on a Wikipedia domain. Most tokens show small changes. The large degradations are concentrated in a handful of non-informative tokens (newlines, punctuation, whitespace) that dominate the aggregate perplexity score and obscure what is actually happening to domain-relevant vocabulary.
</div>

When you break perplexity down to the token level, the degradation on Wikipedia domains is not spread evenly across the vocabulary. It is concentrated in a small number of domain-irrelevant tokens — newline characters, punctuation, generic function words. These tokens appear frequently in any text, so they dominate the aggregate perplexity. A model could be learning your domain vocabulary perfectly and still show degraded perplexity if its predictions for `\n` and `.` have shifted slightly.

Perplexity is an average over all tokens, domain-relevant and irrelevant alike. As an adaptation signal, it is fundamentally noisy.

<div class="alert alert-warning" role="alert">
  <strong>Practical takeaway:</strong> If your target domain is well-represented in the base model's pretraining data — think Wikipedia-style text, news, general web content — standalone domain adaptation will likely hurt more than help. Before spending compute, measure the distributional distance between your target corpus and the model's pretraining data (e.g., using sentence embeddings). If the distance is _small_, lean on the base model's existing capabilities instead.
</div>


## 2. What to use instead: prediction rank on domain-specific targets

If perplexity is the wrong metric, what is the right one? My answer, developed in follow-up work, is prediction rank computed on domain-specific target terms.

The idea is simple. Rather than asking "how well does the model predict the average token in this document?", we ask "how well does the model predict the vocabulary that is specific and diagnostic of this domain?" For example, a model that ranks `backpropagation` highly when given a prompt about gradient descent is demonstrating genuine domain knowledge. A model that predicts `\n` well is not.

<div class="row justify-content-sm-center">
    <div class="col-sm-11 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/domain-adapt/pipeline.png" title="Benchmark construction pipeline" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Overview of our benchmark construction pipeline. Starting from a raw domain corpus, the pipeline extracts domain-specific keywords, matches them to relevant sentences, constructs a target vocabulary, and assembles prompt–target pairs. Models are then evaluated by the rank they assign to the correct target token given the prompt.
</div>

The pipeline to construct domain-specific prompt-target pairs is fully automated: given any raw text corpus, it extracts domain keywords using n-gram statistics and embedding-based deduplication, matches sentences to keywords, and constructs prompt–target pairs where the target is a domain-specific term. For example, for the keyword *reinforcement learning*, a target might be `replay` given the prompt *"Prior attempts at improving data efficiency in reinforcement learning involved the use of an Experience"*. No human annotation, no LLM calls, no contamination risk. Conveniently, the benchmark can be regenerated from fresh data at any time.

We validate this against a manually curated expert benchmark built from a graduate-level deep learning textbook (281 hand-crafted prompt–target pairs). The correlation between our automated pipeline's model rankings and the expert benchmark rankings is r = 0.99 (p < 0.001). The pipeline is not just convenient, it also agrees with expert human judgment better than multiple choice question benchmarks do (e.g., their predictions substantially change when options are shuffled).

Now let's see what this metric reveals that perplexity misses, specifically when evaluating adapted models across model families.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/domain-adapt/base_chat.png" title="Base vs chat model evaluation" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Prediction ranks and the correlations between pipeline– and Claude-generated benchmarks for six base models (left) and their instruction-tuned chat counterparts (right) on the General Economics domain.
</div>

Two things stand out here. First, the pipeline ranks correlate strongly with the reference benchmark we generated with Claude for both base and chat models — perplexity holds up for base models but collapses as a signal once you add instruction tuning. Second, base models are almost always better their chat-aligned counterparts on domain knowledge: instruction tuning trades away factual domain knowledge for conversational compliance. This "alignment tax" is real, measurable, and heterogeneous — Llama-2-7B and Mistral-7B show severe drops while Qwen2-1.5B and Llama-3.1-8B are much more robust.


## 3. Continual pretraining: a better adaptation strategy

So far we have discussed *whether* to adapt and *how to measure* adaptation. Now we switch to *how* to adapt, specifically when you need to specialize a model across multiple domains over time.

**Continual pretraining** is the process of training a model sequentially on a series of domain corpora, one after another, without retraining from scratch. At each step, the model is updated on a new domain corpus and a checkpoint is saved. The goal is to end up with a model that has accumulated knowledge across all domains while retaining what it learned earlier.

To understand what continual pretraining actually does, consider the following experiment. We take Llama-2-7B and train it sequentially on 12 domains, starting from domains dissimilar to our target (CS.AI) and progressively moving toward it. At each checkpoint, we measure domain knowledge using the prediction rank pipeline from Section 2.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/domain-adapt/continual_llama.png" title="Continual pretraining of Llama-2-7B" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Evaluation results for 12 continual pretraining checkpoints of Llama-2-7B, measured on the CS.AI target domain. The bottom-left panel shows the training domain sequence, ordered by dissimilarity to CS.AI. Top row: prediction ranks on (a) the Claude-generated reference benchmark and (b–c) two benchmarks generated by our pipeline. Panel (d) shows the evolution of perplexity. Only the rank-based metrics track the expected pattern — improving as training domains become more similar to CS.AI. Perplexity show no reliable correlation.
</div>

The rank-based metrics tell a coherent story: domain knowledge of CS.AI initially drops as the model trains on unrelated domains, stabilizes in the middle, then improves as training domains converge toward CS.AI. Perplexity shows no such pattern — it oscillates without correlation to what the model actually knows. This is the clearest demonstration I have of why perplexity is the wrong tool for tracking continual learning progress.

### 3.1. Continual pretraining vs. standalone domain adaptation

Now for the key practical comparison: is it better to continuously pretrain across many domains, or to adapt directly to your target domain in a single step?

Across GPT-2 model sizes (Small, Medium, Large, XL) and RoBERTa, continual pretraining consistently achieves lower final perplexity on the target domain than standalone domain-adaptive pretraining (DAPT). The margin is substantial for smaller models and narrows but persists for larger ones.

<div class="row justify-content-sm-center">
    <div class="col-sm-11 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/domain-adapt/continual_vs_dapt.png" title="Continual pretraining vs domain adaptive pretraining" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Test perplexity across model sizes and training orders. For reference, zero-shot and domain-adaptive pretraining (DAPT) perplexities are included. Continual pretraining consistently outperforms DAPT across all GPT-2 variants. Larger models start from a lower baseline and show smaller but still consistent improvements. Llama2-7B does not benefit — the individual domain corpora in this setup are simply too small to shift a 7B parameter model.
</div>

There is an important caveat for large models. Llama-2-7B did not benefit from continual pretraining in our setup. The reason is not architectural — it is a data quantity issue. The individual domain corpora we used (circa. 7B tokens from the M2D2 dataset) are too small to meaningfully shift a 7B parameter model in a single epoch. If your target domain corpus is large enough and your model is large, standalone DAPT may be sufficient. The threshold scales with model size.

### 3.2. Domain ordering matters (perhaps surprisingly)

When you have a sequence of training domains, does the order matter? Yes, substantially. We compared two orderings: **similar-order** (semantically related domains follow one another) and **random-order** (domains are shuffled).

The counterintuitive finding: **random domain ordering leads to better final perplexity and less forgetting** in the general case. When the model moves between very different domains, the parameter updates are more varied and seem to produce broader, more transferable representations. Gradients from dissimilar domains are less likely to interfere destructively with each other.

Similar-order training wins in one specific scenario: when you are deliberately building a domain expert and care about specialization over generality. If your final training domain is your target and all preceding domains are closely related to it, the model specializes tightly. Whether that is desirable depends entirely on your downstream use case.

### 3.3. Smaller models: faster learners, faster forgetters

One finding that carries immediate practical weight: **smaller models show the highest rates of both learning and forgetting**. When a small model (GPT-2 Small or Medium) is trained on a new domain, it adapts rapidly — but it also loses prior domain knowledge faster than larger models. Larger models adapt more slowly but retain more of what they previously knew.


<div class="row justify-content-sm-center">
    <div class="col-sm-11 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/domain-adapt/transfer_forgetting.png" title="Continual pretraining vs domain adaptive pretraining" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Left: Backward transfer — improvement in perplexity on previously seen domains throughout continual pretraining (random order). All models stay above the dashed zero line, meaning continual pretraining never hurts old domains on average. Smaller models show larger gains but also higher variance. Right: Average forgetting across all GPT-2 sizes. Smaller models forget more (2.21 for GPT-2 Small vs. 1.48 for GPT-2 XLarge). The two panels together tell the same story: smaller models are faster learners and faster forgetters; larger models trade peak adaptation speed for stability.

</div>

<div class="alert alert-warning" role="alert">
  <strong>Practical takeaway for deployment decisions:</strong> If you are building a specialist model and do not care about retaining general knowledge, a smaller model will specialize more efficiently and cheaply. If you need a model that accumulates knowledge over time without degrading on earlier domains, investing in a larger base is not just about capability — it is also about memory stability.
</div>


## 4. A decision checklist

Putting everything together, here is how I now approach domain adaptation decisions:

1. **Measure distributional distance first.** Compute MMD or Fréchet distance between your target domain and the base model's pretraining data (if known). If distance is low — your domain is generic, web-like, or similar to Wikipedia — standalone adaptation will likely hurt more than help.

2. **Use prediction rank, not perplexity, to evaluate.** Build a small domain-specific benchmark using our pipeline described above. Track model rankings rather than aggregate perplexity. This will correctly identify when adaptation has improved genuine domain knowledge versus when it has just shifted the distribution of function tokens.

3. **Prefer continual pretraining over standalone domain adaptation** when you have access to multiple related corpora and are working with models up to ~1.5B parameters. The multi-domain scaffold consistently improves final domain performance.

4. **Randomize domain order** unless you are deliberately building a narrow specialist. For general-purpose accumulation of domain knowledge, shuffled training leads to better transfer and less catastrophic forgetting.

5. **Check alignment tax if using a chat model.** Base models consistently outperform their chat-aligned counterparts on domain knowledge tasks. If your application is knowledge retrieval or completion rather than conversation, consider whether you need instruction tuning at all — or at least measure what it costs you.

---

*This post synthesizes findings from two collaborative papers: [Adaptation Odyssey in LLMs: Why Does Additional Pretraining Sometimes Fail to Improve? (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.1108/) and [Investigating Continual Pretraining in Large Language Models: Insights and Implications (TMLR 2025)](https://openreview.net/forum?id=aKjJoEVKgO). The evaluation pipeline described in Section 2 is part of ongoing work.*
