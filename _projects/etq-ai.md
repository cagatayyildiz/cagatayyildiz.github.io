---
layout: mediumpost
title: ETQ-AI Data Collection App
description: What I learned building an AI-powered classroom data collection app
date: 2026-02-11
tags: applied-ml speech-recognition product-development
category: work
importance: 1
img: assets/img/etqai/etqai-logo.jpg
---


Classroom observation is expensive, subjective, and logistically difficult to scale. In Germany, high-quality teaching feedback is constrained not only by cost, but also by strict data protection law and institutional trust requirements.

In the [ETQ-AI (Enhancing Teaching Quality with Artificial Intelligence)](https://uni-tuebingen.de/en/faculties/faculty-of-economics-and-social-sciences/subjects/department-of-social-sciences/education-sciences-and-psychology/research/current-studies/etq-ai/) project, our goal is to automate parts of teaching quality assessment: record classroom discourse, transcribe it, and use LLMs to score dimensions such as classroom management and student cognitive engagement. The long-term vision is simple but ambitious:

<div class="alert alert-info" role="alert">
  A teacher records a lesson and receives structured, actionable feedback — without a human observer in the room.
</div>

What sounds like an ML problem quickly became a systems, privacy, and deployment problem. As the technical lead and product owner for ETQ-AI data collection app, I owned the full product lifecycle, from system architecture and GDPR compliance to managing the development team. This post details how I bridged the gap between academic ML prototypes and a robust, privacy-first production mobile app used in real German classrooms. Coming from an academic background, a summary of lessons I learned in this applied projecct are:

1. Real-world deployment interests differ so much from academic benchmark metrics.
2. Ensuring privacy requires broad, system-level handling.
3. Failures of ML models can be handled with minimal engineering efforts.
4. Infrastructure decisions significantly influence timelines.
5. Ownership of a project requires maintaining coherence across architecture, user needs, compliance, and research goals.

Let's dive deeper into my project!


### 1. The challenge: beyond the model

The core research question was automated teaching quality assessment: can automated speech recognition (ASR) + LLMs replace human raters?
{% include pipeline-diagram.html %}

However, to answer that, we first needed a reliable data collection pipeline. We needed **recordings from real classrooms** with parental consent, strict GDPR compliance, and a format researchers could actually use. We also needed a user-friendly tool that teachers would actually want to use.

I acted as the bridge between research requirements and engineering reality. My role involved:
* **Product management:** Defining the scope of data collection, roadmap, and user requirements.
* **Technical architecture:** Designing the end-to-end data flow and making trade-offs between privacy, latency, and model accuracy.
* **Team leadership:** Managing two student developers (Usman Amjad and Nitin Jain) who implemented the Flutter frontend, and a PhD student (Puja Maharjan) who built the ML pipeline.
* **Stakeholder management:** Coordinating between project members and teachers, and translating legal requirements into technical specifications.

### 2. Starting simple: the recording interface

The app's core function is straightforward: record audio, transcribe it, let the teacher review and upload.
<div class="row justify-content-center">
    <div class="col-sm-9">
        <div class="row">
            <div class="col-sm mt-3 mt-md-0">
                {% include figure.liquid path="assets/img/etqai/etqai-login.png" title="Login screen" class="img-fluid rounded z-depth-1" %}
            </div>
            <div class="col-sm mt-3 mt-md-0">
                {% include figure.liquid path="assets/img/etqai/etqai-main.png" title="Main screen" class="img-fluid rounded z-depth-1" %}
            </div>
            <div class="col-sm mt-3 mt-md-0">
                {% include figure.liquid path="assets/img/etqai/etqai-recording.png" title="Recording in progress" class="img-fluid rounded z-depth-1" %}
            </div>
        </div>
    </div>
</div>
<div class="caption">
    Left: Login via email or Google — we collect no personal data beyond authentication. Center: The main screen before any recordings. Right: A recording in progress.
</div>

Login supports email and Google authentication. We deliberately don't collect any personal data at this stage. The main screen is minimal: a record button and a list of past recordings. This simplicity was intentional — our users are teachers in the middle of a workday, not power users.


### 3. GDPR shaped the architecture

Working with classroom audio in Germany means working under GDPR, and this constraint influenced almost every design decision.

We, as researchers, need data on a server to run experiments, but teachers (and students' parents) need to trust that their data is handled properly. We landed on a design where **personal data stays on the teacher's device by default**, and upload to our research servers is an explicit, separate action. The teacher can review the transcription before uploading, and can delete data from the device, from the server, or from both — independently. We also added a full account deletion option: one tap removes all associated data everywhere. This sounds simple, but implementing proper cascading deletion across device storage, cloud storage, and authentication records required careful coordination.

<div class="row justify-content-center">
    <div class="col-sm-6">
        <div class="row">
            <div class="col-sm mt-3 mt-md-0">
                {% include figure.liquid path="assets/img/etqai/etqai-upload.png" title="Upload confirmation" class="img-fluid rounded z-depth-1" %}
            </div>
            <div class="col-sm mt-3 mt-md-0">
                {% include figure.liquid path="assets/img/etqai/etqai-delete.png" title="Granular delete options" class="img-fluid rounded z-depth-1" %}
            </div>
        </div>
    </div>
</div>
<div class="caption">
    Left: Upload confirmation showing both audio and transcript delivered to the server. Right: Granular deletion — teachers can remove data from their device, from the cloud (S3), or both.
</div>

Infrastructure strategy:
We store uploaded data on **Amazon S3 servers in Frankfurt (eu-central-1)** to ensure that the data stay within the EU. However, for transcription, we utilize [high performance computing (HPC) servers at Göttingen](https://gwdg.de/en/hpc/). Separating storage (Frankfurt) from compute (Göttingen) was a strategic decision I made after mapping the data governance capabilities of each site. GWDG offered the necessary GPU power but was only authorized for transient processing, while AWS provided the compliant long-term storage we needed. Orchestrating the alignment between these sites was a key non-technical challenge I solved.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/etqai/system-design.png" title="System design" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The three-component infrastructure: audio recorded on the teacher's smartphone is sent to GWDG HPC servers in Göttingen for transcription, while long-term storage lives on GDPR-compliant S3 servers in Frankfurt. Separating compute from storage was a deliberate architectural decision driven by differing data governance requirements at each institutional site.
</div>


### 4. Real world deployment challenges

This project taught me quite a bit about the differences between ideal academic research and real world deployment. Here are some examples:


#### 4.1. On-device transformers didn't survive contact with reality

Our initial plan was quite fun: running [Whisper](https://github.com/openai/whisper) on-device for transcription so that classroom audio never needs to leave the teacher's phone at all. Maximum privacy, minimum infrastructure.

To make this work, we built on [whisper.cpp](https://github.com/ggml-org/whisper.cpp), a C/C++ port of Whisper optimized for local, CPU-based inference on resource-constrained devices. By stripping away Python/PyTorch dependencies and relying on low-level [GGML](https://github.com/ggml-org/ggml) tensor operations, it can run substantially smaller model variants directly on a smartphone without GPU acceleration. To integrate this into our Flutter app, we used [flutter_rust_bridge](https://github.com/fzyzcjy/flutter_rust_bridge), which auto-generates the FFI (Foreign Function Interface) bindings between Dart and Rust. On the Rust side, [whisper-rs](https://github.com/tazz4843/whisper-rs) wraps the raw C API of whisper.cpp, giving us a safe interface. The resulting call chain became Flutter UI → Dart → Rust → whisper.cpp (C/C++).

We tested several Whisper variants. On a typical smartphone, only **tiny**, **base**, and **small** were feasible in terms of memory and latency. But because their transcription quality isn't great, we turned to **server-side transcription via API calls to HPC servers**, where we could run large-v3 and large-v3-turbo. The accuracy difference was substantial. This meant the audio does travel to a server for transcription, which we handle by requiring explicit user consent and ensuring the audio is processed transiently (not stored on the transcription server).

Lesson learned: on-device ML is great for demos but is often impractical for production because of the limitations of mobile hardware.

#### 4.2. Academic vs real-world benchmarking

We benchmarked a range of ASR models on both standard datasets and our own classroom recordings. The gap was stark. Even the best-performing models — Voxtral-Mini-3B-2507 and Whisper-large-v3-turbo — achieved word error rates (WER) around 0.64–0.66 on student speech and 0.28 on teacher speech. On standard LibriSpeech benchmarks, those same models score 0.02–0.07. That is roughly a 10× degradation in accuracy when moving from clean, studio-recorded speech to a real classroom.

Student speech was consistently harder than teacher speech, largely because of microphone placement: a phone sitting on the teacher's desk captures the teacher clearly and students at a distance, often through background noise. Beyond microphone distance, classrooms introduce overlapping speech, scraping chairs, background noise, and code-switching between German and technical vocabulary — conditions far removed from the clean audio that most ASR training sets are built on. The training data for most of these models isn't public, but the distribution shift is evident in the numbers. One notable case is that NVIDIA's Canary-1b-v2 achieves an impressive 0.021 WER on LibriSpeech English, yet collapses to a WER above 1.0 on student classroom. High benchmark scores are simply not predictive of classroom performance.

{% comment %}
<div class="row justify-content-sm-center">
    <div class="col-sm-9 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/etqai/wer_heatmap.png" title="WER comparison" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    ASR model benchmark on classroom audio (student and teacher speech) vs. standard LibriSpeech datasets. All models show dramatically higher WER in classroom conditions, with student speech being the hardest. Even the top-ranked Voxtral-Mini-3B-2507 achieves only 0.64 WER on student speech compared to 0.02 on LibriSpeech English.
</div>
{% endcomment %}

#### 4.3. Output repetition issue

We also discovered a surprising failure mode: models keep repeating incorrect transcriptions for dozens of times! We show an example below, where we transcribe a random audio from YouTube:

<div class="row justify-content-sm-center">
    <div class="col-sm-3 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/etqai/etqai-repetition.png" title="Whisper hallucination" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Whisper hallucinating on a classroom recording. After a few plausible sentences, the model enters a repetition loop before degenerating into repeated "(speaks in foreign language)" tags.
</div>

It's impossible to pin this down to a single factor, but our manual inspections show that such hallucinations occur when the words are not clear, and grow significantly with audio duration — especially with noisy or multilingual input. The fix was to **chunk the audio into ~30-second segments** before sending each to the transcription API, then stitch the results together. We discovered that WER rises steeply with segment length across all models, roughly doubling as segments grow from 30 to 300 seconds. The 30-second sweet spot is short enough to avoid hallucinations while long enough to preserve context across sentence boundaries.

{% comment %}
<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/etqai/wer_duration.png" title="WER vs audio duration" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Word error rate increases steeply with audio duration across all models. Whisper-tiny and Canary-1b-v2 show the most severe degradation, while large Whisper variants and Voxtral degrade more gracefully. This motivated our choice to chunk audio into ~30-second segments before transcription.
</div>
{% endcomment %}

But chunking introduced its own engineering challenge: the segmentation and sequential API calls need to happen **in the background** while the teacher continues using the app. In Flutter, managing background and foreground processes properly is non-trivial, especially on iOS where background execution is tightly restricted. We spent more time on getting background transcription reliable than on any ML-related problem.


#### 4.4. Audio recording: deceptively hard on mobile

Another "should be simple but isn't" problem is continuous audio recording on a mobile device. We discovered that **recording stops abruptly** when the user switches to another app, when the screen times out and goes black, or when the phone enters power-saving mode. For a teacher recording a 45-minute lesson, this is a not acceptable.

The solution requires properly implementing background audio services — registering the app as a background audio provider, handling lifecycle events, managing the audio session correctly. On Android this is manageable; on iOS (where Flutter's background execution support is more limited), it required significant workarounds. This is the kind of platform-specific systems engineering that you never think about when you're designing a pipeline in a Jupyter notebook.


### 5. What I would do differently
Looking back, I would 
- **start with deployment constraints.** We initially assumed transcription would be “good enough.” It wasn’t. Downstream components must be designed around noisy, imperfect inputs.
- **budget 3x more time for platform-specific issues.** Mobile OS constraints consumed more engineering time than all ML components combined.
- **design for the user's worst moment.** If recording fails once during a 45-minute lesson, the tool loses trust. Reliability outweighs sophistication.



### 6. Transferable lessons for applied ML
This project changed how I think about applied machine learning:
1. **Benchmarks are not deployment metrics:**  Leaderboard performance does not predict out-of-domain robustness.
2. **Privacy is a systems problem:** GDPR influenced storage location, compute separation, consent flows, and deletion logic more than model choice did.
3. **Model failures can be handled architecturally:** Repetition hallucinations were mitigated through segmentation and orchestration — not model fine-tuning.
4. **Infrastructure decisions dominate timelines:**  Background execution, mobile OS constraints, and data governance consumed more effort than training experiments.
5. **Ownership means maintaining coherence:**  Keeping architecture, user needs, compliance, and research goals aligned required explicit technical leadership.



### 7. The broader takeaway
Building ETQ-AI data collection app taught me that the interesting engineering problems in applied ML are rarely about the models. They're about the system around the models: how data flows, where it's stored, who controls it, what happens when the model fails, and how to make it all invisible to the user. This is the kind of work that doesn't produce papers but determines whether a research prototype becomes a usable tool. It's also the lens I now bring to research problems more broadly: not just "what's the best model?" but "what does it take to make this actually work?"

---

*ETQ-AI is a joint project between the [Hector Research Institute of Education Sciences and Psychology](https://uni-tuebingen.de/en/faculties/faculty-of-economics-and-social-sciences/subjects/department-of-social-sciences/education-sciences-and-psychology/institute/) and the [Cluster of Excellence "Machine Learning for Science"](https://uni-tuebingen.de/en/research/core-research/cluster-of-excellence-machine-learning/home/) at the University of Tübingen. The app was implemented by Usman Amjad and Nitin Jain. ML experiments were conducted by Puja Maharjan.*
