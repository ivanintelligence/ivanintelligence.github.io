---
layout: subpage
hero: /img/projects/plant-growth-multimodal-time-series-model/plant-growth-multimodal-time-series-model.jpeg
---

<title>Predicting Plant Growth Structures with Multimodal Data Using Convolutional Long Short-Term Memory (ConvLSTM)</title>

A multimodal deep learning model that predicts the next frame of plant growth by understanding two input modalities: growth patterns and environmental conditions. For instance, the model changes its prediction when the user adjusts pH level. Images of growth patterns pass through convolutional long short-term memory (ConvLSTM) layers, while environmental data passes through a series of layers such as time-distributed, reshape, and lambda. This produces two tensors with aligned temporal dimensions, which are concatenated to become a combined multimodal tensor. To generate the predicted frame, a 3D Convolution layer is employed.

In addition, a custom loss function is used to leverage the temporal dynamics and visual fidelity of Temporal Consistency and Mean Squared Error losses. The weighted sum of both is evaluated against each standalone loss quantitatively using metrics like Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Total Variation, and qualitatively through visual inspection. Results show that the custom loss delivers slightly better performance in preserving visual fidelity and temporal dynamics of slow-moving objects.

<tag>Long Short-Term Memory (LSTM)</tag>
<tag>Recurrent Neural Networks (RNN)</tag>
<tag>Predictive Modeling</tag>
<tag>Feature Engineering</tag>
<tag>Deep Learning</tag>
<tag>Optimization Algorithms</tag>
<tag>Model Training</tag>
<tag>Backpropagation</tag>
<tag>Neural Networks</tag>
<tag>Model Validation</tag>

<a href="https://www.johnivandiaz.com" class="arrow-link">See source code</a>

<br>

<h1>Discovery Phase</h1>

<h2>Use Case Definition</h2>

Within recent deep learning advancements, a convolutional long short-term memory
(ConvLSTM) architecture exists that predicts future outcomes based on historical
spatiotemporal data, such as images. It extends traditional LSTMs by replacing matrix
multiplications with convolution operations, effectively handling spatiotemporal sequence
prediction tasks. Other studies have utilized this architecture for tasks like weather
forecasting and anomaly detection. They relied on established loss functions, such
as Mean Squared Error (MSE), which focuses on pixel-level accuracy. Other established
loss functions, like Temporal Consistency (TC), emphasize smooth temporal progression to
avoid abrupt, unnatural transitions. However, these loss functions generally address only
one aspect or strength at a time.

As the proponents pursue this foundational study, certain gaps emerge:

<ol>
  <li>There is currently no multimodal deep learning architecture designed to predict plant growth structures using both past appearances and environmental factors.</li>
  <li>There is a need for an improved loss function that captures multiple aspects necessary for learning plant growth patterns in images.</li>
</ol>

To address these gaps, the proponents aim to develop:

<ol>
  <li>A multimodal convolutional long short-term memory architecture capable of predicting future plant growth frames with respect to both past growth structures.</li>
  <li>An improved loss function that unifies the attributes of Mean Squared Error and Temporal Consistency losses to capture pixel-level accuracy and temporal dependencies of plant growth frames during model training.</li>
</ol>

<h2>Data Exploration</h2>




<h1>Development Phase</h1>

<h2>Data Pipeline Creation</h2>

The figure below illustrates the concept of multimodal input for predicting plant growth.

<figure style="--img-max: 560px;">
  <img src="/img/projects/plant-growth-multimodal-time-series-model/multimodal-learning.jpg">
  <figcaption>Concept of the multimodal learning for predicting plant growth.</figcaption>
</figure>






<h2>Model Building</h2>




<h2>Model Evaluation</h2>



