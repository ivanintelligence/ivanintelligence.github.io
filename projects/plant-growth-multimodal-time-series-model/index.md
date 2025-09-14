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




<h1>Development Phase</h1>



<h2>Data Pipeline Creation</h2>


<h3>Dataset</h3>


<h3>Image Rectification</h3>




<h3>Image Segmentation</h3>


<h3>Resize</h3>


<h3>Normalization</h3>

<h3>Sequence Length Standardization</h3>


<h3>Augmentation</h3>




<h2>Model Building</h2>


<h3>Sequence Learning</h3>

<h3>Feedforward Mapping</h3>

<h3>Feature Fusion</h3>

<h3>RGB Mapping</h3>

<h3>Deep Learning Structure</h3>

<h3>Architecture Summary</h3>

<h3>Loss Function</h3>






<h2>Model Evaluation</h2>

<h3>Evaluation Metrics</h3>

<h3>Evaluation Flowchart</h3>

<h3>Training Loss Curves</h3>

<h3>Training Metric Curves</h3>

<h3>Quantitative Testing Results</h3>

<h3>Qualitative Testing Results</h3>











<h1>Discovery Phase</h1>

<h2>Use Case Definition</h2>

Within recent deep learning advancements, a convolutional long short-term memory (ConvLSTM) architecture exists that predicts future outcomes based on historical spatiotemporal data, such as images. It extends traditional LSTMs by replacing matrix multiplications with convolution operations, effectively handling spatiotemporal sequence prediction tasks. Other studies have utilized this architecture for tasks like weather forecasting and anomaly detection. They relied on established loss functions, such as Mean Squared Error (MSE), which focuses on pixel-level accuracy. Other established loss functions, like Temporal Consistency (TC), emphasize smooth temporal progression to avoid abrupt, unnatural transitions. However, these loss functions generally address only one aspect or strength at a time.

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

The dataset was split into three subsets: training, validation, and test sets. Conceptually, if one were to quickly view all images of a single sequence in succession, one would see the plant gradually growing over time. Formally, for the s<sup>th</sup> sequence, there is a set of frames:

<div class="equation">
$$I_{s,t} = \{I_{s,1}, I_{s,2}, \ldots, I_{s,T_s}\}$$
</div>

where $I_{s,t}$ is an RGB image (at original resolution) of the plant on day $t$.

In addition to visual data, each sequence contains a CSV file holding parameters that describe environmental and growth conditions for each frame. Here parameters include ambient temperature, soil moisture, luminosity, and soil pH. Thus, for each day $t$ in sequence $s$:

<div class="equation">
$$P_{s,t} = \left[ p_{s,t}^{(\text{temp})}, p_{s,t}^{(\text{soilmoisture})}, p_{s,t}^{(\text{luminosity})}, p_{s,t}^{(\text{soilpH})} \right]$$
</div>

By pairing each image $I_{s,t}$ with its parameters $P_{s,t}$, the model receives a multimodal representation of plant growth, capturing both the visual appearance and the underlying conditions at each time step.







<h1>Development Phase</h1>

<h2>Data Pipeline Creation</h2>

The figure below illustrates the concept of multimodal input for predicting plant growth.

<figure style="--img-max: 560px;">
  <img src="/img/projects/plant-growth-multimodal-time-series-model/multimodal-learning.jpg">
  <figcaption>Concept of the multimodal learning for predicting plant growth..</figcaption>
</figure>





<h2>Model Building</h2>




<h2>Model Evaluation</h2>



