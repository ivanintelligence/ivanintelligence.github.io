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

Within recent deep learning advancements, a convolutional long short-term memory (ConvLSTM) architecture exists that predicts future outcomes based on historical spatiotemporal data, such as images. It extends traditional LSTMs by replacing matrix multiplications with convolution operations, effectively handling spatiotemporal sequence prediction tasks. Other studies have utilized this architecture for tasks like weather forecasting and anomaly detection. They relied on established loss functions, such as Mean Squared Error (MSE), which focuses on pixel-level accuracy. Other established loss functions, like Temporal Consistency (TC), emphasize smooth temporal progression to avoid abrupt, unnatural transitions. However, these loss functions generally address only one aspect or strength at a time.

This project aims to develop:

<ol>
  <li>A multimodal convolutional long short-term memory architecture capable of predicting future plant growth frames with respect to both past growth structures and environmental factors.</li>
  
  <li>An improved loss function that unifies the attributes of Mean Squared Error and Temporal Consistency losses to capture pixel-level accuracy and temporal dependencies of plant growth frames during model training.</li>
</ol>

<h1>Development Phase</h1>

<h2>Data Pipeline Creation</h2>

The dataset was split into three subsets: training, validation, and test sets. Conceptually, if one were to quickly view all images of a single sequence in succession, one would see the plant gradually growing over time. Formally, for the $s^{\text{th}}$ sequence, there is a set of frames:

<div class="equation">
$$I_{s,t} = \{I_{s,1}, I_{s,2}, \ldots, I_{s,T_s}\}$$
</div>

where $I_{s,t}$ is an RGB image (at original resolution) of the plant on day $t$.

In addition to visual data, each sequence contains a CSV file holding parameters that describe environmental and growth conditions for each frame. Here, parameters include ambient temperature, soil moisture, luminosity, and soil pH. Thus, for each day $t$ in sequence $s$:

<div class="equation">
$$P_{s,t} = \left[ P_{s,t}^{(\text{temp})}, P_{s,t}^{(\text{soilmoisture})}, P_{s,t}^{(\text{luminosity})}, P_{s,t}^{(\text{soilpH})} \right]$$
</div>

By pairing each image $I_{s,t}$ with its parameters $P_{s,t}$, the model receives a multimodal representation of plant growth, capturing both the visual appearance and the underlying conditions at each time step.

Preprocessing is done to convert the raw data, initially stored as large RGB images and unscaled parameter values, into a format suitable for efficient model training. It involves image rectification, segmentation, resize, normalization, sequence length standardization, and augmentation.

<h3>Image Rectification</h3>

Stereo images suffer lens distortion and misalignment between left and right views [22][23]. Objects in both images must share identical scan-lines. Let $I_{s,t} \in \mathbb{R}^{H_0 \times W_0 \times 3}$ denote the RGB image captured on day $t$ of sequence $s$. Each image $I_{s,t}$ is rectified by a calibration map:

<div class="equation">
$$\Phi = (M_x, M_y), \quad M_x, M_y \in \mathbb{R}^{H \times W}$$
</div>

Derivation of this map is discussed in Chapter 6. Each pair $(M_x^{(i,j)}, M_y^{(i,j)})$ stores the sub-pixel source coordinates in the distorted image that should be mapped to the rectified location $(i, j)$. The rectification operator is defined as:

<div class="equation">
$$\mathcal{R}_\Phi(I) (i, j) = I(M_x^{(i,j)}, M_y^{(i,j)}), \quad 0 \leq i < H, 0 \leq j < W.$$
</div>

Hence, the epipolar-aligned frame is obtained by:

<div class="equation">
$$I_{s,t}^{\text{rect}} = \mathcal{R}_\Phi(I_{s,t}).$$
</div>

<h3>Image Segmentation</h3>

The proponents isolate the plant from the background to focus the model's attention on the primary subject. To do this, a pre-trained YOLOv8 instance segmentation model is employed. Given an original image $I_{s,t}$, the segmentation model outputs a binary mask $M_{s,t}$ that highlights plant pixels only. The segmented image $\hat{I}_{s,t}$ is obtained via element-wise multiplication:

<div class="equation">
$$\hat{I}_{s,t} = I_{s,t}^{\text{rect}} \odot M_{s,t}$$
</div>

<h3>Resize</h3>

After segmentation, each image $\hat{I}_{s,t}$ is resized to a standardized dimension $(H,W)$ to reduce computational overhead and ensure a consistent input size:

<div class="equation">
$$I_{s,t}^{\text{res}} = \text{resize}(\hat{I}_{s,t}, (H,W))$$
</div>

<h3>Normalization</h3>

Subsequently, pixel intensities are normalized to the range $[0,1]$ to keep all input values within a controlled scale:

<div class="equation">
$$I_{s,t}^{\text{norm}} = \frac{I_{s,t}^{\text{res}}}{255.0}$$
</div>

The parameters in $P_{s,t}$ may differ widely in scale. To ensure that each parameter dimension contributes proportionately, global means $\mu_p$ and standard deviations $\sigma_p$ are computed across the training set. Each parameter vector is then normalized:

<div class="equation">
$$P_{s,t}^{\text{norm}} = \frac{P_{s,t} - \mu_p}{\sigma_p}$$
</div>

This yields zero-mean, unit-variance parameter distributions that allow the model to treat all parameters more uniformly.

<h3>Sequence Length Standardization</h3>

Since different sequences may have varying lengths $T_s$, a fixed sequence length $L$ is enforced. Here, $L = 45$ is used. If $T_s < L$, pre-padding is applied. The initial part of the sequence is padded with zero-valued frames and repeated initial parameter values to reach length $L$:

<div class="equation">
$$\{I_{s,1}^{\text{norm}}, \ldots, I_{s,T_s}^{\text{norm}}\} \rightarrow \{\underbrace{0, \ldots, 0}_{L-T_s}, I_{s,1}^{\text{norm}}, \ldots, I_{s,T_s}^{\text{norm}}\}$$
</div>

For parameters:

<div class="equation">
$$\{P_{s,1}^{\text{norm}}, \ldots, P_{s,T_s}^{\text{norm}}\} \rightarrow \{\underbrace{P_{s,1}^{\text{norm}}, \ldots, P_{s,1}^{\text{norm}}}_{L-T_s}, P_{s,1}^{\text{norm}}, \ldots, P_{s,T_s}^{\text{norm}}\}$$
</div>

<h3>Augmentation</h3>

The proponents diversified the training data by applying random transformations to the input images during training. These transformations include horizontal flips or small rotations. If $\alpha$ is the probability of applying augmentation, each image $I_{s,t}^{\text{norm}}$ becomes $I_{s,t}^{\text{aug}}$ with probability $\alpha$. Here, $\alpha = 1$ is used.

<h2>Model Building</h2>

After all preprocessing steps, each sequence is represented as:

Images: $X_s = \{X_{s,1}, \ldots, X_{s,L}\}$, where each $X_{s,t}$ is the normalized (and augmented) image.

Parameters: $Q_s = \{Q_{s,1}, \ldots, Q_{s,L}\}$, where each $Q_{s,t} = P_{s,t}^{\text{norm}}$.

For one-step-ahead prediction, the first $L - 1$ frames and parameters are used as input, and the $L^{\text{th}}$ frame is the target. Thus:

<div class="equation">
$$X_s^{\text{input}} = \{X_{s,1}, \ldots, X_{s,L-1}\}, \quad Q_s^{\text{input}} = \{Q_{s,1}, \ldots, Q_{s,L-1}\}, \quad X_s^{\text{target}} = X_{s,L}$$
</div>

In this manner of structuring data, the model learns to predict the next frame given a sequence of past frames and corresponding parameters.

<h3>Sequence Learning</h3>

To capture the temporal evolution of plant growth and the spatial structure of each frame, the proponents employ a Convolutional LSTM (ConvLSTM) architecture. It replaces the fully connected operations inside LSTM cells with convolutions that make it well-suited for spatial data. At each timestep $t$, the ConvLSTM updates its hidden and cell states $(H_t, C_t)$ based on the input $X_{s,t}^{\text{input}}$ and previous states $(H_{t-1}, C_{t-1})$:

<div class="equation">
$$(H_t, C_t) = \text{ConvLSTM}(X_{s,t}^{\text{input}}, H_{t-1}, C_{t-1})$$
</div>

Here, 2 ConvLSTM layers are employed. After processing the $L - 1$ input frames, a sequence of hidden states $\{H_1, \ldots, H_{L-1}\}$ is obtained that captures how the plant's appearance evolves over time.

<h3>Feedforward Mapping</h3>

In parallel, the parameter sequences $Q_{s,t}^{\text{input}}$ are passed through a time-distributed dense layer:

<div class="equation">
$$Z_{s,t} = \text{ReLU}(W_p Q_{s,t}^{\text{input}} + b_p)$$
</div>

Each parameter vector is mapped into a feature embedding $Z_{s,t}$. This embedding is then broadcast (tiled) across the spatial dimensions $(H, W)$ to produce $Z_{s,t}^{\text{tile}}$.

<h3>Feature Fusion</h3>

By concatenating $Z_{s,t}^{\text{tile}}$ with the ConvLSTM output $H_t$, the model integrates both the visual and parametric modalities that allow the model to directly correlate changes in plant structure with environmental conditions:

<div class="equation">
$$Y_t = \text{concat}(H_t, Z_{s,t}^{\text{tile}})$$
</div>

<h3>RGB Mapping</h3>

A final 3D convolutional layer processes $\{Y_1, \ldots, Y_{L-1}\}$ to predict the next frame:

<div class="equation">
$$\hat{X}_{s,L} = \sigma(\text{Conv3D}(Y_{1:L-1}))$$
</div>

where $\sigma$ is a sigmoid activation ensuring the predicted frame lies within $[0,1]$.

<h3>Deep Learning Structure</h3>

<figure>
  <img src="/img/projects/plant-growth-multimodal-time-series-model/deep-learning-structure.jpg">
  <figcaption>Deep learning structure. $t$ refers to time step, $temp$ refers to ambient temperature, $moist$ refers to soil moisture, $pH$ refers to pH level, $lum$ refers to luminosity.</figcaption>
</figure>

<h3>Architecture Summary</h3>

<h3>Loss Function</h3>






<h2>Model Evaluation</h2>

<h3>Evaluation Metrics</h3>

<h3>Evaluation Flowchart</h3>

<h3>Training Loss Curves</h3>

<h3>Training Metric Curves</h3>

<h3>Quantitative Testing Results</h3>

<h3>Qualitative Testing Results</h3>
