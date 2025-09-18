---
layout: subpage
hero: /img/projects/plant-segmentation-model/plant-segmentation-model.webp
---

<title>Comparing YOLOv8 and Detectron2 Architectures for Plant Class Segmentation Using Transfer Learning</title>

By John Ivan Diaz

Two deep learning models trained on popular segmentation architectures, YOLOv8 and Detectron2, for segmenting the plant class. A dataset of plant images with diverse types and backgrounds was collected, and both models were trained using identical hyperparameters, then adjusted in parallel to compare performance, training speed, and memory consumption. Quantitative metrics such as average precision, loss, training time, RAM, and disk usage were used for evaluation. Results show that YOLOv8 trains faster and maintains consistent performance, while Detectron2 achieves higher accuracy when using larger hyperparameter settings.

<tag>Object Detection</tag>
<tag>Transfer Learning</tag>
<tag>TensorFlow</tag>
<tag>Model Validation</tag>
<tag>Hyperparameter Tuning</tag>

<a href="https://github.com/ivanintelligence/comparing-yolov8-and-detectron2-architectures-for-plant-class-segmentation-using-transfer-learning" class="arrow-link">See source code</a>

<hr class="hr-custom">
<br>

<h1>Discovery Phase</h1>

<h2>Use Case Definition</h2>

Correct segmentation of plant objects can be useful to other domains requiring it, such as plant phenotyping. Threshold-based masking can be used to segment plant classes. However, as plants grow size and appearance change when working with multiple images, this static approach often fails, leading to incorrect segmentation when the plant falls outside the predefined thresholds.

To address this issue, this project investigated segmentation architectures that are solely trained for the plant class, determining strengths and weaknesses of each one that are helpful for plant segmentation tasks.

<h2>Data Exploration</h2>

Images of various plant types were used to increase diversity. These images were downloaded online, totaling over a thousand for training. Furthermore, plant objects were annotated using Roboflow.

<h2>Architecture and Algorithm Selection</h2>

YOLOv8 and Detectron2 were the architectures chosen for comparison. YOLOv8 is the latest version in the You Only Look Once (YOLO) family, featuring a CSPDarknet backbone, a PANet neck, and an anchor-free decoupled head, while Detectron2 is a framework developed by Facebook, which uses a two-stage pipeline: a Region Proposal Network (RPN) to generate candidate object regions, followed by a stage for bounding box and mask prediction.

<h1>Development Phase</h1>

<h2>Model Building</h2>

Three sets of decreasing hyperparameter values were tested on each architecture. From Set A to Set C, the project decreased the epoch sizes to 100, 75, and 50; the image input sizes to 800, 750, and 640 pixels; the batch sizes to 16, 8, and 3; and the worker counts to 8, 6, and 4, respectively. The initial and final learning rates were kept constant at 0.01 and 0.001, respectively.

<h2>Model Evaluation</h2>

<h3>Evaluation Metrics</h3>

Performance, training speed, and training memory consumption were compared between the models. Metrics for performance included Box Average Precision at a 50% IoU threshold (Box AP/50), Mask Average Precision at a 50% IoU threshold (Mask AP/50), Box Loss, and Mask Loss. Metric for training speed was train time per hour. Metrics for training memory consumption included system RAM consumption, GPU RAM consumption, and disk usage. Higher values of Box AP/50 and Mask AP/50 indicate better segmentation accuracy. Lower values of Box Loss and Mask Loss indicate better model performance. A lower train time indicates better training speed, while lower system RAM consumption, GPU RAM consumption, and disk usage indicate better training memory efficiency.

<h3>Evaluation Results</h3>

<figure style="--img-max: 480px;">
  <img src="/img/projects/plant-segmentation-model/evaluation-results.webp">
  <figcaption>Comparison of YOLOv8 and Detectron2 architectures for plant class segmentation across varying hyperparameter values.</figcaption>
</figure>

The table above shows the performance, training speed, and training memory consumption between YOLOv8 and Detectron2 architectures for plant class segmentation across varying hyperparameter values.

<h3>Performance</h3>

In terms of performance for Hyperparameter Set A, Detectron2 outperformed YOLOv8 in segmentation accuracy. Detectron2 achieved a Box AP/50 of 0.870 compared to YOLOv8's 0.850, and a Mask AP/50 of 0.880 compared to YOLOv8's 0.840. It also exhibited lower loss values, with a Box Loss of 0.180 compared to YOLOv8's 0.400, and a Mask Loss of 0.110 compared to YOLOv8's 0.800. As the hyperparameters decreased from Set A to Set C, YOLOv8 maintained relatively stable accuracy, with Box AP/50 decreasing slightly from 0.850 to 0.830 and Mask AP/50 from 0.840 to 0.830. In contrast, Detectron2 experienced a more substantial drop in performance. Its Box AP/50 decreased from 0.870 to 0.740, and Mask AP/50 dropped from 0.880 to 0.730, showing that its accuracy was more sensitive to reductions in hyperparameter values.

<h3>Training Speed</h3>

In terms of training speed for Hyperparameter Set A, YOLOv8 trained faster, completing training in 1.400 hours, compared to Detectron2's 1.590 hours. As the hyperparameters decreased, YOLOv8's training time continued to improve, reaching 0.870 hours in Set C. Detectron2, however, showed slower training speed as the values decreased. Its training time increased to 2.020 hours in Set C, which indicates less efficiency when trained with smaller hyperparameter configurations.

<h3>Training Memory Consumption</h3>

In terms of training memory consumption for Hyperparameter Set A, YOLOv8 consumed more GPU and system RAM, consuming 9.200 gigabytes of GPU RAM and 5.500 gigabytes of system RAM. Detectron2, in contrast, used only 3.500 gigabytes of GPU RAM and 4.600 gigabytes of system RAM. However, YOLOv8 required less disk storage, using 33.100 gigabytes compared to Detectron2's 35.900 gigabytes. As hyperparameters decreased, both models lowered their memory consumption. YOLOv8 reduced its GPU RAM usage to 5.200 gigabytes and its system RAM usage to 4.900 gigabytes. Detectron2 also reduced its GPU RAM to 2.500 gigabytes and system RAM to 4.000 gigabytes. Despite these reductions, Detectron2 consistently consumed more disk space, which remained around 35.900 gigabytes.

<h3>Strengths</h3>

YOLOv8 showed faster training speed, maintained stable segmentation accuracy even with fewer epochs, used less disk storage, and performed effectively with smaller hyperparameter settings. Detectron2 showed superior accuracy when trained with higher hyperparameter values and consistently consumed less GPU and system RAM across all settings.

<h3>Weaknesses</h3>

YOLOv8 showed slightly lower segmentation accuracy compared to Detectron2 and demanded higher GPU and system RAM, especially when trained with larger hyperparameters. Detectron2 suffered from significant performance drops when trained with smaller hyperparameters, had slower training times, and required consistently high disk storage throughout all experiments.