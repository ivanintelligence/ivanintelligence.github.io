---
layout: subpage
hero: /img/projects/2d-to-3d-image-reconstruction-computer-vision-algorithm/2d-to-3d-image-reconstruction-computer-vision-algorithm.webp
---

<title>Developing Custom Computer Vision Algorithm for Reconstructing 3D Models from Constrained 2D Stereoscopic Images Using OpenCV and Open3D</title>

By John Ivan Diaz

A computer vision algorithm that reconstructs a 3D model from constrained 2D stereoscopic images captured from the front, left, back, and right of an object. Unlike traditional photogrammetry that requires many angles, this approach is designed to address limited views. It combines standalone computer vision techniques into a unified process, leveraging OpenCV and Open3D libraries. It is useful in applications such as plant phenotyping, where capturing multiple angles may be challenging yet still sufficient for constructing 3D models for visual inspection and analysis.

<tag>Computer Vision</tag>
<tag>3D Reconstruction</tag>
<tag>Point Clouds</tag>
<tag>OpenCV</tag>
<tag>Python</tag>

<a href="https://github.com/ivanintelligence/developing-custom-computer-vision-algorithm-for-reconstructing-3d-models-from-constrained-2d-images" class="arrow-link">See source code</a>

<hr class="hr-custom">
<br>

<h1>Discovery Phase</h1>

Three-dimensional (3D) models have become integral to modern computing, offering a sophisticated means of representing objects with spatial depth defined by x, y, and z coordinates. Unlike two-dimensional (2D) images, which rely solely on x and y coordinates, 3D models enable viewers to observe objects from all angles within a digital environment, closely emulating the experience of examining physical objects in the real world. This capability is facilitated by data structures such as point clouds, which consist of numerous points in 3D space, each specified by x, y, and z coordinates, effectively mapping the location and shape of objects within a scene.

Various technological methods exist to convert 2D images into 3D models, such as photogrammetry and LiDAR (Light Detection and Ranging) scanning. Photogrammetry involves constructing a 3D model by processing multiple 2D images, captured from different viewpoints around an object, utilizing techniques such as feature matching and triangulation to reconstruct the object's geometry. LiDAR scanning, in contrast, generates 3D models by emitting laser pulses toward an object and measuring the time it takes for the light to reflect back, thereby determining the distance to the object's surfaces and creating a precise spatial map.

Despite the advancements in these techniques, there remains a lack of effective methodologies for reconstructing 3D models from 2D stereo images captured exclusively from the front, back, left, and right sides of an object at 90-degree horizontal rotations. This scenario differs from traditional photogrammetry, as the images are not taken from a continuous range of angles around the object, and it utilizes stereo vision.

To address this gap, this project aims to integrate existing independent computer vision techniques into a unified process tailored to the challenges associated with these specific 2D stereo input images.

<h1>Development Phase</h1>

<h3>Input</h3>

<figure style="--img-max: 560px;">
  <img src="/img/projects/2d-to-3d-image-reconstruction-computer-vision-algorithm/stereo-images.webp">
  <figcaption>Four stereo pairs, each capturing 90-degree horizontal rotations around the plant.</figcaption>
</figure>

<h3>Calibration Map</h3>

<div>
A series of planar checkerboard images is captured that are displaced arbitrarily within the field of view. By detecting corresponding image points $(u_i, v_i)$ and known object points $(X_i, Y_i, 0)$, they estimate each camera's intrinsic matrix:
</div>

<div class="equation">
$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
</div>

<div>
and distortion coefficients $(k_1, k_2, p_1, p_2)$ via minimization of the reprojection error:
</div>

<div class="equation">
$$\min_{K,\mathbf{r},\mathbf{t}}\sum_{i}\|\mathbf{m}_i - \pi(K, [\mathbf{r} \mid \mathbf{t}][X_i, Y_i, 0, 1]^{\top})\|^2$$
</div>

<div>
where $\pi(\cdot)$ denotes the pinhole projection with radial and tangential distortion.<br><br>
</div>

<div>
Fixing intrinsics, stereo calibration then solves for the rotation $R$ and translation $T$ between cameras by optimizing:
</div>

<div class="equation">
$$\min_{R,T}\sum_{i}\|\mathbf{x}_{R,i} - \pi(K_R, [R \mid T]\mathbf{X}_i)\|^2$$
</div>

<div>
subject to the epipolar geometry.<br><br>
</div>

<div>
Finally, stereo rectification produces remapping functions stereoMapL$_x$, stereoMapL$_y$ and stereoMapR$_x$, stereoMapR$_y$, where each pair of images is row-aligned for depth estimation.
</div>

<h3>Depth Map Generation</h3>

<div>
With rectified image pairs, a block-matching algorithm is employed to compute the disparity:
</div>

<div class="equation">
$$d(u,v) = u_{\text{left}} - u_{\text{right}}$$
</div>

<div>
Depth is recovered via the relation:
</div>

<div class="equation">
$$Z(u,v) = \frac{f B}{d(u,v)}$$
</div>

<div>
where $f$ is the focal length in pixels and $B$ is the baseline. The resulting depth map $Z(u,v)$ encodes per-pixel distances and is normalized and filtered (e.g., bilateral filtering) to preserve edges while reducing noise.
</div>

<figure style="--img-max: 560px;">
  <img src="/img/projects/2d-to-3d-image-reconstruction-computer-vision-algorithm/depth-map-generation.webp">
  <figcaption>Extracting information about the plant's position along the z-axis.</figcaption>
</figure>

<h3>Point Cloud Generation</h3>

<div>
Each filtered depth map is reprojected to 3D using the rectification matrix $Q$:
</div>

<div class="equation">
$$\begin{bmatrix} X \\ Y \\ Z \\ W \end{bmatrix} = Q \begin{bmatrix} u \\ v \\ d(u,v) \\ 1 \end{bmatrix}, \quad (x,y,z) = \left(\frac{X}{W}, \frac{Y}{W}, \frac{Z}{W}\right)$$
</div>

<div>
For every valid mask pixel, 3D coordinates $(x,y,z)$ and corresponding RGB color are recorded, and the combined $(x,y,z,R,G,B)$ data is exported into PLY files. This yields four viewpoint-specific point clouds (front, right, back, left).
</div>

<figure style="--img-max: 560px;">
  <img src="/img/projects/2d-to-3d-image-reconstruction-computer-vision-algorithm/point-cloud-generation.webp">
  <figcaption>Building the foundational structure for 3D objects.</figcaption>
</figure>

<h3>Point Cloud Refinement</h3>

<div>
To remove spurious points caused by matching errors and noise, statistical outlier removal is applied, discarding any point whose average distance to its $k$ nearest neighbors exceeds:
</div>

<div class="equation">
$$\mu + \alpha \sigma$$
</div>

<div>
where $\mu$ and $\sigma$ are the mean and standard deviation of neighbor distances, respectively, and $\alpha$ is a user-defined threshold. Radius-based filtering then eliminates all points with fewer than $n$ neighbors within a radius $r$.<br><br>
</div>

<div>
Following cleaning, each point cloud is registered to a common frame via the rigid-body transformation:
</div>

<div class="equation">
$$T = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}$$
</div>

<div>
where $R \in \mathbb{R}^{3 \times 3}$ is the rotation matrix and $\mathbf{t} \in \mathbb{R}^3$ is the translation vector, with a uniform scaling factor applied to maintain dimensional consistency.<br><br>
</div>

<div>
To correct residual curvature artifacts, an affine warp is applied:
</div>

<div class="equation">
$$T_{\text{warp}} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & \gamma & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
</div>

<div>
where $\gamma$ is a tunable shear parameter that straightens remaining bending before merging.
</div>

<figure style="--img-max: 560px;">
  <img src="/img/projects/2d-to-3d-image-reconstruction-computer-vision-algorithm/point-cloud-refinement.webp">
  <figcaption>Fixing warping and outlier issues in point clouds.</figcaption>
</figure>

<h3>Point Cloud Registration</h3>

<div>
After individual refinement, the four point clouds are aligned into a common reference frame by applying the respective transforms $T_{\text{front}}$, $T_{\text{back}}$, $T_{\text{left}}$, and $T_{\text{right}}$. The merged point cloud is then expressed as the union:
</div>

<div class="equation">
$$\mathcal{P}_{\text{merged}} = T_{\text{front}}\mathcal{P}_{\text{front}} \cup T_{\text{back}}\mathcal{P}_{\text{back}} \cup T_{\text{left}}\mathcal{P}_{\text{left}} \cup T_{\text{right}}\mathcal{P}_{\text{right}}$$
</div>

<div>
A final round of statistical and radius outlier removal is performed on $\mathcal{P}_{\text{merged}}$, to yield a clean representation of the plant's structure.
</div>

<figure style="--img-max: 560px;">
  <img src="/img/projects/2d-to-3d-image-reconstruction-computer-vision-algorithm/point-cloud-registration.webp">
  <figcaption>Connecting front, back, left, and right point clouds into one.</figcaption>
</figure>

<h3>Point Cloud Meshing</h3>

<div>
From the downsampled and normalized merged point cloud, a triangular mesh is generated using the Ball Pivoting Algorithm (BPA). The mean nearest-neighbor distance $\bar{d}$ is estimated, and a sequence of radii $\{r, 10r\}$ is defined with:
</div>

<div class="equation">
$$r = 2\bar{d}$$
</div>

<div>
BPA iteratively pivots spheres of radius $r$ around edges of existing triangles, forming new triangles whenever the sphere touches exactly three points.
</div>

<figure style="--img-max: 560px;">
  <img src="/img/projects/2d-to-3d-image-reconstruction-computer-vision-algorithm/point-cloud-meshing.webp">
  <figcaption>Applying smooth continuous surface to point cloud.</figcaption>
</figure>

<h3>3D Reconstruction Algorithm</h3>

<figure>
  <img src="/img/projects/2d-to-3d-image-reconstruction-computer-vision-algorithm/full-process.webp">
  <figcaption>Visual results of reconstructing stereo images into a 3D model.</figcaption>
</figure>