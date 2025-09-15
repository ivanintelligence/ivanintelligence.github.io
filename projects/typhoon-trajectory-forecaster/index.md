---
layout: subpage
hero: /img/projects/typhoon-trajectory-forecaster/typhoon-trajectory-forecaster.jpeg
---

<title>Forecasting Typhoon Trajectory Using Interpolation and Extrapolation</title>

A simple desktop application that predicts the next path of a typhoon based on observed data points of its previous trajectory. Python and PyQt were used for full-stack development. Although the forecasting is limited to historical path data and does not account for other factors influencing a typhoon’s movement, the application is primarily designed to illustrate the concepts of interpolation and extrapolation.

<tag>Time Series Forecasting</tag>
<tag>Python</tag>
<tag>Data Visualization</tag>
<tag>Software Development</tag>
<tag>Data Preprocessing</tag>

<a href="https://github.com/ivanintelligence/forecasting-typhoon-trajectory-using-interpolation-and-extrapolation" class="arrow-link">See source code</a>

<br>

{% include video.html 
   url="/img/projects/typhoon-trajectory-forecaster/typhoon-trajectory-forecaster.mp4"%}

<h1>Discovery Phase</h1>

Tropical cyclones, formidable storm systems characterized by low pressure, intense winds, and heavy rainfall, are intricately tied to the Earth's geographical features. The planet's latitude and longitude lines define the specific locations where tropical cyclones form, with warm ocean waters serving as their incubators. Earth's division into the Northern and Southern Hemispheres plays a crucial role in the behavior of these storms, as the Coriolis effect influences wind directions differently in each hemisphere. This hemispheric distinction is pivotal when discussing tropical cyclones, as it determines their rotation and trajectory. In the Northern Hemisphere, these storms are known as typhoons or hurricanes, while in the Southern Hemisphere, they are referred to simply as cyclones. This nomenclatural variation reflects the importance of understanding the hemispheric context in comprehending the development and movement of these powerful meteorological phenomena.

This project involves a simulation for mapping the trajectory of tropical cyclones. Its goal is to monitor the changes in longitude and latitude distances of a tropical cyclone over time, from the past to the future. The program gathers historical speed data of the cyclone at specific times and smartly predicts speeds between and beyond the given data points using interpolation and extrapolation techniques. By combining this speed information with predefined time intervals, the program calculates distances traveled, providing insights into the cyclone's position in terms of both longitude and latitude. The results are then visually presented on a graph, offering users a clear understanding of the cyclone's movement.

<h1>Development Phase</h1>

<h3>Features</h3>

The program can map the historical and future trajectory of a tropical cyclone in terms of both longitude and latitude. It first collects speed data at intervals: 18 hours ago, 12 hours ago, 6 hours ago, and the current speed. To fill in the gaps between and beyond these timestamps (where speeds are unknown), the program employs interpolation (for in-between data) and extrapolation (for data outside the provided timestamps) using the divided difference method at 3-hour intervals. It calculates distance by multiplying speed and a fixed time value of 3 hours. From the distance data, the program determines longitude and latitude metrics. The results are then displayed graphically, with labels indicating specific time points like 3 hours ago or 1 day later. The user can hover the cursor over the graph to track and view precise longitude and latitude values, providing an interactive way to interpret the typhoon's accurate position at different times.

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/features.jpg">
</figure>

<h3>Limitations</h3>

The application doesn't display precise Earth coordinates of the tropical cyclone's current location and direction. Instead of mapping the trajectory on a real-world map, it simply offers insights into the cyclone's distance over time in terms of longitude and latitude. The simulation also doesn't consider certain real-world factors, like the presence of other nearby cyclones or the potential for landfall, making the focus on the cyclone's general movement rather than accounting for specific conditions.

<h3>Input</h3>

The program involves the acquisition of crucial inputs through a user prompt-driven system. The program solicits specific information related to a tropical cyclone, focusing on two key parameters:

<ol>
  <li>
    Tropical Cyclone Speed:<br>
    Speed readings are gathered at distinct time intervals: 18 hours ago, 12 hours ago, 6 hours ago, and the present moment (Now).
  </li>
</ol>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/input-1.jpg">
</figure>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/input-2.jpg">
</figure>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/input-3.jpg">
</figure>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/input-4.jpg">
</figure>

<ol start="2">
  <li>
    Hemisphere Location:<br>
    Users are prompted to specify the hemisphere in which the tropical cyclone is situated, providing options for either the Northern or Southern Hemisphere.
  </li>
</ol>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/input-5.jpg">
</figure>

This aims to enhance the understanding of tropical cyclone dynamics, facilitating comprehensive analyses of speed variations and their correlation with specific hemispheric locations over time.

<h3>Output</h3>

The program generates a graphical representation that conveys valuable insights to the users, comprising:

<ul>
  <li>
    Longitudinal distance of the tropical cyclone over time, including past occurrences, the current moment, and future projections at 3-hour intervals.
  </li>
  <li>
    Latitudinal distance of the tropical cyclone over time, including past occurrences, the current moment, and future projections at 3-hour intervals.
  </li>
</ul>

The graphical representation provides a comprehensive overview, allowing for a nuanced examination of the tropical cyclone's distance behavior in both longitudinal and latitudinal dimensions over time.

<h3>Algorithm</h3>

<ol>
  <li>
    Main function
  </li>
</ol>

<figure style="--img-max: 640px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/algorithm-1.jpg">
</figure>

<figure style="--img-max: 640px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/algorithm-2.jpg">
</figure>

<figure style="--img-max: 640px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/algorithm-3.jpg">
</figure>

<figure style="--img-max: 640px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/algorithm-4.jpg">
</figure>

<ol start="2">
  <li>
    DividedDifferencePart1 function
  </li>
</ol>

<figure style="--img-max: 640px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/algorithm-5.jpg">
</figure>

<ol start="3">
  <li>
    DividedDifferencePart2 function
  </li>
</ol>

<figure style="--img-max: 640px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/algorithm-6.jpg">
</figure>

<h3>Sample Input-Output</h3>

Overview of where inputs are requested and where interpolation and extrapolation are carried out:

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/sample-input-output.jpg">
</figure>

Scenario 1:

<ul>
  <li>
    Speed at 18 hours ago: 55 km
  </li>
  <li>
    Speed at 12 hours ago: 65 km
  </li>
  <li>
    Speed at 6 hours ago: 75 km
  </li>
  <li>
    Speed at present moment: 85 km
  </li>
</ul>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/scenario-1-1.jpg">
  <figcaption>Table of values of scenario 1.</figcaption>
</figure>

<ul>
  <li>
    Hemisphere location: Southern hemisphere
  </li>
</ul>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/scenario-1-2.jpg">
  <figcaption>Graphical representation of scenario 1 when hemisphere location is south.</figcaption>
</figure>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/scenario-1-3.jpg">
  <figcaption>Graphical representation of scenario 1 in the program when hemisphere location is south.</figcaption>
</figure>

<ul>
  <li>
    Hemisphere location: Northern hemisphere
  </li>
</ul>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/scenario-1-4.jpg">
  <figcaption>Graphical representation of scenario 1 when hemisphere location is north.</figcaption>
</figure>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/scenario-1-5.jpg">
  <figcaption>Graphical representation of scenario 1 in the program when hemisphere location is north.</figcaption>
</figure>

Scenario 2:

<ul>
  <li>
    Speed at 18 hours ago: 95 km
  </li>
  <li>
    Speed at 12 hours ago: 115 km
  </li>
  <li>
    Speed at 6 hours ago: 120 km
  </li>
  <li>
    Speed at present moment: 210 km
  </li>
</ul>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/scenario-2-1.jpg">
  <figcaption>Table of values of scenario 2.</figcaption>
</figure>

<ul>
  <li>
    Hemisphere location: Southern hemisphere
  </li>
</ul>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/scenario-2-2.jpg">
  <figcaption>Graphical representation of scenario 2 when hemisphere location is south.</figcaption>
</figure>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/scenario-2-3.jpg">
  <figcaption>Graphical representation of scenario 2 in the program when hemisphere location is south.</figcaption>
</figure>

<ul>
  <li>
    Hemisphere location: Northern hemisphere
  </li>
</ul>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/scenario-2-4.jpg">
  <figcaption>Graphical representation of scenario 2 when hemisphere location is north.</figcaption>
</figure>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/scenario-2-5.jpg">
  <figcaption>Graphical representation of scenario 2 in the program when hemisphere location is north.</figcaption>
</figure>

Scenario 3:

<ul>
  <li>
    Speed at 18 hours ago: 70 km
  </li>
  <li>
    Speed at 12 hours ago: 90 km
  </li>
  <li>
    Speed at 6 hours ago: 95 km
  </li>
  <li>
    Speed at present moment: 90 km
  </li>
</ul>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/scenario-3-1.jpg">
  <figcaption>Table of values of scenario 3.</figcaption>
</figure>

<ul>
  <li>
    Hemisphere location: Southern hemisphere
  </li>
</ul>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/scenario-3-2.jpg">
  <figcaption>Graphical representation of scenario 3 when hemisphere location is south.</figcaption>
</figure>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/scenario-3-3.jpg">
  <figcaption>Graphical representation of scenario 3 in the program when hemisphere location is south.</figcaption>
</figure>

<ul>
  <li>
    Hemisphere location: Northern hemisphere
  </li>
</ul>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/scenario-3-4.jpg">
  <figcaption>Graphical representation of scenario 3 when hemisphere location is north.</figcaption>
</figure>

<figure style="--img-max: 560px;">
  <img src="/img/projects/typhoon-trajectory-forecaster/scenario-3-5.jpg">
  <figcaption>Graphical representation of scenario 3 in the program when hemisphere location is north.</figcaption>
</figure>