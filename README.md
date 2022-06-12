# Feature Detection and Tracking - [Computer Vision Final Assignment]

---

# Table of Contents
1. [What is Feature Detection?](#1-what-is-feature-detection)
2. [Workflow](#2-workflow)
3. [Results](#3-results)
4. [Conclusions](#4-conclusions)
5. [Bibliography](#bibliography)

---

## 1. What is Feature Detection? 
In Computer Vision, a **_feature_** is a piece of information about the content of an image. Features are typically explored 
to assert whether a region of an image has certain properties.\
More broadly a feature is any piece of information which is relevant for solving the computational task related to a 
certain application.

Feature detection is a **low-level** image processing operation. That is, it is usually performed as the first operation 
on an image, and inspects pixels to see whether there is a feature present in that region or not. Because feature 
detection algorithms search into each and every pixel, they are usually employed as part of a larger algorithm. This way
the algorithm will typically only examine the image in the region of the features. As a built-in 
pre-requisite to feature detection, the input image is commonly smoothed by a Gaussian kernel in a scale-space 
representation and one or several feature images are computed, often expressed in terms of local image derivative 
operations and computed by means of operators such as the Laplacian, Sobel, Roberts and Prewitt [[1]](#bibliography).

Once features have been detected, they are (almost) ready to be extracted. Note that such an extraction may involve 
considerable amounts of resources, and for this reason there are techniques that are engineered specifically to reduce 
computational burden whilst maintaining a high level of accuracy.\
**Intensity** and **direction** of edges is one of the most salient features that can help us to characterize objects [[2]](#bibliography).

---

## 2. Workflow
Feature extractors used in this project are:
- [**AKAZE**](https://docs.opencv.org/4.x/db/d70/tutorial_akaze_matching.html): a local binary descriptor showing 
improved results in terms of speed and performance compared to state-of-the-art methods such as local feature descriptors.
- [**ORB**](https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html): it uses FAST to find keypoints, then applies 
Harris corner measure to find the top N keypoints. It is rotation invariant.
- [**SIFT**](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html): applies an increasingly intense Gaussian 
blur filter to the `train' image. It performs pixel-wise difference between octaves to look for the most intense 
variation in value. SIFT is both rotation and scale invariant.

Trackers used in this project are:
- [**BFM**](https://docs.opencv.org/4.x/d3/da1/classcv_1_1BFMatcher.html):
- [**FLANN**](https://docs.opencv.org/3.4/dc/de2/classcv_1_1FlannBasedMatcher.html):


Each and every tracker is tested with each and every feature extractor. Such an experiment is hopefully able to 
highlight any change in performance due to unique matches. **Performance** is evaluated by keeping track of both the 
**framerate** and the number of **good features** detected.\
Good features are obtained following the guidelines of Lowe's paper. 
In a nutshell, only the two best matches for each keypoint -- hence the two keypoints with the _smallest distance_ 
measurement -- are considered to be significant.
Lowe's test also checks that the two distances are _sufficiently different_. If they are not, then the keypoint is eliminated 
and will not be used for further calculations.

---

## 3. Results 

<table style="text-align: center">
<tr>
 <th>Detector</th>
 <th>Average fps</th>
 <th>Feats per Frame</th>
</tr>
<tr>
 <td>AKAZE + BFM</td>
 <td>1.41</td>
 <td>5006.70</td>
</tr>
<tr>
 <td>ORB + BFM</td>
 <td>7.36</td>
 <td>6754.30</td>
</tr>
<tr>
 <td>SIFT + BFM</td>
 <td>1.3</td>
 <td>3686.72</td>
</tr>
<tr></tr>
<tr>
 <td>AKAZE + FLANN</td>
 <td>0.65</td>
 <td>2008.15</td>
</tr>
<tr>
 <td>ORB + FLANN</td>
 <td>7.58</td>
 <td>8674.25</td>
</tr>
<tr>
 <td>SIFT + FLANN</td>
 <td>1.38</td>
 <td>1000.17</td>
</tr>
</table>

Whenever paired with BFM, detectors like SIFT and AKAZE are able to find a number of features in line with ORB. However,
in an online application they may struggle a lot with keeping up with the frame rate. For this reason we tried to set a 
_cap_ to the number of features to be detected, however both algorithms are still stuck at just above 1 fps.\
As mentioned in the Workflow section, we carried out our second experiment pairing each and every feature detector with 
FLANN matcher. The result yields values that are fairly consistent with BFM. SIFT and ORB are along the lines of their 
match with brutal force matcher as far as framerate is concerned. However, ORB shows a surge in the number of features 
it is able to detect. This is almost certainly due to the different approach in storing and displaying good matches. 
(see the [code](src/trackers) for reference).

---

## 4. Conclusions
To sum up, the performances of AKAZE, SIFT, and ORB are almost equivalent in terms of speed whenever it is the case 
that we are dealing with static material like images (see standalone files in the code). However, if we need to deal 
with an online scenario like the one in analysis, the gap in performance between ORB and any other detector analyzed 
becomes unbridgeable.\
One more thing which needs to be taken into consideration is the application field of each detector. For instance, 
comparing the `train` image of an object with a `query` image of another object of the same category differs a lot to 
comparing a `train` image of that object with a `query` image containing several other objects of the same category.\
Bearing this concept in mind, we compared subsequent frames of the same video containing fairly the same number of objects.

---

## Useful Links
- [Github](https://github.com/aambroo/Computer-Vision-Final-Assignment)
- [Demo Video](https://github.com/aambroo/Computer-Vision-Final-Assignment/tree/main/outputs)
- [Small Presentation Video](https://drive.google.com/file/d/1fRD-JyYXOSUcOdc-wXBpLXVveVpILnZu/view?usp=sharing)

## Bibliography
[1] Pratt, W.K., 2007. Digital image processing (4th ed.). John Wiley & Sons, Inc. pp. 465–522

[2] N. Conci, 2021-22. Computer Vision (ch.6): _Local Feature Extraction_, pp. 2-6