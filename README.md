# Feature Detection and Tracking - [Computer Vision Final Assignment]

# Table of Contents
1. [What is Feature Detection?](#1-what-is-feature-detection)
2. [Bibliography](#bibliography)

## 1. What is Feature Detection? 
In Computer Vision, a _feature_ is a piece of information about the content of an image. Features are typically explored 
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
computational burden.


# Bibliography
[1] Pratt, W.K., 2007. Digital image processing (4th ed.). John Wiley & Sons, Inc. pp. 465–522