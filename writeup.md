##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/hog_visualization_features.png
[image3]: ./output_images/test1.png
[image4]: ./output_images/test2.png
[image5]: ./output_images/test3.png

[video1]: ./processed_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. This README includes information on how I addressed all the rubric points defined [Here](https://review.udacity.com/#!/rubrics/513/view) for the Vehicle detection project. 

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.


I started by reading in all the `vehicle` and `non-vehicle` images from the dataset provided .  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I used the `skimage.hog()` parameter which helps extract the Histogram of Oriented Gradients (HOG) features for a given image.I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` , `cells_per_block=(2, 2)` and `visualise=True`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and used the HOG visualization feature on some test images. The combination of parameters worked best both for positive detection and reduced false positives.  
`color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9 # HOG orientations
pix_per_cell = 12 # HOG pixels per cell 
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"`
I also found that enabling the `transform_sqrt` to apply power law compression to normalize the images before processing gave better results at positive detection of images.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using StandardScaler to normalize the images.. I trained it on all samples of car/ not car images. The linear SVC was sufficiently good at classifying the batch of test data that was split from the input samples of car/ not car images. I verfied this by calculating the accuracy.

Using: 9 orientations 12 pixels per cell and 2 cells per block
Feature vector length: 4896
9.53 Seconds to train SVC...
Test Accuracy of SVC =  0.989
Saving classifier, scaler and parameters into file:  svc_classifier_pickle.p

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

 To implement a sliding window search, I had to decide what size window tile to use to search on the image, along with where to start and stop the search, and how much of the windows needed to overlap. This approach extract features at each window position, and predict with your classifier on each set of features.The drawback is that this could lead to timeouts on a larger sample of images as input. I instead settled to use the Hog Sub-sampling Window Search, which is a more efficient method for doing the sliding window approach, one that allows us to only have to extract the Hog features once. Here is an example test image with a hog sub-sampling window approach used. I optimized this to apply the feature extraction limited only to the region where the cars appear in the images using (ystart = 336 and ystop = 700 parameters). 


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example test images and corresponding output images derived after applying the pipeline:

![alt text][image3]

![alt text][image4]

![alt text][image5]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

![alt text][vedio1]

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I record the positions of positive detections in each frame of the video， then created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

 I tried to minimise the false positives after the initial classification and cleaning up the noisy results. There are instances where the bounded box is still not overlapping the vehicle completed in some instances which can be improved in the solution. The Linear SVC classifier works well but could still be improved by augmenting the input data. The labeling process and constructing bounding boxes based on blobs in the heatmap has worked well most of the time but there are instances, noticeable in the video, where boxes do not completly overlap the vehicle.
To make it more robust ，I think we should use deep learning to learn more robust features.
