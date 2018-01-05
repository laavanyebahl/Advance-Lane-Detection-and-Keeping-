**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./examples/draw_lines.png "Cal Lines"
[image1a]: ./examples/undistort_output.png "Undistorted call"
[image1]: ./examples/undistort.png "Undistorted"
[image2]: ./examples/thresholding.png "Road threshold"
[image3]: ./examples/hlsgrad.png "HLS + gradient"
[image4]: ./examples/warped_straight_lines.jpg "Warp straight lines"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/lane_output.png "Output"
[image7]: ./examples/warp_thresh.png "Warp Example"
[image8]: ./examples/histogram.png "Histogram"

[video1]: ./output_videos/project_video_out.mp4 "Video"


### Camera Calibration

#### 1. Computation of the camera matrix and distortion coefficients and example of a distortion corrected calibration image.

The code for this step is contained in one of the first code cell of the IPython notebook located in "advance lane keeping.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

![alt text][image0]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1a]

### Pipeline (single images)

#### 1. An example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image1]

#### 2. Used color transforms, gradients or other methods to create a thresholded binary image.  Here is an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at code cells under the heading "Color and Gradient thresholding combined" in `advance lane keeping.ipynb`).  Here's an example of my output for this step. 

![alt text][image2]   

![alt text][image3]


#### 3. Description of perspective transform and an example of a transformed image.

The code for my perspective transform includes a function called `warp_img()`, which appears under the cell heading of "Warping - Perpesctive Transform and Birds Eye View" in the file `advance lane keeping.ipynb`. The `warp_img()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src_points = np.float32([                    
                [0.177 * x, y],
                [(0.5 * x) - (x*0.055), (2/3)*y],
                [(0.5 * x) + (x*0.055), (2/3)*y],
                [x - (0.177 * x), y]
                ])


dst_points = np.float32([
                [0.25 * x, y],
                [0.25 * x, 0],
                [x - (0.25 * x), 0],
                [x - (0.25 * x), y]
                ])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Identification of lane-line pixels and fit their positions with a polynomial.

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]   

![alt text][image7]

![alt text][image8]   



#### 5. Calculation of the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in under the cell heading of "Lane Lines" in the file `advance lane keeping.ipynb`  in the function `draw_lane()`

#### 6. Example image of result plotted back down onto the road where the lane area is identified clearly.

I implemented this stepunder the cell heading of "Lane Lines" in the file `advance lane keeping.ipynb` in the function `draw_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Link to final video output.  Pipeline performs reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures causes the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video_out.mp4)

---

### Discussion

#### 1. Problems / issues faced in implementation of this project. 

I want to further develop adaptive thresholding techniques when there is a lot of darkness or sunishine or rain.  
Pipeline will fail when there are a lot of drastic curves  while driving the car very fast.   
Sometimes the pavements on the side confuse the pipeline.  
I want to make the code more robust for various kinds of lane lines like zig zag and also support it when there is no one side of lane (maybe use pavement for alternate detection). Optimization can also be done for various elevation.  
