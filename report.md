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
[image1a]: ./examples/undistort_cal.png "Undistorted call"
[image1]: ./examples/undistort.png "Undistorted"
[image2]: ./examples/grad1thresholding.png "Road threshold"
[image3]: ./examples/combinedthresh1.png "combinedthresh1"
[image4]: ./examples/combinedthresh2.png "combinedthresh1"
[image5]: ./examples/colorthresholding.png "colorthresholding"
[image6]: ./examples/warp_points.png" "Warp src points"
[image7]: ./examples/warp_thresh.png "Warp Example"
[image8]: ./examples/warp_thresh2.png "Warp Example2"
[image9]: ./examples/histogram.png "Histogram"
[image10]: ./examples/lane_output1.png "Output1"
[image11]: ./examples/lane_output2.png "Output1"


[video1]: ./output_videos/project_video_out.mp4 "Video"


### Camera Calibration

#### 1. Computation of the camera matrix and distortion coefficients and example of a distortion corrected calibration image.

The code for this step is contained in second and third cells of the IPython notebook located in "camera_calibrate.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Here is an example of how to draw the corners with the openCV function :     
`ret, corners = cv2.findChessboardCorners(gray, (9,6),None)`   


![alt text][image0]    

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1a]

### Pipeline (single images)

#### 1. An example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:   
![alt text][image1]  

I reload the pickled data after one time calibration for undistortion

#### 2a. Use of various gradients. 

I used 4 gradient thresholding :  
1. sobel x direction
2. sobel y direction
3. Magnitude thresholding
4. Direction thresholding   

I then combined these to get best results with the formula logic of :     
`     combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
`
I experimented with a lot of thresholding values and kernel size and finally chose the one written in the code cells under the heading of "Combining the thresholds of gradients".

Here are the examples :   

![alt text][image2]

This is final combined images :
![alt text][image3]
![alt text][image4]



#### 2b. Use of color transforms.

I used 3 gradient thresholding :  
1. s color from the HLS color scheme.
2. l channel from LUV color scheme
3. b channel from LAB color scheme    (To detect yellow line)

I then combined these to get best results with the formula logic of :     
`        combined_binary[(s_binary == 1) | (l_binary == 1) | (b_binary == 1)] = 1
`
I experimented with a lot of thresholding values and finally chose the one written in the code cells under the heading of "Color filters and thresholding".


#### 2c. Merging of color transforms and gradients to create a thresholded binary image.  


Finally I combined both color filtering and gradient thresholding to get a binary image clearly showing the lane lines.
The following image illustrates the whole process :

![alt text][image5]   


#### 3. Description of perspective transform.

The code for my perspective transform includes a function called `warp_img()`, which appears under the cell heading of "Warping - Perpesctive Transform and Birds Eye View" in the file `advance lane keeping.ipynb`. The `warp_img()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
    horizon = np.uint(2*image_size[0]/3)
    center_lane = np.uint(image_size[1]/2)
    offset = 0.2

    x_left_bottom = center_lane - center_lane
    x_right_bottom = 2*center_lane
    x_right_upper = center_lane + offset*center_lane
    x_left_upper = center_lane - offset*center_lane


    src_points = np.float32([
                           [x_left_bottom, y],
                           [x_right_bottom, y],
                           [x_right_upper, horizon],
                           [x_left_upper, horizon]
                          ])

    dst_points = np.float32([
                            [0.15*x,y],
                            [x - 0.15*x,y],
                            [x - 0.15*x,0],
                            [0.15*x,0]
                            ])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
|  0, 720       | 192, 720       | 
| 1280, 720     | 1088, 720    |
| 768, 480      | 1088, 0       |
| 512, 480      | 192, 0         |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]

#### 4. Identification of lane-line pixels and fit their positions with a polynomial.


![alt text][image7]

![alt text][image8]   

![alt text][image9]   


#### 5. Calculation of the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in under the cell heading of "LaneLines" in the file `advance lane keeping.ipynb` 

#### 6. Example image of result plotted back down onto the road where the lane area is identified clearly.

I implemented this step under the cell heading of "Lane Lines" in the file `advance lane keeping.ipynb` in the function `draw_lane()`.  Here is an example of my result on a test image:

![alt text][image10]    
![alt text][image11]

---

### Pipeline (video)

#### 1. Link to final video output.  Pipeline performs reasonably well on the entire project video.

Here's a [link to my video result](./output_videos/project_video_out.mp4)

---

### Discussion

#### 1. Problems / issues faced in implementation of this project. 

I want to further develop adaptive thresholding techniques when there is a lot of darkness or sunishine or rain.  
Pipeline will fail when there are a lot of drastic curves  while driving the car very fast.   
Sometimes the pavements on the side confuse the pipeline.  
I want to make the code more robust for various kinds of lane lines like zig zag and also support it when there is no one side of lane (maybe use pavement for alternate detection). Optimization can also be done for various elevation.  
