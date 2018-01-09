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
[image1]: ./examples/undistort1.png "Undistorted"
[image2]: ./examples/gradthresholding1.png "Road threshold"
[image3]: ./examples/combinedthresh1.png "combinedthresh1"
[image4]: ./examples/combinedthresh2.png "combinedthresh1"
[image5]: ./examples/colorthresholding.png "colorthresholding"
[image6]: ./examples/warp_points.png "Warp src points"
[image7]: ./examples/warp_thresh.png "Warp Example"
[image8]: ./examples/warp_thresh2.png "Warp Example2"
[image9]: ./examples/histogram.png "Histogram"
[image10]: ./examples/lane_lines.png "lanes"
[image11]: ./examples/lane_output3.png "Output1"
[image12]: ./examples/lane_output4.png "Output1"


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
Here is the vizualization - 

![alt text][image6]   


This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
|  0, 720       | 192, 720       | 
| 1280, 720     | 1088, 720    |
| 768, 480      | 1088, 0       |
| 512, 480      | 192, 0         |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.


![alt text][image7]

![alt text][image8] 


#### 4. Identification of lane-line pixels and fit their positions with a polynomial.
  
After calculating the warped image. First I calculate the histogram across the image, this was implemented using numpy.sum() to sum all the pixels on the rows, the result is the following:      

![alt text][image9]   

Then I passed the histogram to a function find_two_peaks_image to find the center of each of the peaks. This code can be found on the IPython notebook called `advance lane keeping.ipynb` on cell #15 and #16, then I implemented a function called sliding_window to iterate through the image and find the x coordinates and y coordinates of the pixels that corresponded to the lanes starting from the centers that I found earlier. Margin is set to be 100.

I also implemented a guided_search function to Obtain the coordinates of the pixels of a line starting at the same center of the last detected line to avoid doing the sliding window.

Once I got the pixels I fit them to a 2nd order polynomial of the form:      
`f(y) =  Ay^2 + By + C`

The result obtained for one of the test images is the following:    

![alt text][image10]   


#### 5. Calculation of the radius of curvature of the lane and the position of the vehicle with respect to center.

I defined the function `get_curvature_meters` under the cell heading of "Lane Lines Class for storing characteristics" in the file `advance lane keeping.ipynb`.

It is used to convert pixel curves to metres curve according to the assumption that lane width is 3.7m.        

Position is calculated in the function draw_lanes() with the formula to calculate the distance of the center of detected lanes from the center of the image -

```python
            center = und_image.shape[1]/2
            xm_per_pix = 3.7/700

            lanes_middle_distance = abs(right_lane.recent_xfitted[-1][0] + left_lane.recent_xfitted[-1][0])/2
            position_car_pixels = center - lanes_middle_distance 
            position_car_meters = position_car_pixels *xm_per_pix
 ```            

 Radius of curvature is calculated according to :    
 
```python
def get_curvature_meters(self, yvals, y_eval, ym_per_pix = 30/720, xm_per_pix = 3.7/700 ):

   side_fit_cr = np.polyfit(self.ally*ym_per_pix, self.allx*xm_per_pix, 2)
   side_curverad = ((1 + (2*side_fit_cr[0]*y_eval*ym_per_pix  + side_fit_cr[1])**2)**1.5) \
                       /np.absolute(2*side_fit_cr[0])


   return side_curverad
```
         
 ym_per_pixel and xm_per_pixel are the factors used for converting from pixels to meters. This conversion was also used to generate a new fit with coefficients in terms of meters.
 
 which is calculated as :-      
 
 xm_per_pixel = 3.675 / 700 (3.675 actual width of lane, 700 px on warped image)
 ym_per_pixel = 3.048 / 190 (3 m actual length of dashed line, 190 px length on warped image)
 
#### 6. Example image of result plotted back down onto the road where the lane area is identified clearly.

I implemented this step under the cell heading of "Lane Lines" in the file `advance lane keeping.ipynb` in the function `draw_lane()`.  Here is an example of my result on a test image:

![alt text][image11]  

![alt text][image12]

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
