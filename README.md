## Advanced Lane Finding and Keeping

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


Find the summary in - `report.md`    
Find the pipeline in - `advance lane keeping.ipynb`   
Find the calibration code in - `camera_calibrate.ipynb`



[//]: # (Image References)

[image0]: ./examples/draw_lines.png "Cal Lines"
[image1a]: ./examples/undistort_cal.png "Undistorted call"
[image1]: ./examples/undistort1.png "Undistorted"
[image2]: ./examples/gradthresholding1.png "Road threshold"
[image3]: ./examples/combinedthresh1.png "combinedthresh1"
[image4]: ./examples/combinedthresh2.png "combinedthresh1"
[image5]: ./examples/colorthresholding.png "colorthresholding"
[image5a]: ./examples/color_gradient_thresholding.png "color gradient thresholding"
[image6]: ./examples/warp_points.png "Warp src points"
[image7]: ./examples/warp_thresh.png "Warp thresh"
[image8]: ./examples/warp_points_result.png "Warp result"
[image9]: ./examples/histogram.png "Histogram"
[image11]: ./examples/sliding_windows.png "sliding windows"
[image12]: ./examples/lane_outputs.png "Output1"


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

I used 5 gradient thresholding :  
1. s channel from the HLS color scheme.
2. l channel from the HLS color scheme.
3. l channel from LUV color scheme
4. b channel from LAB color scheme    (To detect yellow line)
5. y channel from Ycbcr color scheme   

Among these Lab and Ycbcr gave the best results to detect yellow and white lines respectively.
I then combined these to get best results with the formula logic of :      
    
`      
    combined_binary[(ycbcr_y_binary == 1) | (lab_b_binary == 1) ] = 1
`      
      
I experimented with a lot of thresholding values and finally chose the one written in the code cells under the heading of "Color filters and thresholding".

![alt text][image5]   

#### 2c. Merging of color transforms and gradients to create a thresholded binary image.  


Finally I combined both color filtering and gradient thresholding to get a binary image clearly showing the lane lines.
The following image illustrates the whole process :

![alt text][image5a]   


#### 3. Description of perspective transform.

The code for my perspective transform includes a function called `warp()`, which appears under the cell heading of "Warping - Perpesctive Transform and Birds Eye View" in the file `advance lane keeping.ipynb`. The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python

src_points = np.float32([
                  (260, 682), 
                  (1049,682), 
                  (684,450),
                  (595,450)
                ])

dst_points = np.float32([                 
                  (450,y),
                  (x-450,y),
                  (x-450,0),
                  (450,0)
                        ])

```    
Here is the vizualization -    

![alt text][image6]     
 

I verified that my perspective transform was working as expected by drawing the `dst` points onto a warped image to verify that the lines appear parallel in the warped image.     

![alt text][image8]      

Here the warp results of all test images :     


![alt text][image7]   


#### 4. Identification of lane-line pixels and fit their positions with a polynomial.
  
After calculating the warped image. First I calculate the histogram across the image, this was implemented using numpy.sum() to sum all the pixels on the rows, the result is the following:      

![alt text][image9]   

Then I implemented a function called `sliding_window_polyfit()` which takes the histogram and iterates through the image and find the x coordinates and y coordinates of the pixels that corresponded to the lanes. Margin is set to be 60 and number of windows 10.

I also implemented a `polyfit_using_prev_fit()`. If both left and right lines were detected in last frame we use sliding_window_polyfit(), otherwise we use sliding window.


Once I got the pixels I fit them to a 2nd order polynomial of the form:      
`f(y) =  Ay^2 + By + C`

The result obtained showing the sliding windows and plotted lane lines for all the test images can be seen below:    

![alt text][image11]  


#### 5. Calculation of the radius of curvature of the lane and the position of the vehicle with respect to center.

I defined the function `calculate_rad_curvature_and_position()` in the file `advance lane keeping.ipynb` to calculate the radius of curvature and position of car by calculating distance between center of detected lanes from the center of the image -

I  convert pixel curves to metres curve according to the assumption that lane width is 3.7m.        

Position is calculated according to the formula :

```python
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center_dist = (car_position - lane_center_position) * xm_per_pix
 ```            

 Radius of curvature is calculated according to the formula :    
 
```python
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
  
```
         
 ym_per_pixel and xm_per_pixel are the factors used for converting from pixels to meters. This conversion was also used to generate a new fit with coefficients in terms of meters.
 
 which is calculated as :-      
 
 ```
 xm_per_pixel = 3.7 /375 ( 3.7 actual width of lane, 375 px on warped image)
 ym_per_pixel = 3.0 / 75 (3.0 m actual length of dashed line, 75 px length on warped image) 
 ```
 Radius  more than 8000 is displayed as 'inf'
 
 
#### 6. Example image of result plotted back down onto the road where the lane area is identified clearly.

I implemented this step in the function `draw_lane()` in  `advance lane keeping.ipynb`.  

A class `LaneLines()` is defined which also stores the recently detected lines and averages the values calculated from these lines for the new ones.

`process_image()` defines the pipeline and calls `draw_lane()` which further calls `calculate_rad_curvature_and_position()`.

Here are the results of all test images:


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
