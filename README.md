# Self Driving Car Simulator

## Proposition


**Vehicle Lane Detection** 

In this project I use deep learning to detect vehicle lanes from road images. The model uses a fully convolutional neural network
which outputs an image of a predicted lane.

## Dataset

- The data consists of a set of 12,000 raw training images along with marked lanes lanes used as labels for each image
- The labeled images were generated from polynomial coefficents marking the lane for each image
- Each image is downsized to a uniform format of (80x160x3)
- 




## EDA


### Image Processing



## Running the code
**All code is stored in base directory**
- To train the model, simply run
    ```
    python model.py
    ```

## Training

<img src = "figures/model_summary.png" width = 300>





## Results

### Predicted Lane Segments on previously seen data
<img src = "figures/Combined_CNN2.png" width = 300>
<img src = "figures/Combined_CNN3.png" width = 300>
<img src = "figures/Combined_image_CNN3.png" width = 300>



### Predicted Lane Segments on unseen data
<img src = "figures/combined_unseen1.png" width = 300>
<img src = "figures/combined_unseen4.png" width = 300>
<img src = "figures/combined_unseen8.png" width = 300>




## Moving Forward

## Technologies Used
* [Matplotlib](https://matplotlib.org)
* [Pandas](https://pandas.pydata.org)
* [Tensorflow Keras](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
* [OpenCV](https://pypi.org/project/opencv-python/)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)
* [NumPy](https://numpy.org)
* [Autonomous Vehicle Simulator](https://github.com/udacity/self-driving-car-sim)


### Citations:
