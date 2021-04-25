# Self Driving Car Simulator

## Proposition


**Vehicle Lane Detection** 

In this project I use deep learning to detect vehicle lanes from road images. The model uses a fully convolutional neural network
which outputs an image of a predicted lane.

## Dataset

- The data consists of a set of 12,764 raw training images along with marked lanes lanes used as labels for each image
- The labeled images were generated from polynomial coefficents marking the lane for each image
- Each image is downsized to a uniform format of (80x160x3)


## Running the code
**All code is stored in base directory**
- To train the model, run
    ```
    python main.py
    ```
- To generate image predictiction, run
    ```
    python predict_lines.py
    ```

## Training

The best trained model uses a fully convoluational neural network
The model takes in the road image and encodes it up to a filter size of 1024 before decoding it back down to
the final layer with a filter size of 1. The encoder decoder architecture 

<img src = "figures/encoder_decoder.png">


<img src = "figures/model_summary.png" width = 300>


## Results

### Predicted Lane Segments on previously seen data
<img src = "figures/Combined_CNN2.png" width = 200>
<img src = "figures/Combined_CNN3.png" width = 300>
<img src = "figures/Combined_image_CNN3.png" width = 300>

The model showed posed relatively accurate lane segments when predicting on the previously trained data. This is clear as the model has already been
fitted to this data with raw images and polynomial labels fitted to the data.



### Predicted Lane Segments on unseen data
<img src = "figures/combined_unseen1.png" width = 300>
<img src = "figures/combined_unseen4.png" width = 300>
<img src = "figures/combined_unseen8.png" width = 300>

When applying the model to unseen images, entire roadways tend to be marked rather than specific lanes.
This is likely due to the fact that lane markings can be difficult to identify as the difference in pixel color is not significant compared
to the pixel change between roadway and surrounding area.




## Moving Forward

## Technologies Used
* [Matplotlib](https://matplotlib.org)
* [Pandas](https://pandas.pydata.org)
* [Tensorflow Keras](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
* [OpenCV](https://pypi.org/project/opencv-python/)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)
* [NumPy](https://numpy.org)


### Citations:
