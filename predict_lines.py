import numpy as np
import cv2
# from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import load_model
import pickle

class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def make_lines(image, model, lanes):

    # Get image ready for feeding into model
    small_img = cv2.resize(image, (80, 160), interpolation = cv2.INTER_AREA)
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    prediction = model.predict(small_img)[0] * 255

    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]
    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = cv2.resize(lane_drawn, (720, 1280), interpolation = cv2.INTER_AREA)

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result



if __name__=="__main__":

    model = load_model('full_CNN_model.h5')

    lanes = Lanes()

    pkl = open('full_CNN_train.p', 'rb')
    imgs = pickle.load(pkl)

    result = make_lines(imgs[0], model, lanes)
    print(result)

