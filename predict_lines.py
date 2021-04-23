import numpy as np
import cv2
# from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import load_model
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import glob

class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def make_lines(image, model, lanes):
    """ Generates predicted lane segment based on trained model model
    Takes original image, passes it into model to predict lanes, then overlays
    original image predicted image
    returns combined image"""

    # Get image ready for feeding into model
    #print(image.shape)
    # small_img = cv2.resize(image, (80, 160), 3)
    small_img = image # Test
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    prediction = model.predict(small_img)[0] * 255
    print(prediction)

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
    lane_image = cv2.resize(lane_drawn, (160, 80), 3)
    # lane_image = cv2.resize(lane_drawn, (720, 1280), 3)

    # Merge the lane drawing onto the original image

    #print(image.shape)
    #print(lane_image.shape)
    #print(type(image))
    #print(type(lane_image))

    #img1[image[:, :, 1:].all(axis=-1)] = 0
    #img2[lane_image[:, :, 1:].all(axis=-1)] = 0

    # dst = cv2.addWeighted(img1, 1, img2, 1, 0)

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(lane_image)
    axs[1].imshow(image)
    # axs[2].imshow(dst)
    plt.show()

    # combined = (image*1 + lane_image*1).astype(int)
    #print((image.shape))
    #print((lane_image.shape))
    #result = cv2.addWeighted(image, 0.7, lane_image, 0.3, 0)
    #result = image + lane_image
    #print(combined)
    #plt.imshow(combined)
    #plt.show()

    return lane_image # SHould be combined

def make_video():
    """ Takes images from image directory and generates 
    video of overlayed images """
    img_array = []
    for filename in glob.glob('unseen_images/*.png'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter('unseen_images.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def plotter(result):
    """ Plots image overlayed image"""
    plt.imshow(result)
    plt.show()

def load_images(path):
    """ Loads image from filepath into image_list array"""

    image_list = []
    i = 0
    for filename in glob.glob(path):
        im = Image.open(filename)
        image_list.append(im)
        if i > 50: # Debugging error with having too many files open simulataneously
            break
        i += 1
    return image_list

def main():
    #model = load_model('full_CNN_model.h5')
    #model = load_model('baseline_cnn.h5')
    model = load_model('model3.h5')

    lanes = Lanes()

    pkl = open('full_CNN_train.p', 'rb')
    imgs = pickle.load(pkl)

    path = "/Users/adam/Desktop/galvanize/\
    capstones/semantic_lane_detect/data/data_road/testing/image_2/*.png"
    #img = cv2.imread(path)
    # load_image(path)
    #img = cv2.resize(img, (160,80), 3)

    #result = make_lines(img, model, lanes)

    #plotter(result)

    #road_lines = []
    #print(model.evaluate())

    #imgs = load_images(path)
    #make_lines(imgs[30], model, lanes)
    make_lines(imgs[3], model, lanes)

    # for i, img in enumerate(imgs[0:70]):
    #     combined = make_lines(img, model, lanes)
    #     print(type(combined))

    #     print((combined.shape))
    #     im = Image.fromarray((combined).astype(np.uint8))
    #     im.save(f"unseen_images/img{i}.png")

    make_video()

    #print(result)
    #plt.imshow(result)
    #plt.show()


if __name__=="__main__":

    main()
