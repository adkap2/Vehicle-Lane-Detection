import numpy as np
import cv2
from keras.models import load_model
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import glob

class Lanes():
    """ Stores lanes values over multiple images
    """
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

def make_lines(image, model, lanes):
    """ Generates predicted lane segment based on trained model model
    Takes original image, passes it into model to predict lanes, then overlays
    original image predicted image
    returns combined image"""

    # Get image ready for feeding into model
    small_img = image
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # OpenCV takes images in 0-255 format while NeuralNets prefer 0-1 format
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
    lane_image = cv2.resize(lane_drawn, (160, 80), 3)

    # Plot two images side by side
    # plot_side_by_side(image, lane_image)

    # Merge the lane drawing onto the original image
    combined = (image*1 + lane_image*1).astype(int)
    return combined # Should be combined


def make_video():
    """ Takes images from image directory and generates
    video of overlayed images """
    img_array = []
    files = sorted(glob.glob('continous_driving/*.png'))
    for filename in files:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter('continous_driving.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def plot_combined(result):
    """ Plots image overlayed image"""

    plt.imshow(result)
    plt.show()


def plot_side_by_side(image, lane_image):
    """ Plots predicted and original image in side by side
    format"""
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(lane_image)
    axs[1].imshow(image)
    plt.show()


def load_images(path):
    """ Loads image from filepath into image_list array"""

    image_list = []
    i = 0
    files = sorted(glob.glob(path))
    for filename in files:
        if len(filename) > 1:
            print(filename)
            im = Image.open(filename)
            image_list.append(im)
            if i > 200: # Debugging error with having too many files open simulataneously
                break
            i += 1
    return image_list


def predict_unseen(path, model, lanes):
    """Generates predictions from unseen image data
    Currently has poor performance
    """
    imgs = load_images(path)

    for i, img in enumerate(imgs[0:200]):
        img = np.array(img)
        img = cv2.resize(img, (160,80))
        combined = make_lines(img, model, lanes)
        # plot_combined(combined)

        im = Image.fromarray((combined).astype(np.uint8))
        im.save(f"continous_driving/img{i}.png")


def make_pred_seen(model, lanes):
    """ makes prediction on already seen images """
    #img = cv2.imread(path)
    # load_image(path)

    pkl = open('full_CNN_train.p', 'rb')
    imgs = pickle.load(pkl)
    combined0 = make_lines(imgs[400], model, lanes)
    combined1 = make_lines(imgs[12], model, lanes)
    plot_combined(combined0)
    plot_combined(combined1)
    im = np.array(imgs[8000])
    pkl = open('full_CNN_labels.p', 'rb')
    labels = pickle.load(pkl)
    lab = np.array(labels[8000])
    cv2.imwrite('example_data_image2.png', im)
    cv2.imwrite('example_data_label2.png', lab)


def get_model():
    """ Returns selected CNN model to generate predictions on"""

    model = load_model('full_CNN_model.h5')
    # model = load_model('baseline_cnn.h5')
    # model = load_model('model3.h5')

    return model


def main():
    """ Main functions which calls all other functions in program file
    """
    model = get_model()
    print(model.summary())
    lanes = Lanes()
    make_pred_seen(model, lanes)
    # path = "/Users/adam/Desktop/galvanize/capstones/semantic_lane_detect/data/data_road/testing/image_2/*.png"
    path = "/Users/adam/Desktop/galvanize/capstones/semantic_lane_detect/driving_dataset/partc/*.jpg"

    predict_unseen(path, model, lanes)
    # make_video()


if __name__=="__main__":
    main()
