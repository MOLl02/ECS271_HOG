import imutils
import time
import cv2
from skimage.io import imread, imsave, imshow
from skimage.feature import hog
from skimage.color import rgb2gray
import time
from joblib import dump, load


def pyramid(image, scale=1.5, minSize=(64, 128)):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def act(image_path):
    # load the image and define the window width and height
    image = cv2.imread(image_path)
    (winW, winH) = (64, 128)

    # loop over the image pyramid
    for resized in pyramid(image, scale=1.5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW
            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            time.sleep(0.025)


def detect(image_path):
    image = imread(image_path)[:, :, :3]
    (winW, winH) = (64, 128)
    clf = load('svm_classifier.joblib')  # Load trained model here

    for resized in pyramid(image, scale=1.5):
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # Visualize the current sliding window
            clone = resized.copy()
            clone = cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Sliding Window", clone)
            cv2.waitKey(1)
            time.sleep(0.025)  # Delay for visibility

            # Convert the window to grayscale and extract HOG features
            gray_window = rgb2gray(window)
            hog_features = hog(gray_window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

            # Convert HOG features to a PyTorch tensor and pass through the network
            #hog_tensor = torch.tensor(hog_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            output = clf.predict([hog_features])  # Predict

            # YOUR_CRITERION_HERE - Define how you determine if the window contains the object
            if int(output[0]) == 1:
                cv2.imshow("Detected Object", clone)
                cv2.waitKey(1)
                time.sleep(0.025)  # Delay for visibility
                print("Detect!")
    cv2.destroyAllWindows()


def slide_window(image, window_size=(64, 128), step_size=32, output_folder='./slide/'):
    """
    This function slides a window over an image, ensures that the window does not exceed the image boundaries,
    and saves each window as a separate image along with a copy of the original image that shows the specific window
    with a rectangle. It returns two kinds of images:
    the original image with drawn rectangles, and the saved window images each with its corresponding original image.

    :param image: The image over which the window will slide.
    :param window_size: The size of the sliding window (width, height).
    :param step_size: The step size for sliding the window.
    :param output_folder: The folder where the window images and their corresponding originals will be saved.
    :return: The original image with rectangles and paths to the first few saved window images and their originals for example.
    """
    import os

    saved_window_paths = []
    saved_original_paths = []
    window_count = 0

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            # Draw a rectangle on the copy of the original image
            img_with_current_rectangle = image.copy()
            cv2.rectangle(img_with_current_rectangle, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)

            # Extract the window
            window = image[y:y + window_size[1], x:x + window_size[0]]

            # Save the window image and its corresponding original with rectangle
            window_path = os.path.join(output_folder, f'window_{window_count}.png')
            original_with_rectangle_path = os.path.join(output_folder, f'original_with_window_{window_count}.png')
            cv2.imwrite(window_path, cv2.cvtColor(window, cv2.COLOR_RGB2BGR))
            cv2.imwrite(original_with_rectangle_path, cv2.cvtColor(img_with_current_rectangle, cv2.COLOR_RGB2BGR))
            saved_window_paths.append(window_path)
            saved_original_paths.append(original_with_rectangle_path)
            window_count += 1

    return saved_window_paths[:5], saved_original_paths[:5]
