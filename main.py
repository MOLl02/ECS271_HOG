import random
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datapreprocess
from network import train, test
from torch.utils.data import DataLoader


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


if __name__ == '__main__':
    #solve('train')
    #solve('test')
    #result = np.load("./feature/train.npy", allow_pickle=True)
    #print(result)

    # for i, data in enumerate(result):
    #     # get the inputs
    #     inputs, labels = data
    #     print(inputs,labels)
    #     if i == 1:
    #         break

    # Apply the function and get the list of window images
    #random_image = imread('./INRIAPerson/Test/pos/crop001520.png')
    #slide_window(random_image)
    #print(random_image.shape)

    print('Start')

    train_data = np.load("./feature/train.npy", allow_pickle=True)
    test_data = np.load("./feature/test.npy", allow_pickle=True)
    # Assuming 'data' is your dataset variable
    train_dataset = datapreprocess.CustomDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = datapreprocess.CustomDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    trained_net = train(train_loader)
    test(test_loader, trained_net)

