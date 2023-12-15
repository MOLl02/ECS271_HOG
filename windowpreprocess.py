import os
import numpy as np
import random
import cv2
from skimage.io import imread, imsave, imshow
from skimage.feature import hog
from skimage.color import rgb2gray
import math
from skimage.transform import resize


def extract_data_from_file(filename):
    with open(filename, 'r', encoding='latin-1') as file:
        lines = file.readlines()

    image_path = None
    image_size = None
    bounding_boxes = []

    for line in lines:
        if 'Image filename' in line:
            image_path = line.split(':')[1].strip().replace('"', '').strip()

        if 'Image size' in line:
            parts = line.split(':')[1].strip().split('x')
            image_size = [int(parts[0].strip()), int(parts[1].strip())]  # Width x Height

        if 'Bounding box for object' in line:
            parts = line.split(':')[1].strip()
            coords = parts.split('-')
            xmin_ymin = coords[0].strip()[1:-1].split(',')
            xmax_ymax = coords[1].strip()[1:-1].split(',')

            # Convert string coordinates to integers
            xmin, ymin = map(int, xmin_ymin)
            xmax, ymax = map(int, xmax_ymax)

            bounding_boxes.append([xmin, ymin, xmax, ymax])

    return image_path, image_size, bounding_boxes


def process_folder(folder_path):
    data = []

    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            image_path, image_size, bounding_boxes = extract_data_from_file(file_path)
            data.append({
                'image_path': image_path,
                'image_size': image_size,
                'bounding_boxes': bounding_boxes
            })

    return data


def calculate_overlap_percentage(box1, box2):
    """
    Calculate the overlapping percentage between two bounding boxes.
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    overlap_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])

    return overlap_area / box1_area


def process_single_entry(entry):
    """
    Extract 64 windows from the image where one is the bounding box and the others are shifted versions of it.
    Calculate the overlapping percentage with the bounding box for each window.
    """

    image_path = './INRIAPerson/' + entry['image_path']
    image_size = entry['image_size']
    bounding_box = entry['bounding_boxes'][0]

    print('Start: ' + image_path)

    windows = []
    overlaps = []
    hog_feature = []

    bbox_width = bounding_box[2] - bounding_box[0]
    bbox_height = bounding_box[3] - bounding_box[1]

    horizontal_step = (image_size[0] - bbox_width) / 7
    vertical_step = (image_size[1] - bbox_height) / 7

    for i in range(8):
        for j in range(8):
            window = [
                math.floor(i * horizontal_step),
                math.floor(j * vertical_step),
                math.floor(i * horizontal_step + bbox_width),
                math.floor(j * vertical_step + bbox_height)
            ]
            windows.append(window)
            overlap = calculate_overlap_percentage(window, bounding_box)
            overlaps.append(overlap)

    for window in windows:
        img = imread(image_path)[:, :, :3]  # Remove Alpha channel
        # Extracting coordinates from the window
        x_min, y_min, x_max, y_max = window

        # Crop the image using the bounding box coordinates
        crop_img = img[y_min:y_max, x_min:x_max, :]
        fd = hog(resize(rgb2gray(crop_img), (128, 64)), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        hog_feature.append(fd)

    return np.array(overlaps), np.array(hog_feature)


def process_all_entries(data):
    all_windows = []
    all_overlaps = []
    all_hog_features = []

    for entry in data:
        overlaps, hog_feature = process_single_entry(entry)

        # Concatenate the results
        all_overlaps.extend(overlaps)
        all_hog_features.extend(hog_feature)

    return np.array(all_overlaps), np.array(all_hog_features)


def extract_windowdata_from_file(filename):
    with open(filename, 'r', encoding = 'latin-1') as file:
        lines = file.readlines()

    image_size = None
    bounding_boxes = []
    for line in lines:
        if 'Image size' in line:
            parts = line.split(':')[-1].strip().split('x')
            image_size = [int(parts[0].strip()), int(parts[1].strip())]  # Width x Height

        if 'Bounding box for object' in line:
            parts = line.split(':')[-1].strip()
            coords = parts.split('-')
            xmin_ymin = coords[0].strip()[1:-1].split(',')
            xmax_ymax = coords[1].strip()[1:-1].split(',')

            # Convert string coordinates to integers
            xmin, ymin = map(int, xmin_ymin)
            xmax, ymax = map(int, xmax_ymax)

            # Calculate width and height
            width = xmax - xmin
            height = ymax - ymin
            bounding_boxes.append([width, height])

    return image_size, bounding_boxes


def process_winodow_data_folder(folder_path):
    # Extract the bounding box size and image size only
    x = []  # Image sizes
    y = []  # Bounding box sizes

    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            image_size, boxes = extract_windowdata_from_file(file_path)
            for box in boxes:
                x.append(image_size)
                y.append(box)

    return np.array(x), np.array(y)


def extract_window(input_path, output_path):
    x, y = process_winodow_data_folder(input_path)

    # Combine X and Y into a dictionary
    data = {'X': x, 'Y': y}

    # Save the combined data in one file
    np.save(output_path, data)



