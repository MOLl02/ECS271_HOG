import random
import os
import cv2
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave, imshow
from skimage.feature import hog
from skimage.color import rgb2gray
import time
from joblib import dump, load
import datapreprocess
from hog_network import train, test, grid_search
from window_network import train as train_window, test as test_window, train_overlapping, test_overlapping, plot_metrics
from torch.utils.data import DataLoader
import torch
from skimage.transform import resize
from windowpreprocess import process_folder, process_all_entries, process_single_entry, extract_window
from pyramid_img import act, detect
from custom_hog import custom_hog_svm

if __name__ == '__main__':
    """Extract HOG"""
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

    """Train NN for HOG"""
    # train_data = np.load("./feature/train.npy", allow_pickle=True)
    # test_data = np.load("./feature/test.npy", allow_pickle=True)
    # # Assuming 'data' is your dataset variable
    # train_dataset = datapreprocess.CustomDataset(train_data)
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_dataset = datapreprocess.CustomDataset(test_data)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    # trained_net, accuracys= train(train_loader, 20, 0.001)
    # test(test_loader, trained_net)
    # plot_metrics(accuracys)
    # torch.save(trained_net, 'hog_nnmodel.pth')

    """Grid search"""
    # Define the range of epochs and learning rates to try
    # epochs_list = list(range(5, 16))  # This will create a list from 5 to 20
    # lr_list = [0.001, 0.01, 0.1, 0.5, 1]  # Example values for learning rates
    # best_accuracy, best_params = grid_search(train_loader, test_loader, epochs_list, lr_list)
    # print(best_accuracy, best_params)

    """Extract window size"""
    # print('Start extracting window')
    # input_path = './INRIAPerson/Train/annotations/'
    # output_path = './feature/window_test.npy'
    # extract_window(input_path, output_path)
    # train_window = np.load("./feature/window_train.npy", allow_pickle=True)

    """ Train window size"""
    # train_data = np.load("./feature/window_train.npy", allow_pickle=True)
    # test_data = np.load("./feature/window_test.npy", allow_pickle=True)
    #
    # train_dataset = datapreprocess.CustomDataset(train_data)
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # test_dataset = datapreprocess.CustomDataset(test_data)
    # test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    #
    # trained_net = train_window(train_loader, 100, 0.001)
    # test_window(test_loader, trained_net)

    """extract window and overlapping"""
    # train_overlap_data = process_folder('./INRIAPerson/Train/annotations/')
    # train_all_overlaps, train_all_hog_features = process_all_entries(train_overlap_data)
    # np.savez('./feature/train_data.npz', features=train_all_hog_features, labels=train_all_overlaps)
    # test_overlap_data = process_folder('./INRIAPerson/Test/annotations/')
    # test_all_overlaps, test_all_hog_features = process_all_entries(test_overlap_data)
    # np.savez('./feature/test_data.npz', features=test_all_hog_features, labels=test_all_overlaps)

    """Pyramid Image Iteration"""
    # act('./INRIAPerson/Test/pos/person_012.png')

    """Train network for overlapping"""
    # traindata_overlapping = datapreprocess.OverlappingDataset('./feature/train_data.npz')
    # testdata_overlapping = datapreprocess.OverlappingDataset('./feature/test_data.npz')
    # train_loader = DataLoader(traindata_overlapping, batch_size=64, shuffle=True)
    # test_loader = DataLoader(testdata_overlapping, batch_size=64, shuffle=True)
    # trained_net, epoch_losses = train_overlapping(train_loader, 20, 0.01)
    # test_overlapping(test_loader, trained_net)
    # plot_metrics(epoch_losses)
    # Test: 0.05139

    """Apply slide window to detect """
    # Train a simple SVM
    # train_data = np.load("./feature/train.npy", allow_pickle=True)
    # train_X = train_data.item()['X']
    # train_Y = train_data.item()['Y']
    # test_data = np.load("./feature/test.npy", allow_pickle=True)
    # test_X = test_data.item()['X']
    # test_Y = test_data.item()['Y']
    # clf = svm.SVC()
    # clf.fit(train_X, train_Y)
    # dump(clf, 'svm_classifier.joblib')

    detect('./INRIAPerson/Test/pos/crop001638.png')

    """Custom HOG and SVM"""
    # model = custom_hog_svm()



