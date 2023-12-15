import numpy as np
import math
import os
import matplotlib.pyplot as plt
from skimage import io
from skimage import color
from skimage.transform import resize
from skimage.feature import hog
import random

def getJ(angle):
    if angle == 180:
        return 8
    return math.floor(angle / 20)


def getPosition(angle):
    return round(angle / 20, 9)


def custom_hog(img):
    mag = []
    theta = []
    for i in range(128):
        magnitudeArray = []
        angleArray = []
        for j in range(64):
            # checking color change in horizontal dir
            # bdy cases
            if j-1 < 0:
                Gx = img[i][j+1] - 0
            elif j+1 == 64:
                Gx = 0 - img[i][j-1]
            else:
                Gx = img[i][j+1] - img[i][j-1]

            # checking color change in vertical dir
            if i-1<0:
                Gy = img[i+1][j] - 0
            elif i+1 == 128:
                Gy = 0 - img[i-1][j]
            else:
                Gy = img[i+1][j] - img[i-1][j]

            # compute mag
            magnitude = math.sqrt(pow(Gx, 2) + pow(Gy, 2))
            magnitudeArray.append(round(magnitude, 9))

            # compute angle
            # avoid zero division
            if Gx == 0:
                angle = math.degrees(0.0)
            else:
                angle = math.degrees(abs(math.atan2(Gy, Gx)))
                # angle = math.degrees(abs(math.atan2(Gy, Gx)))
                # if angle<0:
                #     angle += 180.0
            angleArray.append(round(angle, 9))
        mag.append(magnitudeArray)
        theta.append(angleArray)
    mag = np.array(mag)
    theta = np.array(theta)
    num_of_bins = 9
    step_size = 180 / num_of_bins
    # a 16*8 matrix of histograms
    histograms = []
    # iterate over all the 8*8 cells
    # 16*8 8by8 cells
    for i in range(0, 128, 8):
        histograms.append([])
        for j in range(0, 64, 8):
            # compute the histogram for this cell
            histogram_vec = [0.0 for i in range(9)]
            for k in range(8):
                for l in range(8):
                    # the global index is (i+k, j+l)
                    # print(f"{i+k}, {j+l}") indexing looks correct
                    bin_num = getJ(theta[i+k][j+l])
                    pos_num = getPosition(theta[i+k][j+l])
                    # print(f"{pos_num}, {bin_num}")
                    histogram_vec[bin_num] += (bin_num + 1 - pos_num) * mag[i+k][j+l]
                    # handle edge case where the angle is between 160 to 180
                    if bin_num == 8:
                        histogram_vec[0] += (pos_num - bin_num) * mag[i+k][j+l]
                    else:
                        histogram_vec[bin_num+1] += (pos_num - bin_num) * mag[i+k][j+l]
            histograms[-1].append(histogram_vec)
    feature_vec = []
    fv = np.zeros((15*3*2, 7*3*2))
    eps = 1e-5
    # 2*2 blocks in 16*8 grid
    for i in range(15):
        for j in range(7):
            block_vec = []
            summ = 0.0
            # each cell in 2*2 block
            for k in range(2):
                for l in range(2):
                    # print(i+k+16*(j+l)) the storage order is row major block structure
                    for m in range(9):
                        block_vec.append(histograms[i+k][j+l][m])
                        summ += pow(histograms[i+k][j+l][m], 2)
            # compute r2 norm
            summ = round(math.sqrt(summ), 9)
            # normalize
            if summ > 0:
                block_vec = [round(x/summ, 9) for x in block_vec]
            for num in block_vec:
                feature_vec.append(num)

            # update fv for visualization
            for k in range(2):
                for l in range(2):
                    for m in range(3):
                        for n in range(3):
                            fv[i*6+k*3+m][j*6+l*3+n] = block_vec[18*k+9*l+3*m+n]
    return np.array(feature_vec)


class SVM:
    def __init__(self) -> None:
        # add one row at the end to be the row of biases
        self.w = np.zeros(3781)
        self.c = 1.0

    def setC(self, c):
        self.c = c

    def objective(self, features, labels, w=np.zeros(2)):
        if len(w) != 3781:
            w = self.w
        term1 = 0.0
        term2 = 0.0
        # add the square norm of w
        for i in range(3781):
            term1 += pow(w[i], 2)
        term1 /= 2.0
        # iterate over all the data rows
        for i in range(features.shape[0]):
            term2 += max(0, 1.0 - labels[i] * np.dot(features[i], w))
        term2 *= self.c
        return term1 + term2

    def gradient(self, features, labels, w=np.zeros(2)):
        if len(w) != 3781:
            w = self.w
        grad = w.copy()
        for i in range(features.shape[0]):
            if labels[i] * np.dot(features[i], w) > 1:
                # correct prediction, no change for w
                continue
            else:
                grad += self.c * (-labels[i]) * features[i]
        return grad

    def fd_validation(self):
        # random w, random x and random d * eps
        eps = 10e-5
        for i in range(5):
            x = np.random.normal(1, 3, (20, 3781))
            y = np.zeros(20)
            for k in range(20):
                y[k] = random.choice([-1, 1])
            w = np.random.normal(0, 2, 3781)
            for j in range(5):
                d = np.random.normal(0, 1, 3781)
                lhs = np.dot(self.gradient(x, y, w), d)
                rhs = (self.objective(x, y, w + eps * d) - self.objective(x, y, w - eps * d)) / (2.0 * eps)
                print((rhs - lhs) / rhs)

    def train(self, features, labels, alpha=0.001, iters=5):
        # gradient descent on w
        for i in range(iters):
            self.w -= alpha * self.gradient(features, labels)

    def predict(self, feature):
        score = np.dot(self.w, feature)
        if score > 0:
            return 1
        else:
            return -1

    def test(self, features, labels):
        total = features.shape[0]
        correct = 0
        for i in range(features.shape[0]):
            score = self.predict(features[i])
            if score == labels[i]:
                correct += 1
        return round(correct / total, 5)


def custom_hog_svm():
    # extract all images, resize them, add them into numpy array
    # also make a label vector
    features = []
    labels = []
    image_names1 = os.listdir("./INRIAPerson/train_64x128_H96/pos/")
    image_names2 = os.listdir("./INRIAPerson/train_64x128_H96/neg/")
    # from positive categories
    for image_name in image_names1:
        features.append(
            resize(color.rgb2gray(io.imread("./INRIAPerson/train_64x128_H96/pos/" + image_name)[:, :, :3]), (128, 64)))
        labels.append(1)
    for image_name in image_names2:
        img = io.imread("./INRIAPerson/train_64x128_H96/neg/" + image_name)
        if img.shape[0] >= 128 and img.shape[1] >= 64:
            for i in range(10):
                x = random.randint(0, img.shape[0] - 128)  # 左上角x坐标
                y = random.randint(0, img.shape[1] - 64)  # 左上角y坐标
                crop_img = img[x:x + 128, y:y + 64, :]
                features.append(color.rgb2gray(crop_img[:, :, :3]))
                labels.append(-1)
    features = np.array(features)
    labels = np.array(labels)

    # split into training and testing set
    inds = np.random.permutation(features.shape[0])
    features = features[inds]
    labels = labels[inds]
    images_train = features[0:11000, :, :]
    labels_train = labels[0:11000]
    images_test = features[11000:, :, :]
    labels_test = labels[11000:]

    # get hog features
    hog_train = []
    hog_test = []
    for i in range(images_train.shape[0]):
        hog_train.append(custom_hog(images_train[i]))
    for i in range(images_test.shape[0]):
        hog_test.append(custom_hog(images_test[i]))
    hog_train = np.array(hog_train)
    hog_test = np.array(hog_test)

    # add a column of ones to the hog features
    ones_train = np.ones((hog_train.shape[0], 1))
    ones_test = np.ones((hog_test.shape[0], 1))
    hog_train = np.append(hog_train, ones_train, axis=1)
    hog_test = np.append(hog_test, ones_test, axis=1)

    model = SVM()
    num_of_iters = 100
    iters_list = [i for i in range(num_of_iters)]
    accs = []
    for i in range(num_of_iters):
        model.train(hog_train, labels_train)
        accs.append(model.test(hog_test, labels_test))

    plt.title("accuracy improvement over training iterations")
    plt.xlabel("iterations")
    plt.ylabel("testing accuracy")
    plt.plot(iters_list, accs)
    plt.show()
    return model
