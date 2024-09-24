import os
import cv2
import numpy as np

def image_process_enhanced(img):
    return cv2.equalizeHist(img)

def label_to_code(label_img):
    row, column, channels = label_img.shape
    for i in range(row):
        for j in range(column):
            if label_img[i, j, 0] >= 0.75:
                label_img[i, j, :] = [1, 0, 0]
            elif (label_img[i, j, 0] < 0.75) & (label_img[i, j, 0] >= 0.5):
                label_img[i, j, :] = [0, 1, 0]
            elif (label_img[i, j, 0] < 0.5) & (label_img[i, j, 0] >= 0.25):
                label_img[i, j, :] = [0, 0, 1]
    return label_img

def load_image(root, data_type, size=None, need_name_list=False, need_enhanced=False):
    image_path = os.path.join(root, data_type, "image")
    label_path = os.path.join(root, data_type, "label")

    image_list = []
    label_list = []
    image_name_list = []

    for file in os.listdir(image_path):
        image_file = os.path.join(image_path, file)
        label_file_name = file.split(".")[0] + ".png"
        label_file = os.path.join(label_path, label_file_name)

        if need_name_list:
            image_name_list.append(file)

        img = cv2.imread(image_file)
        label = cv2.imread(label_file)

        if size is not None:
            img = cv2.resize(img, (size[1], size[0]))
            label = cv2.resize(label, (size[1], size[0]))

        if need_enhanced:
            img = image_process_enhanced(img)

        img = img / 255.0
        label = label / 255.0
        label = label_to_code(label)

        image_list.append(img)
        label_list.append(label)

    if need_name_list:
        return np.array(image_list), np.array(label_list), image_name_list
    else:
        return np.array(image_list), np.array(label_list)