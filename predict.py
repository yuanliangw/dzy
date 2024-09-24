import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from data_loader import load_image, resize_test_images, resize_predictions
from loss import dice_coff
from model import tensorToimg
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default="./dataset_2", required=False, help='path to dataset')
    parser.add_argument('--model-path', default='./models/model.h5', help='folder of model checkpoints to predict')
    parser.add_argument('--outf', default="./test/test-output", required=False, help='path of predict output')
    args = parser.parse_args()
    os.makedirs(args.outf, exist_ok=True)
    return args

def predict_level3():
    args = get_parser()
    test_img, test_label, test_name_list = load_image(args.data_root, "test", need_name_list=True)
    model = load_model(args.model_path, custom_objects={'dice_coefficient': dice_coefficient,
                                                        'dice_coefficient_loss': dice_coefficient_loss})

    test_img_resized = resize_test_images(test_img, target_size=(288, 384))
    result_resized = model.predict(test_img_resized)
    result = resize_predictions(result_resized, original_size=(500, 574))

    dc = dice_coff(test_label, result)
    print("the dice coefficient is:", dc)

    for i in range(result.shape[0]):
        final_img = tensorToimg(result[i])
        ori_img = test_img[i]
        ori_gt = tensorToimg(test_label[i])

        plt.figure(figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.imshow(ori_img, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(ori_gt, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(final_img, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(args.outf, test_name_list[i].split('.')[0] + '.png'))
        plt.close()

if __name__ == "__main__":
    predict_level3()
