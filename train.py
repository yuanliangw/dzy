# train.py
import argparse
import os
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from data_loader import load_image
from model import VGG16_unet_model
from loss import dice_coefficient_loss, dice_coefficient
from plotting import plot_history  # 假设后面会有绘图功能


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default="./dataset_2", required=False, help='path to dataset')
    parser.add_argument('--img_enhanced', default=False, help='image enhancement')
    parser.add_argument('--batch-size', type=int, default=8, help='input batch size')
    parser.add_argument('--image-size', default=(288, 384, 3),
                        help='the (height, width, channel) of the input image to network')
    parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--model-save', default='./models/level3_model_another.h5',
                        help='folder to output model checkpoints')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.model_save), exist_ok=True)
    return args


def train_level3():
    args = get_parser()
    train, train_label = load_image(args.data_root, "train", need_enhanced=args.img_enhanced)
    val, val_label = load_image(args.data_root, 'val', need_enhanced=args.img_enhanced)

    model = VGG16_unet_model(input_size=args.image_size, if_transfer=True, if_local=True)
    model.compile(optimizer=Adam(learning_rate=args.lr), loss=dice_coefficient_loss, metrics=[dice_coefficient])

    model_checkpoint = ModelCheckpoint(args.model_save, monitor='loss', verbose=1, save_best_only=True)
    history = model.fit(train, train_label, batch_size=args.batch_size, epochs=args.niter, callbacks=[model_checkpoint],
                        validation_data=(val, val_label))

    plot_history(history, args.outf)


if __name__ == "__main__":
    train_level3()
