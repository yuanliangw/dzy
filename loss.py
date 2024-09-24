import keras.backend as K

def dice_coefficient(y_true, y_pred, smooth=1e-10):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def dice_coff(label, predict):
    return np.sum(2 * label * predict) / (np.sum(label) + np.sum(predict))