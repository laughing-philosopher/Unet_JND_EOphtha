from tensorflow.keras import backend as K

# Function for Sensitivity & Specificity

def ce_loss(y_true, y_pred):
    term_0 = (1 - y_true) * K.log(1 - y_pred + K.epsilon())
    term_1 = y_true * K.log(y_pred + K.epsilon())
    out = -K.mean(term_0 + term_1)                               # This is Binary Cross Entropy Loss
    return out

def acc_met(y_true,y_pred):
    out = K.mean(K.equal(y_true, y_pred))
    return out

def sensitivity(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    neg_y_pred = 1 - y_pred_f
    tp = K.sum(y_true_f * y_pred_f)
    fn = K.sum(y_true_f * neg_y_pred)
    return tp / (tp+fn+K.epsilon())

def specificity(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    neg_y_true = 1 - y_true_f
    neg_y_pred = 1 - y_pred_f
    fp = K.sum(neg_y_true * y_pred_f)
    tn = K.sum(neg_y_true * neg_y_pred)
    return tn / (tn + fp + K.epsilon())