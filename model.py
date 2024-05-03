from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Dropout, Lambda, Add, Multiply


def simAM(inp1, inp2):

    lamda = 1e-4
    d = K.square(inp1 - K.mean(inp1, axis=[1,2], keepdims=True))
    n = (inp1.shape[1] * inp1.shape[2]) - 1
    v = K.sum(d, axis=[1,2], keepdims=True) / n
    td_weights = (d / (4*(v + lamda))) + 0.5
    td_weights = Activation('sigmoid')(td_weights)
    out = Multiply()([inp2, td_weights])

    return out, td_weights

# A Convolutional Block is composed of 2 consecutive (conv + BN + relu) operations, with a BN in between.

def conv_block(inp, num_filters):

    x = Conv2D(num_filters, (3,3), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

# A Encoder Block is combination of conv_block & max pooling operation.

def encoder_block(inp, num_filters):

    x = conv_block(inp, num_filters)
    p = MaxPool2D((2,2))(x)

    return x,p

# Defining Decoder Block
def decoder_block(inp1, inp2, num_filters):

    x = Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")(inp1)
    skip_features, d3_weights = simAM(x, inp2)

    x = Concatenate()([x, skip_features])
    x = Dropout(0.2)(x)

    x = Conv2D(num_filters, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x, d3_weights

def build_UNet(inp1_shape,inp2_shape):

    inps1 = Input(inp1_shape)
    inps2 = Input(inp2_shape)

    s = Lambda(lambda k:k/255)(inps1)
    c1, p1 = encoder_block(s, 32)
    p1 = Dropout(0.2)(p1)
    c2, p2 = encoder_block(p1, 64)
    p2 = Dropout(0.2)(p2)
    c3, p3 = encoder_block(p2, 128)
    p3 = Dropout(0.2)(p3)
    c4, p4 = encoder_block(p3, 256)
    p4 = Dropout(0.2)(p4)

    c5 = conv_block(p4, 512)

    u6, d34_weights = decoder_block(c5, c4, 256)
    u7, d33_weights = decoder_block(u6, c3, 128)
    u8, d32_weights = decoder_block(u7, c2, 64)
    u9, d31_weights = decoder_block(u8, c1, 32)

    output1 = Conv2D(1, (1,1), padding="same", activation="sigmoid")(u9)
    output2 = K.min((K.sum((inps2 * K.square(d31_weights)), axis=[1,2]) / (K.sum(inps2, axis=[1,2]) + K.epsilon())),axis=[1])

    model = Model(inputs=[inps1,inps2], outputs=[output1,output2], name="UNet")

    return model