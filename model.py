import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Dropout, Lambda, Add, Multiply


class SimAM(Layer):
    def __init__(self, lamda=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.lamda = lamda

    def call(self, inputs):
        # inputs is a list/tuple: (inp1, inp2)
        inp1, inp2 = inputs

        # ensure numeric dtype (but do it inside the Layer so it's valid for KerasTensor)
        inp1 = tf.cast(inp1, tf.float32)
        inp2 = tf.cast(inp2, tf.float32)

        # local mean and squared difference
        mean = tf.reduce_mean(inp1, axis=[1, 2], keepdims=True)
        d = tf.square(inp1 - mean)

        # dynamic shape for n to be compatible at graph-build time
        shape = tf.shape(inp1)
        h = tf.cast(shape[1], tf.float32)
        w = tf.cast(shape[2], tf.float32)
        n = (h * w) - 1.0

        v = tf.reduce_sum(d, axis=[1, 2], keepdims=True) / (n + K.epsilon())

        td_weights = (d / (4.0 * (v + self.lamda))) + 0.5
        td_weights = tf.sigmoid(td_weights)

        out = inp2 * td_weights
        return [out, td_weights]


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

    # Use the SimAM layer
    skip_and_weights = SimAM()( [x, inp2] )
    skip_features = skip_and_weights[0]
    d3_weights = skip_and_weights[1]

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