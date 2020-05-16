import math

import tensorflow.keras
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Convolution2D, PReLU, Activation, ReLU, Add, Multiply, Lambda, ZeroPadding2D, Cropping2D, Dropout, Input,  InputSpec
from tensorflow.keras.models import Model
from tensorflow.keras.activations import tanh, sigmoid
from tensorflow.keras.optimizers import Nadam

import matplotlib.pyplot as plt


print(tensorflow.keras.__version__)


class MaskedConvolution2D(Convolution2D):
    def __init__(self, *args, mask='B', mask_direction='horizontal', n_channels=3, mono=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_type = mask
        self.mask_direction = mask_direction
        self.mask = None

    def build_mask(self, kernel_shape):
        # Create a numpy array of ones in the shape of our convolution weights.
        self.mask = np.ones(kernel_shape)

        # We assert the height and width of our convolution to be equal as they should.
        # assert mask.shape[0] == mask.shape[1]

        # Since the height and width are equal, we can use either to represent the size of our convolution.
        filter_size = self.mask.shape[0]
        filter_center = filter_size / 2

        # Zero out all weights below the center.
        self.mask[math.ceil(filter_center):] = 0

        # Zero out all weights to the right of the center.
        self.mask[math.floor(filter_center):, math.ceil(filter_center):] = 0

        # If the mask type is 'A', zero out the center weigths too.
        if self.mask_type == 'A':
            self.mask[math.floor(filter_center), math.floor(filter_center)] = 0

        if self.mask_direction == 'vertical':
            self.mask[math.floor(filter_center)] = 0

        # Convert the numpy mask into a tensor mask.
        return K.constant(self.mask, dtype='float32', shape=kernel_shape)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel_mask = self.build_mask(kernel_shape)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, x, mask=None):
        ''' I just copied the tensorflow.keras Convolution2D call function so don't worry about all this code.
            The only important piece is: self.W * self.mask.
            Which multiplies the mask with the weights before calculating convolutions. '''
        output = K.conv2d(
            x,
            self.kernel * self.kernel_mask,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)
        if self.use_bias:
            output = K.bias_add(
                output,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(output)
        return output

    def get_config(self):
        # Add the mask type property to the config.
        return dict(list(super().get_config().items()) + list({'mask': self.mask_type}.items()))


class VerticalConvolution2D(object):
    def __init__(self, filters, kernel_size, mask_type='B'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.mask_type = mask_type

    def __call__(self, x):
        v_kernel_size = self.kernel_size // 2 + 1
        if self.mask_type == 'A':
            v_kernel_size -= 1
        output = ZeroPadding2D(((v_kernel_size, 0), (self.kernel_size // 2, self.kernel_size // 2)))(x)
        output = Convolution2D(self.filters, (v_kernel_size, self.kernel_size))(output)
        output = Cropping2D(((0, 1), (0, 0)))(output)
        return output


class HorizontalConvolution2D(object):
    def __init__(self, filters, kernel_size, mask_type='B'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.mask_type = mask_type

    def __call__(self, x):
        h_kernel_size = self.kernel_size // 2+1
        if self.mask_type == 'A':
            h_kernel_size -= 1
        output = ZeroPadding2D(((0, 0), (h_kernel_size, 0)))(x)
        output = Convolution2D(self.filters, (1, h_kernel_size))(output)
        output = Cropping2D(((0, 0), (0, 1)))(output)
        return output


class GatedActivation(object):

    def __call__(self, model):
        filters = model.shape[-1]

        sigmoid_layer = Lambda(lambda _inputs: _inputs[:, :, :, :filters // 2])(model)
        tanh_layer = Lambda(lambda _inputs: _inputs[:, :, :, filters // 2:])(model)

        sigmoid_layer = Activation('sigmoid')(sigmoid_layer)
        tanh_layer = Activation('tanh')(tanh_layer)

        return Multiply()([sigmoid_layer, tanh_layer])


class DoubleStackBlock(object):
    def __init__(self, filters, kernel_size, mask_type='B',return_vertical=True):
        self.filters = filters
        self.kernel_size = kernel_size
        self.mask_type = mask_type
        self.return_vertical = return_vertical

    def __call__(self, vertical_model, horizontal_model):
        vertical_stack = vertical_model
        horizontal_stack = horizontal_model
        # vertical_stack = MaskedConvolution2D(2*self.filters, (3, 3), mask_direction='vertical', padding='same')(vertical_stack)

        # horizontal_stack = MaskedConvolution2D(2*self.filters, (1, 3), padding='same')(horizontal_stack)

        vertical_stack = VerticalConvolution2D(2 * self.filters, kernel_size=self.kernel_size)(vertical_stack)
        horizontal_stack = HorizontalConvolution2D(2 * self.filters, kernel_size=self.kernel_size)(horizontal_stack)

        stack_crossing = Convolution2D(2 * self.filters, (1, 1))(vertical_stack)

        horizontal_stack = Add()([horizontal_stack, stack_crossing])
        if self.return_vertical:
            vertical_stack = GatedActivation()(vertical_stack)
        horizontal_stack = GatedActivation()(horizontal_stack)
        horizontal_stack = Convolution2D(self.filters, (1, 1))(horizontal_stack)

        horizontal_stack = Add()([horizontal_stack, horizontal_model])
        if not self.return_vertical:
            return horizontal_stack
        return vertical_stack, horizontal_stack


class ResidualBlockList(object):
    def __init__(self, filters, depth):
        self.filters = filters
        self.depth = depth

    def __call__(self, x):
        for i in range(self.depth):
            if i == 0:
                x = DoubleStackBlock(self.filters, kernel_size=3)(x, x)
            elif i == self.depth - 1:
                x = DoubleStackBlock(self.filters, kernel_size=3, return_vertical=False)(x[0], x[1])
            else:
                x = DoubleStackBlock(self.filters, kernel_size=3)(x[0], x[1])

        return x


def create_model():
    shape = (48, 96, 1)
    filters = 64
    depth = 10

    input_img = Input(shape)

    model = MaskedConvolution2D(filters=filters, kernel_size=(7, 7), padding='same', mask='A')(input_img)

    # model = InitialBlock(filters)(input_img)

    model = ResidualBlockList(filters, depth)(model)

    for _ in range(2):
        model = Convolution2D(filters, (1, 1), padding='valid')(model)
        model = ReLU()(model)

    outs = Convolution2D(1, (1, 1), padding='valid')(model)
    outs = Activation('sigmoid')(outs)

    model = Model(input_img, outs)
    model.compile(optimizer=Nadam(), loss='binary_crossentropy')
    model.summary()

    return model
