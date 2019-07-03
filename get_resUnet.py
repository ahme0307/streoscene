
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,BatchNormalization,Activation,UpSampling2D,Flatten, Dense,AveragePooling2D,add,AveragePooling2D,add,Dropout,ZeroPadding2D,Convolution2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.optimizers import RMSprop
from keras.losses import binary_crossentropy
from params import *
from keras.models import Sequential
import cv2
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from keras.utils import get_file
from keras import layers
from keras import utils
from keras import engine





#weights_path="enco_weights4.hdf5"
weights_path='/home/a252/Documents/Python/CCE/Segmentation/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
Ynetweigh='Y10_net.hdf5'
ResWeight='resnet50_weights_tf_dim_ordering_tf_kernels.h5'



def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + 1e-10) / (union + 1e-10)


def dice(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    return (2 * intersection + 1e-10) /(K.sum(y_true) + K.sum(y_pred) + 1e-10)
def bce_dice_loss(y_true, y_pred):
    loss = (binary_crossentropy(y_true[:,:,:,0:7], y_pred[:,:,:,0:7]) + dice_coef_loss(y_true[:,:,:,0:7], y_pred[:,:,:,0:7]))+1.2*(binary_crossentropy(y_true[:,:,:,7:10],  y_pred[:,:,:,7:10]) +  dice_coef_loss(y_true[:,:,:,7:10],  y_pred[:,:,:,7:10]))+(binary_crossentropy(y_true[:,:,:,10:11], y_pred[:,:,:,10:11]) + dice_coef_loss(y_true[:,:,:,10:11], y_pred[:,:,:,10:11]))
    
    return loss
def dice_coef_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss
def dice_coeff(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    smooth=0.01
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    
LR_mult_dict = {}
reduce=0.01
LR_mult_dict['block1_conv1']=reduce
LR_mult_dict['block1_conv2']=reduce

LR_mult_dict['block2_conv1']=reduce
LR_mult_dict['block2_conv2']=reduce

LR_mult_dict['block3_conv1']=reduce
LR_mult_dict['block3_conv2']=reduce
LR_mult_dict['block3_conv3']=reduce
LR_mult_dict['block3_conv4']=reduce


LR_mult_dict['block4_conv1']=reduce
LR_mult_dict['block4_conv2']=reduce
LR_mult_dict['block4_conv3']=reduce
LR_mult_dict['block4_conv4']=reduce

LR_mult_dict['block5_conv1']=reduce
LR_mult_dict['block5_conv2']=reduce
LR_mult_dict['block5_conv3']=reduce
LR_mult_dict['block5_conv4']=reduce   
class CustomRMSprop(Optimizer):
    """RMSProp optimizer.
    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).
    This optimizer is usually a good choice for recurrent
    neural networks.
    # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """

    def __init__(self, lr=0.001, rho=0.9, epsilon=None, decay=0.,multipliers=None,
                 **kwargs):
        super(CustomRMSprop, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.rho = K.variable(rho, name='rho')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.lr_multipliers = multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        for p, g, a in zip(params, grads, accumulators):
            # update accumulator
            if p.name in self.lr_multipliers:
                new_lr = lr * self.lr_multipliers[p.name]
            else:
                new_lr = lr                 
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))
            new_p = p - new_lr * g / (K.sqrt(new_a) + self.epsilon)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': float(K.get_value(self.rho)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(CustomRMSprop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    
    

#Experiment ResUnet    
###################################################################
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same', name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def conv_block_vgg(input_tensor, filters, stage, block, strides=(2, 2)):
    conv_name_base = 'ColNet_' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_uniform')(input_tensor)
    x = BatchNormalization(name=bn_name_base)(x)
    x = Activation('selu')(x) 
    
    return x

def conv_block_decoder(input_tensor, filters, stage, block, strides=(2, 2)):
    conv_name_base = 'ColNet_' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_uniform')(input_tensor)
    x = BatchNormalization(name=bn_name_base)(x)
    x = Activation('selu')(x) 
    
    return x














############################ Y-RESNET##############################################
def YnetResNet(netpram,include_top=True, weights=None, input_tensor=None,  input_shape=None,pooling=None, classes=1000):


    include_top=False
    pooling='max'
    classes=1000
    if  netpram.task=='all':
        num_classes=11
    elif  netpram.task=='binary':
        num_classes=1
    elif  netpram.task=='parts':
        num_classes=3
    elif  netpram.task=='instrument':
        num_classes=7 
    #img_input=Input((224, 224, 3))
    inputs_L   = Input((224, 224, 3))
    inputs_R   = Input((224, 224, 3))
    
   
    inputs=[inputs_L,inputs_R]
    bn_axis = 3


    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inputs_L)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x_a = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x_a)

    x_b = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x_b, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x_c = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x_c, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x_d = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x_d, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x_e = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x_e, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if include_top:
        x = layers.AveragePooling2D((7, 7), name='avg_pool')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.

    inputs = inputs_L
    # Create model.
    modelen1 = Model(inputs, x, name='resnet50')
    # Load weights.
    weights_path = utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
    modelen1.load_weights(weights_path)
 #   for layer in modelen1.layers:
 #       layer.trainable=False
  
   # Loaded enoder one- untrainable
    
    x = layers.ZeroPadding2D(padding=(3, 3), name='enc2_conv1_pad')(inputs_R)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      name='enc2_conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='enc2_bn_conv1')(x)
    x_a_e2 = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x_a_e2)

    x_b_e2 = conv_block(x, 3, [64, 64, 256], stage=2, block='a_enc2', strides=(1, 1))
    x = identity_block(x_b_e2, 3, [64, 64, 256], stage=2, block='b_enc2')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c_enc2')

    x_c_e2 = conv_block(x, 3, [128, 128, 512], stage=3, block='a_enc2')
    x = identity_block(x_c_e2, 3, [128, 128, 512], stage=3, block='b_enc2')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c_enc2')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d_enc2')

    x_d_e2 = conv_block(x, 3, [256, 256, 1024], stage=4, block='a_enc2')
    x = identity_block(x_d_e2, 3, [256, 256, 1024], stage=4, block='b_enc2')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c_enc2')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d_enc2')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e_enc2')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f_enc2')

    x_e_e2 = conv_block(x, 3, [512, 512, 2048], stage=5, block='a_enc2')
    x = identity_block(x_e_e2, 3, [512, 512, 2048], stage=5, block='b_enc2_enc2')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c_enc2')

    if include_top:
        x = layers.AveragePooling2D((7, 7), name='avg_pool_enc2')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000_enc2')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.

    inputs = inputs_R
    # Create model.
    modelen2 = Model(inputs, x, name='resnet50')

    # Load weights.
    weights_path = utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
    modelen2.load_weights(weights_path)
        
     #Loaded encoder 2-trainable   
        
        
        
    
    
    #center = add([xold, ynew])
    #center = concatenate([xold, ynew], axis=3)
    center=conv_block_decoder(add([x_e,x_e_e2]), 1024, 7, 'center', strides=(2, 2))
    center=conv_block_decoder(center, 1024, 7, 'center_2', strides=(2, 2))
    # 32
    up4 = UpSampling2D((2, 2))(center)
    #upc=concatenate([x_d,x_d_e2], axis=3)
    up4 = concatenate([add([x_d,x_d_e2]), up4], axis=3)
    up4=conv_block_decoder(up4, 512, 32, 'deconv1', strides=(2, 2))
  #  up4=conv_block_decoder(up4, 512, 32, 'deconv2', strides=(2, 2))
    up4=conv_block_decoder(up4, 512, 32, 'deconv3', strides=(2, 2))
    
    # 64

    up3 = UpSampling2D((2, 2))(up4) 
   # upc=concatenate([down2T,down2B], axis=3)
    #upc=concatenate([x_c,x_c_e2], axis=3)
    up3 = concatenate([add([x_c,x_c_e2]), up3], axis=3)
    up3=conv_block_decoder(up3, 512, 64, 'deconv1', strides=(2, 2))
    up3=conv_block_decoder(up3, 512, 64, 'deconv2', strides=(2, 2))
    up3=conv_block_decoder(up3, 512, 64, 'deconv3', strides=(2, 2))
    # 128

    up2 = UpSampling2D((2, 2))(up3) 
    #upc = concatenate([down1T,down1B], axis=3)
    
    x_b = layers.ZeroPadding2D(padding=[1, 1], name='convdec_pad')(x_b)
    x_b = layers.Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)(x_b)
    
    x_b_e2 = layers.ZeroPadding2D(padding=[1, 1], name='convdec_pad2')(x_b_e2)
    x_b_e2 = layers.Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)(x_b_e2)
    up2 = concatenate([add([x_b,x_b_e2]), up2], axis=3)
    up2=conv_block_decoder(up2, 256, 128, 'deconv1', strides=(2, 2))
  #  up2=conv_block_decoder(up2, 256, 128, 'deconv2', strides=(2, 2))
    up2=conv_block_decoder(up2, 256, 128, 'deconv3', strides=(2, 2))
    # 256

    up1 = UpSampling2D((2, 2))(up2)  
    #upc=concatenate([down0T,down0B], axis=3)
    up1 = concatenate([add([x_a,x_a_e2]), up1], axis=3)
    up1=conv_block_decoder(up1, 128, 256, 'deconv1', strides=(2, 2))
    up1=conv_block_decoder(up1, 128, 256, 'deconv2', strides=(2, 2))
    up1=conv_block_decoder(up1, 128, 256, 'deconv3', strides=(2, 2))


    # 512

    up0a = UpSampling2D((2, 2))(up1)
    #upc= concatenate([down0aT,down0aB],axis=3)
  #  up0a = concatenate([img_input, up0a], axis=3)    
    up0a=conv_block_decoder(up0a, 64, 512, 'deconv1', strides=(2, 2))
    up0a=conv_block_decoder(up0a, 64, 512, 'deconv2', strides=(2, 2))
    up0a=conv_block_decoder(up0a, 64, 512, 'deconv3', strides=(2, 2))
  
    
    

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0a)
        

    model = Model(inputs=[inputs_L,inputs_R], outputs=classify,name='RESNetU')
   # optimizerc =CustomRMSprop(lr=0.00001,multipliers = LR_mult_dict)
    #lr=0.00001, F1=79.8
    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])
   # '''
    return model
    
