import os
import numpy as np
import tensorflow as tf
import h5py
import math


# In[38]:


import tensorflow as tf
import numpy as np
import scipy.misc
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow


get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)
tf.random.set_seed(2)


# # Identy block

# In[39]:


def identity_block(X, f, filters):
    """
    Implementation of the identity block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    initializer -- to set up the initial weights of a layer. Equals to random uniform initializer
    
    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X) # Default axis
    X = Activation('relu')(X)
    
    
    # Second component of main path 
    
    X = Conv2D(filters = F2,kernel_size= f,strides=(1,1),padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation("relu")(X)

    # Third component of main path 
    
    X = Conv2D(filters = F3,kernel_size = 1,strides = (1,1),padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)
    
    # Add shortcut value to main path, and pass it through a RELU activation 
    X = Activation("relu")(X)
    
    return X


# # Convolutional block

# In[40]:


def convolutional_block(X, f, filters, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    s -- Integer, specifying the stride to be used
    initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer, 
                   also called Xavier uniform initializer.
    
    Returns:
    X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    
    # First component of main path 
    X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    
    
    ## Second component of main path 
    X = Conv2D(filters = F2,kernel_size= f,strides=(1,1),padding = 'same')(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation("relu")(X)

    ## Third component of main path 
    X = Conv2D(filters = F3,kernel_size = 1,strides = (1,1),padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)
    
    ##### SHORTCUT PATH ##### 
    X_shortcut = Conv2D(filters = F3,kernel_size = 1,strides = (s,s),padding = 'valid')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut)
    
    
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X


# # ResNet 50

# In[41]:


def ResNet50(input_shape = (64, 64, 3), classes = 6, activation = "softmax"):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    
    
    ## Stage 3 
    
    X = convolutional_block(X,f=3,filters=[128,128,512],s =2)
    
    X = identity_block(X,f=3, filters=[128,128,512])
    X = identity_block(X,f=3, filters=[128,128,512])
    X = identity_block(X,f=3, filters=[128,128,512])

    # Stage 4 
    X = convolutional_block(X,f = 3, filters=[256,256,1024],s=2)
    
    
    X = identity_block(X,f=3,filters=[256,256,1024])
    X = identity_block(X,f=3,filters=[256,256,1024])
    X = identity_block(X,f=3,filters=[256,256,1024])
    X = identity_block(X,f=3,filters=[256,256,1024])
    X = identity_block(X,f=3,filters=[256,256,1024])

    # Stage 5 
    X = convolutional_block(X,f=3,filters=[512,512,2048],s=2)
    
  
    X = identity_block(X,f=3,filters=[512,512,2048])
    X = identity_block(X,f=3,filters=[512,512,2048])

    X = AveragePooling2D()(X)
    
  

    # output layer
    X = Flatten()(X)
    X = Dense(512,activation='relu')(X)
    

    X = Dense(classes, activation=activation)(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X)

    return model
