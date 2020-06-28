import tensorflow as tf
import numpy as np

# Small epsilon value for the BN transform
def model2(X,y,is_training):
    #Hint: you need is_training for batch_normalization
    # Your code starts here
    
    #Layer 1 – 7x7 Conv, Relu Activation
    conv1 = tf.contrib.layers.conv2d(inputs=X, num_outputs=32, kernel_size=7) 
    
    #Layer 2 – Batch Normalization, Max Pool (2x2, stride of 2)
    batch_norm = tf.contrib.layers.batch_norm(inputs=conv1, is_training=is_training)
    max_pool2 = tf.contrib.layers.max_pool2d(inputs=batch_norm, kernel_size=[2, 2], stride=2)
    
    #Layer 3 – Affine layer, Relu Activation
    flat3 = tf.contrib.layers.flatten(inputs=max_pool2)
    dense3 = tf.contrib.layers.fully_connected(inputs=flat3, num_outputs=1024, activation_fn=tf.nn.relu)
    
    #Layer 4 – Affine layer, softmax output
    y_out = tf.contrib.layers.fully_connected(inputs=dense3, num_outputs=10, activation_fn=None)
    return y_out