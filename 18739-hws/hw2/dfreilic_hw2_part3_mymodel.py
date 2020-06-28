import tensorflow as tf
import numpy as np
import math
from hw2_utils import *
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

def my_model(X,y,is_training):    
    ##Layer 1 – Conv, Batch Normalization, Max Pool (3x3, stride of 2)
    conv1 = tf.contrib.layers.conv2d(inputs=X, num_outputs=64, kernel_size=5) 
    batch_norm1 = tf.contrib.layers.batch_norm(inputs=conv1, is_training=is_training)
    max_pool1 = tf.contrib.layers.max_pool2d(inputs=batch_norm1, kernel_size=[3, 3], stride=2)

    ##Layer 2 – Conv, Batch Normalization, Max Pool (3x3, stride of 2)
    conv2 = tf.contrib.layers.conv2d(inputs=max_pool1, num_outputs=64, kernel_size=5) 
    batch_norm2 = tf.contrib.layers.batch_norm(inputs=conv2, is_training=is_training)
    max_pool2 = tf.contrib.layers.max_pool2d(inputs=batch_norm2, kernel_size=[3,3], stride=2)

    ##Layer 3 – Conv, Relu, Batch Normalization
    conv3 = tf.contrib.layers.conv2d(inputs=max_pool2, num_outputs=128, kernel_size=3) 
    relu3 = tf.nn.relu(conv3)
    batch_norm3 = tf.contrib.layers.batch_norm(inputs=relu3, is_training=is_training)

    ##Layer 4 – Conv, Relu
    conv4 = tf.contrib.layers.conv2d(inputs=batch_norm3, num_outputs=128, kernel_size=3) 
    # batch_norm4 = tf.contrib.layers.batch_norm(inputs=conv4, is_training=is_training)
    relu4 = tf.nn.relu(conv4)
    
    ##Layer 5 – Conv, Relu, Batch Norm, Max Pool (3x3, stride of 2)
    conv5 = tf.contrib.layers.conv2d(inputs=relu4, num_outputs=128, kernel_size=3) 
    relu5 = tf.nn.relu(conv5)
    batch_norm5 = tf.contrib.layers.batch_norm(inputs=relu5, is_training=is_training)
    max_pool5 = tf.contrib.layers.max_pool2d(inputs=batch_norm5, kernel_size=[3,3], stride=2)
    
    ##Layer 6 – Fully connected layer, Relu activation
    flat6 = tf.contrib.layers.flatten(inputs=max_pool5)
    dense6 = tf.contrib.layers.fully_connected(inputs=flat6, num_outputs=384, activation_fn=tf.nn.relu)
    
    ##Layer 7 – Fully connected layer, Relu activation
    dense7 = tf.contrib.layers.fully_connected(inputs=dense6, num_outputs=192, activation_fn=tf.nn.relu)

    ##Layer 8 – Affine layer, softmax output
    y_out = tf.contrib.layers.fully_connected(inputs=dense7, num_outputs=10, activation_fn=None)
    return y_out

tf.reset_default_graph()

# define our input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
x = np.random.randn(64, 32, 32,3)

# define model
y_out = my_model(X,y,is_training)
try:
    with tf.Session() as sess:
        with tf.device("/gpu:0") as dev: #"/cpu:0" or "/gpu:0"
            tf.global_variables_initializer().run()
            ans = sess.run(y_out,feed_dict={X:x,is_training:True})
except tf.errors.InvalidArgumentError:
    print("no gpu found, please use PSC if you want GPU acceleration")    
    # rebuild the graph
    # trying to start a GPU throws an exception 
    # and also trashes the original graph
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)
    y_out = model2(X,y,is_training)
    
##Step3: Set up your mean_loss and optimizer for model2 (To do)
#Set up an RMSprop optimizer (using a 1e-3 learning rate) and a crossentropy loss function.
optimizer = tf.train.AdamOptimizer(learning_rate=.001)
cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(cross_entropy_loss)

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)
    
## Train the model
#tf.reset_default_graph()
with tf.Session() as sess:
    with tf.device("/gpu:0"): #"/cpu:0" or "/gpu:0"
        sess.run(tf.global_variables_initializer())
        print('Training')
        run_model(sess,y_out,mean_loss,X,y,is_training,X_train,y_train,30,64,500,train_step)
        ##Test the model
        print('Validation')
        run_model(sess,y_out,mean_loss,X,y,is_training,X_val,y_val,1,64)

##Report your test result: Remeber: only run this once
# print('Test')
# run_model(sess,y_out,mean_loss,X,y,is_training,X_test,y_test,1,64)

