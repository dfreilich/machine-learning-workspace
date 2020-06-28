import tensorflow as tf
import numpy as np
import math
import time
from hw2_utils import get_CIFAR10_data,run_model
from hw2_part3_cpu import model2
##Step1: load the data
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
## Step2: test your GPU
tf.reset_default_graph()

# define our input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
x = np.random.randn(64, 32, 32,3)

# define model
y_out = model2(X,y,is_training)
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
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)
cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(cross_entropy_loss)

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)
    
## Train the model
#tf.reset_default_graph()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print('Training')
run_model(sess,y_out,mean_loss,X,y,is_training,X_train,y_train,1,64,100,train_step)

##Test the model
print('Validation')
run_model(sess,y_out,mean_loss,X,y,is_training,X_val,y_val,1,64)
