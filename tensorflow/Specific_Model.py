import numpy as np
import tensorflow as tf

#train a specific model with the architecture
# tf.reset_default_graph()
# print(tf.get_default_graph())

#define the model:
def complex_model(X,y,is_training):
    #initialize the paramters
    Wconv1=tf.get_variable('Wconv1',shape=[7,7,3,32])
    bconv1=tf.get_variable('bconv1',shape=[32])         #kernel with size 7*7,32 filters in total
    W1=tf.get_variable('W1',shape=[5408,1024])
    b1=tf.get_variable('b1',shape=[1024])
    W2=tf.get_variable('W2',shape=[1024,10])
    b2=tf.get_variable('b2',shape=[10])
    #layer process
    conv1=tf.nn.conv2d(X,Wconv1,strides=[1,1,1,1],padding='VALID')
    relu1=tf.nn.relu(conv1)
    relu1_bn=tf.layers.batch_normalization(relu1,training=is_training)
    max_pool=tf.nn.max_pool(relu1_bn,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    relu1_reshape=tf.reshape(max_pool,[-1,5408])
    affine1=tf.matmul(relu1_reshape,W1)+b1
    relu2=tf.nn.relu(affine1)
    out=tf.matmul(relu2,W2)+b2
    return out
