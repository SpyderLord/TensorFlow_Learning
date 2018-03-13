import tensorflow as tf
import numpy as np

def AlexNet_Model(X,y,is_training):
    """
    build a convolutional network with the architecture of AlexNet.
    The architecture is
    conv1->max pool1->norm1->conv2->max pool2->norm2->conv3->conv4->conv5->max pool3
    ->fc6->fc7->fc8
    :param X:cifar-10-data
    :param y:label names
    :param is_training:
    :return:10 classes scores for N input classes
    """
    #initialize the parameters
    conv_W1=tf.get_variable('conv_W1',shape=[4,4,3,48])     #we use 96 filters at the conv1 layer
    conv_b1=tf.get_variable('conv_b1',shape=[48])
    conv_W2=tf.get_variable('conv_W2',shape=[4,4,48,128])   #we use 256 filters at the conv2 layer
    conv_b2=tf.get_variable('conv_b2',shape=[128])
    conv_W3=tf.get_variable('conv_W3',shape=[3,3,128,128])  #we use 384 filters at the conv3 layer
    conv_b3=tf.get_variable('conv_b3',shape=[128])
    conv_W4=tf.get_variable('conv_W4',shape=[3,3,128,192])  #we use 384 filters at the conv4 layer
    conv_b4=tf.get_variable('conv_b4',shape=[192])
    conv_W5=tf.get_variable('conv_W5',shape=[3,3,192,128])  #we use 384 filters at the conv5 layer
    conv_b5=tf.get_variable('conv_b5',shape=[128])
    fc_W1=tf.get_variable('fc_W1',shape=[6272,3136])
    fc_b1=tf.get_variable('fc_b1',shape=[3136])
    fc_W2=tf.get_variable('fc_W2',shape=[3136,630])
    fc_b2=tf.get_variable('fc_b2',shape=[630])
    fc_W3=tf.get_variable('fc_W3',shape=[630,10])
    fc_b3=tf.get_variable('fc_b3',shape=[10])

    #create the layer model
    conv1=tf.nn.conv2d(X,conv_W1,strides=[1,1,1,1],padding='VALID')+conv_b1
    relu1=tf.nn.relu(conv1)
    bn1=tf.layers.batch_normalization(relu1,training=is_training)

    conv2=tf.nn.conv2d(bn1,conv_W2,strides=[1,1,1,1],padding='VALID')+conv_b2
    relu2=tf.nn.relu(conv2)
    pool1=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    bn2=tf.layers.batch_normalization(pool1,training=is_training)

    conv3=tf.nn.conv2d(bn2,conv_W3,strides=[1,1,1,1],padding='VALID')+conv_b3
    relu3=tf.nn.relu(conv3)
    conv4=tf.nn.conv2d(relu3,conv_W4,strides=[1,1,1,1],padding='VALID')+conv_b4
    relu4=tf.nn.relu(conv4)
    conv5=tf.nn.conv2d(relu4,conv_W5,strides=[1,1,1,1],padding='VALID')+conv_b5
    relu5=tf.nn.relu(conv5)
    conv5_new=tf.reshape(relu5,[-1,6272])

    affine1=tf.matmul(conv5_new,fc_W1)+fc_b1
    relu6=tf.nn.relu(affine1)
    affine2=tf.matmul(relu6,fc_W2)+fc_b2
    relu7=tf.nn.relu(affine2)
    out=tf.matmul(relu7,fc_W3)+fc_b3
    return out

