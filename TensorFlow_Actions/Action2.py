#Create a multi-layer CNN work
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/home/spyder/TensorFlow_Learning/TensorFlow_Actions/MNIST_data/",one_hot=True)

sess=tf.InteractiveSession()
#prepare for the batch
x=tf.placeholder("float",shape=[None,784])
y_=tf.placeholder("float",shape=[None,10])

def weight_variable(shape):
    """
    In order to create this model,we need a large number of weights and biases.A little
    noise should be added to the model,for the purpose of breaking the zero gradient.As
    we use the ReLU active function,it's a good option to initial the bias with a tiny positive
    :param shape:
    :return:
    """
    initial=tf.truncated_normal(shape,stddev=0.1)   #标准差的值等于0.1
    #产生正态分布的值如果和均值的差大于两倍的标准差,就会重新生成.和一般的正态分布的产生的随机数据相比起来，
    #这个函数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的其他的函数是可能的.
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return  tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')


def max_pool_2(x):
    """
    implement the 2*2 pooling layer with stride 2
    :param x:
    :return:
    """
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

'''
Here we comes the first conv layer
'''
W_conv1=weight_variable([5,5,1,32])
b_con1=weight_variable([32])

#reshape the input image and x is the original image data
image_new=tf.reshape(x,[-1,28,28,1])
h_conv1=tf.nn.relu(conv2d(image_new,W_conv1)+b_con1)
h_pool1=max_pool_2(h_conv1)

'''
Here we come the second conv layer
'''
W_conv2=weight_variable([5,5,32,64])
b_conv2=weight_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2(h_conv2)

'''
Here we come the affine layer
'''
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

conv2affine=tf.reshape(h_pool2,[-1,7*7*64])
fc_1=tf.nn.relu(tf.matmul(conv2affine,W_fc1)+b_fc1)

keep_prob=tf.placeholder("float")
dropout=tf.nn.dropout(fc_1,keep_prob)

W_fc2=weight_variable([1024,10])
b_fc2=weight_variable([10])

y_conv=tf.nn.softmax(tf.matmul(dropout,W_fc2)+b_fc2)

cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
sess.run(tf.initialize_all_variables())
for i in range(2000):
    batch=mnist.train.next_batch(50)
    # print(batch[0][2])
    # print(batch[1].shape)
    if i %100==0:
        train_accuracy=accuracy.eval(feed_dict={
            x:batch[0],y_:batch[1],keep_prob:1.0
        })
        print("step %d,training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
# print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,
#                                                   y_:mnist.test.labels,
#                                                   keep_prob:1.0}))
print ("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images[0:500], y_: mnist.test.labels[0:500], keep_prob: 1.0}))

#Finally the output layer

