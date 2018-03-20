import tensorflow as tf
import numpy as np
# import input_data
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/home/spyder/TensorFlow_Learning/TensorFlow_Actions/MNIST_data/",one_hot=True)
# from mnist import  read_data_sets
# mnist=read_data_sets("/home/spyder/TensorFlow_Learning/TensorFlow_Actions/MNIST_data/",one_hot=True)
# print(mnist.test)
print('Training data size:',mnist.train.num_examples)
print(mnist.train.images.shape)

sess=tf.InteractiveSession()
'''
We take  use of the more flexible class InteractiveSession.We can code in a more efficient way with respect to it.
When we run a graph,we can insert some calculations constructed by some operations.If we don't use this class,
we have to create the whole graph before the session is active.
'''

x=tf.placeholder("float",shape=[None,784])
y_=tf.placeholder("float",shape=[None,10])
"""
我们通过为输入图像和目标输出类别创建节点，来开始构建计算图,这里的x和y都不是特定的值,他们都只是一个占位符,可以在TensorFlow运行某一计算时根据
占位符输入具体的值.
"""

w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
'''
Before variables can be used within a session,they must be initialized using that session.This step takes the initial
This step takes the initial values that have already been specified,and assigns them to each variable.This can be done 
for all variables at once.
'''
# sess.run(tf.initialize_all_variables())
sess.run(tf.global_variables_initializer())
# print(type(w))
y=tf.nn.softmax(tf.matmul(x,w)+b)               #calculate the probability for each class
cross_entropy=-tf.reduce_sum(y_*tf.log(y))      #we take the whole minibatch into calculation

#定义好模型和训练用的损失函数,用TensorFlow进行训练就很简单了。因为TensorFlow知道整个计算图，它可以使用计算图
#它可以使用自动微分法找到对于各个变量的损失的梯度值.
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
for i in range(1000):
    batch=mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0],y_:batch[1]})

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#cast a tensor to a new type
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
print(accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

