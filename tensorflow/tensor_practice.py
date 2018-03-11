import tensorflow as tf
import numpy as np
#
# x_data=np.random.rand(2,100)
# x_data=x_data.astype(np.float32)
# # w=np.random.rand()
# w=np.array([0.100,0.200])
# y_data=np.dot(w,x_data)+0.3         #linear process
# # print(x_data)
# # b=tf.Variable(tf.zeros[1])
# # print(b)
#
# # create a linear model
# b=tf.Variable(tf.zeros([1]))
# w=tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
# y=tf.matmul(w,x_data)+b
#
# #minimize the variance
# loss=tf.reduce_mean(tf.square(y-y_data))
# optimizer=tf.train.GradientDescentOptimizer(0.5)
# train=optimizer.minimize(loss)
#
# init=tf.initialize_all_variables()
# sess=tf.Session()
# sess.run(init)
# for step in range(0,201):
#     sess.run(train)
#     if step %20==0:
#         print(step,sess.run(w),sess.run(b))
#
#
# W1=tf.get_variable("W1",shape=[7,7,3,32])
# print(W1.dtype)
# print(W1[0])
# print()

# x=tf.placeholder(tf.float32,shape=(1024,1024))
# y=tf.matmul(x,x)      #this will fail cause it was not fed
# rand_array=np.random.rand(1024,1024)
# feed_dict={x:rand_array}    #its value must be fed using the feed_dict optional argument to
# print(type(feed_dict[x]))
# with tf.Session() as sees:
# print(tf.Session().run(y,feed_dict))
# training=None
# training_now=training is not None
# print(training_now)

# c=tf.constant(value=1)
# print(c.graph)
# print(tf.get_default_graph())       #once we start our mission,the graph will be created
# X=tf.placeholder(tf.float32,[None,32,32,3])
# y=tf.placeholder(tf.int64,int[None])
# is_training=tf.placeholder(tf.bool)

tf.reset_default_graph()
#create a new graph
# a=tf.constant(9)
# b=tf.constant(19)
# c=a*b
#
# # create a new session
# sees=tf.Session()
# d=sees.run(c)
# print(d)
# print(type(d))      #return a numpy value
# print(d.dtype)
# # sees.close()
# print(tf.get_default_graph())
# print('!')
# tf.reset_default_graph()
# a=tf.constant(323.3)
# print(a)
# s=tf.Session()
# a=s.run(a)
# print(a)
# print(type(a))
# print(tf.get_default_graph)
# my_Graph=tf.Graph()
# with my_Graph.as_default():
# a=tf.get_variable('a',shape=[32])
    # print('!')
    # print('\n')
    # a=tf.constant(32.0)
    # b=tf.constant([[1.2,2.0],[2,4]])
    # print(b.op)
    # a=tf.Session().run(a)
    # print(a)
# with tf.Session() as sees:
#     a=sees.run(a)
#     print(a)

def foo():
    with tf.variable_scope('foo',reuse=tf.AUTO_REUSE):
        v=tf.get_variable('v',shape=[1])
        return v
v1=foo()
v2=foo()
print(tf.Session().run(v1))
print(v2)

assert v1==v2

