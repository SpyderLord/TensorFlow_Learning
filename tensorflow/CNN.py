import tensorflow as tf
import numpy as np
import math
import timeit
# import matplotlib.pyplot as plt

from get_data import get_data
from simple_model import *
from Specific_Model import *

X_train, y_train, X_val, y_val, X_test, y_test = get_data()
# print('Train data shape: ', X_train.shape)
# print('Train labels shape: ', y_train.shape)
# print('Validation data shape: ', X_val.shape)
# print('Validation labels shape: ', y_val.shape)
# print('Test data shape: ', X_test.shape)
# print('Test labels shape: ', y_test.shape)
# train a neural newtwork with the help of tensorflow
#when it comes to the loss function,we use the hinge loss.The hinge loss is used for maxium-margin classification,most
#notable for SVM
#softmax使用的是熵来进行损失函数的计算 所以这个例程中应该是使用折叶损失配合SVM损失函数

"""
there are five parameters used in this function and let's have a clear comprehension about the 
fouth parameter--padding
this paramter determines the different mode of convolution with only two optional choices--valid 
and same.
when we choose the valid mode we do not make padding process 
Note that Valid only ever drops the right-most columns 会舍弃最右边的一些列 当滤波器在输入上进行滑动的时候，
如果不能进行整除就会舍弃最右边的列，就相当于是在出发运算中保留商但是舍弃余数
And Same mode tyies to pad evenly left and right,but if the amount of columns to be 
added is odd,it will add the extra column to the right 在这种模式下需要补充元素，左奇右偶
"""
tf.reset_default_graph()    #clear old variables

X=tf.placeholder(tf.float32,[None,32,32,3]) #insert a placeholder for a tensor that will be always fed
#it is important that this tensor will produce an error if evaluated.
#its value must be fed using the feed_dict optional to session.run temsor.eval() or operation.run()

y=tf.placeholder(tf.int64,[None])
is_training=tf.placeholder(tf.bool)

# y_out=simple_model(X,y)
# # print(y_out)
# #define the loss function
# total_loss=tf.losses.hinge_loss(tf.one_hot(y,10),logits=y_out)
# mean_loss=tf.reduce_mean(total_loss)

# #define the optimizer
# optimizer=tf.train.AdamOptimizer(5e-4)
# train_step=optimizer.minimize(mean_loss)

# def run_model(session,predict,loss_val,Xd,yd,
#               epochs=1,batch_size=64,print_every=100,
#               training=None,plot_losses=False):
#     #have tensorflow compute accuracy
#     correct_prediction=tf.equal(tf.argmax(predict,1),y)
#     accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#
#     #shuffle indicies
#     train_indicies=np.arange(Xd.shape[0])
#     np.random.shuffle(train_indicies)       #make it chaos
#     training_now=training is not None
#
#     #set up variables we want to compute
#     variables=[mean_loss,correct_prediction,accuracy]
#     if training_now:
#         variables[-1]=training
#     #counter
#     iter_cnt=0          #create a counter
#     for e in range(epochs):
#         #monitor the losses and accuracy
#         correct=0
#         losses=[]
#         #make sure that we iterate over the dataset once
#         for i in range(int(math.ceil(Xd.shape[0]/batch_size))):  #if we rewrite the code like Xd.shape[0]//batch_size
#             #if this is right ,it will make the code much more brief
#             start_idx=(i*batch_size)%Xd.shape[0]
#             idx=train_indicies[start_idx:start_idx+batch_size]
#             feed_dict={X:Xd[idx,:],y:yd[idx],is_training:training_now}
#             actual_batch_size=yd[idx].shape[0]
#             loss,corr,_=session.run(variables,feed_dict=feed_dict)
#             losses.append(loss*actual_batch_size)
#             correct+=np.sum(corr)
#
#             if training_now and (iter_cnt % print_every) == 0:
#                 print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
#                       .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
#             iter_cnt += 1
#         total_correct=correct/Xd.shape[0]
#         total_loss=np.sum(losses)/Xd.shape[0]       #calculate the average value
#         print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}" \
#               .format(total_loss, total_correct, e + 1))
#         # if plot_losses:
#         #     plt.plot(losses)
#         #     plt.grid(True)
#         #     plt.title('Epoch {} Loss'.format(e + 1))
#         #     plt.xlabel('minibatch number')
#         #     plt.ylabel('minibatch loss')
#         #     plt.show()
#         return total_loss, total_correct

# run a simple model
# with tf.Session() as sess:
#     with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0"
#         sess.run(tf.global_variables_initializer())
#         print('Training')
#         run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step,True )
#         print('Validation')
#         run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
#         sess.close()
#         print(tf.get_default_graph())
#         print('!')

#run a complex model
y_out=complex_model(X,y,is_training)

x=np.random.randn(64,32,32,3)                       #NHWC
with tf.Session() as sess:
    with tf.device("/cpu:0"):# #
# def simple_model(X,y):
#     '''
#
#     :param X:has the shape [NHWC]
#     :param y:
#     :return:
#     '''
#     Wconv1=tf.get_variable('Wconv1',shape=[7,7,3,32])   #imply the number of filter is 32
#     bconv1=tf.get_variable('bconv1',shape=[32])
#     W1=tf.get_variable('W1',shape=[5408,10])
#     b1=tf.get_variable('b1',shape=[10])
#
#     #define the graph(two_layer convnet)
#     a1=tf.nn.conv2d(X,Wconv1,strides=[1,2,2,1],padding='VALID')+bconv1
#     h1=tf.nn.relu(a1)
#     h1_flat=tf.reshape(h1,[-1,5408])
#     y_out=tf.matmul(h1_flat,W1)+b1
#     return y_out

        tf.global_variables_initializer().run()
        ans = sess.run(y_out,feed_dict={X:x,is_training:True})
        # %timeit sess.run(y_out,feed_dict={X:x,is_training:True})
        print(ans.shape)
        print(np.array_equal(ans.shape,np.array([64,10])))
