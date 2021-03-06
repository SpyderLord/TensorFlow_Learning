import tensorflow as tf

def simple_model(X,y):
    '''

    :param X:has the shape [NHWC]
    :param y:
    :return:
    '''
    Wconv1=tf.get_variable('Wconv1',shape=[7,7,3,32])   #imply the number of filter is 32
    bconv1=tf.get_variable('bconv1',shape=[32])
    W1=tf.get_variable('W1',shape=[5408,10])
    b1=tf.get_variable('b1',shape=[10])

    #define the graph(two_layer convnet)
    a1=tf.nn.conv2d(X,Wconv1,strides=[1,2,2,1],padding='VALID')+bconv1
    h1=tf.nn.relu(a1)
    h1_flat=tf.reshape(h1,[-1,5408])
    y_out=tf.matmul(h1_flat,W1)+b1
    return y_out

#

# run a simple model
# with tf.Session() as sess:
#     with tf.device("/gpu:0"): #"/cpu:0" or "/gpu:0"
#         sess.run(tf.global_variables_initializer())
#         print('Training')
#         run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step,True )
#         print('Validation')
#         run_model(sess,y_out,mean_losls s,X_val,y_val,1,64)
#         sess.close()
#         print(tf.get_default_graph())
#         print('!')

#run a complex model