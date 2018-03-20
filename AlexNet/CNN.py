import tensorflow as tf
import numpy as np
import math
import timeit
# import matplotlib.pyplot as plt

from get_data import get_data
from simple_model import *
from Specific_Model import *
from AlexNet import *

X_train, y_train, X_val, y_val, X_test, y_test = get_data()


def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss, correct_prediction, accuracy]
    if training_now:
        variables[-1] = training

    # counter
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
            # generate indicies for the batch
            start_idx = (i * batch_size) % Xd.shape[0]
            idx = train_indicies[start_idx:start_idx + batch_size]

            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx, :],
                         y: yd[idx],
                         is_training: training_now}
            # get batch size
            actual_batch_size = yd[idx].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables, feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss * actual_batch_size)
            correct += np.sum(corr)

            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
                      .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
            iter_cnt += 1
        total_correct = correct / Xd.shape[0]
        total_loss = np.sum(losses) / Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}" \
              .format(total_loss, total_correct, e + 1))
    return total_loss, total_correct

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
tf.reset_default_graph()                        #clear old variables
X=tf.placeholder(tf.float32,[None,32,32,3])     #insert a placeholder for a tensor that will be always fed
#it is important that this tensor will produce an error if evaluated.
#its value must be fed using the feed_dict optional to session.run temsor.eval() or operation.run()
y=tf.placeholder(tf.int64,[None])
is_training=tf.placeholder(tf.bool)

y_out=AlexNet_Model(X,y,is_training)

#define the softmax loss:
mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,10), logits=y_out))

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)

with tf.Session() as sess:
    with tf.device("/gpu:0"):
        sess.run(tf.global_variables_initializer())
        print('Training')
        run_model(sess, y_out, mean_loss, X_train, y_train, 10, 64, 100, train_step, True)
        print('Validation')
        run_model(sess, y_out, mean_loss, X_val, y_val, 1, 64)
        print('Test')
        run_model(sess, y_out, mean_loss, X_test, y_test, 1, 64)