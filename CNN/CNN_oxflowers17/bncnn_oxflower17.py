# encoding: UTF-8

import tensorflow as tf
import math
import input_oxflowers17data


X = tf.placeholder(tf.float32, [None, 224, 224, 3])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 17])
# test flag for batch norm
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
# dropout probability
pkeep = tf.placeholder(tf.float32)
pkeep_conv = tf.placeholder(tf.float32)

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages

def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()

def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape


K = 16  # first convolutional layer output depth
L = 32  # second convolutional layer output depth
M = 64  # third convolutional layer
N = 512  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([7, 7, 3, K], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([7, 7, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([7, 7, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, 17], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [17]))

# The model
# batch norm scaling is not useful with relus
# batch norm offsets are used instead of biases
stride = 1  # output is 112x112
Y1l = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')
Y1P = tf.nn.max_pool(Y1l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
Y1bn, update_ema1 = batchnorm(Y1P, tst, iter, B1, convolutional=True)
Y1r = tf.nn.relu(Y1bn)
Y1 = tf.nn.dropout(Y1r, pkeep_conv, compatible_convolutional_noise_shape(Y1r))

stride = 2  # output is 28x28
Y2l = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
Y2P = tf.nn.max_pool(Y2l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
Y2bn, update_ema2 = batchnorm(Y2P, tst, iter, B2, convolutional=True)
Y2r = tf.nn.relu(Y2bn)
Y2 = tf.nn.dropout(Y2r, pkeep_conv, compatible_convolutional_noise_shape(Y2r))

stride = 2  # output is 7x7
Y3l = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
Y3P = tf.nn.max_pool(Y3l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
Y3bn, update_ema3 = batchnorm(Y3P, tst, iter, B3, convolutional=True)
Y3r = tf.nn.relu(Y3bn)
Y3 = tf.nn.dropout(Y3r, pkeep_conv, compatible_convolutional_noise_shape(Y3r))

YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4l = tf.matmul(YY, W4)
Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4)
Y4r = tf.nn.relu(Y4bn)
Y4 = tf.nn.dropout(Y4r, pkeep)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100
summary_loss = tf.summary.scalar('cross_entropy', cross_entropy)

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
summary_accuracy = tf.summary.scalar('accuracy', accuracy)
# training step
# the learning rate is: # 0.0001 + 0.02 * (1/e)^(step/1600)), i.e. exponential decay from 0.03->0.0001
lr = 0.0001 +  tf.train.exponential_decay(0.02, iter, 1600, 1/math.e)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

filewriter_path = "bncnn_output/filewriter"
merged_summary = tf.summary.merge([summary_accuracy , summary_loss])
train_writer = tf.summary.FileWriter(filewriter_path + '/train')
test_writer = tf.summary.FileWriter(filewriter_path + '/test')

# init

training_epochs = 2000
n_trainsamples = int(1360*0.8)
batch_size = 128
total_batch = int(n_trainsamples / batch_size)

# 分割+读取数据集
#input_oxflowers17data.Xy_train_test_split()

with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    img_batch, label_batch = input_oxflowers17data.get_train_batch("train.tfrecords", batch_size)
    X_test, y_test = input_oxflowers17data.get_test_data("test.tfrecords")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    X_test, y_test = sess.run([X_test, y_test])

    for epoch in range(training_epochs):
        for i in range(total_batch):
            # the backpropagation training step
            X_data, y_data = sess.run([img_batch, label_batch])
            sess.run(train_step, {X: X_data, Y_: y_data, tst: False, iter: i, pkeep: 0.85, pkeep_conv: 0.9})
            sess.run(update_ema, {X: X_data, Y_: y_data, tst: False, iter: i, pkeep: 1.0, pkeep_conv: 1.0})
            a, c, l, merged_sum = sess.run([accuracy, cross_entropy, lr, merged_summary],
                               feed_dict={X: X_data, Y_: y_data, iter: i, tst: False, pkeep: 1.0, pkeep_conv: 1.0})
            print(str(i + 1) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(l) + ")")
            train_writer.add_summary(merged_sum, epoch * total_batch + i + 1)

        a, c, merged_sum = sess.run([accuracy, cross_entropy, merged_summary],
                            feed_dict={X: X_test, Y_: y_test, tst: True, pkeep: 1.0, pkeep_conv: 1.0})
        print(str(i + 1) + ": ********* epoch " + str(epoch + 1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        test_writer.add_summary(merged_sum, ( epoch + 1 ) * total_batch)
    # When done, ask the threads to stop.
    coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)







