#encoding=utf-8

import input_oxflowers17data
import tensorflow as tf
import numpy as np

# weight variable init
def weight_variable(shape, stddev=0.03):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

# parameter init method
def xavier_init(fan_in, fan_out, constant=-1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

# bias variable init
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


# input layer
x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
keep_probability = tf.placeholder(tf.float32)

# hidden layer 1
W_conv1 = weight_variable([7,7,3,16])
b_conv1 = bias_variable([16])
conv1 = tf.nn.bias_add(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME'), b_conv1)
relu1 = tf.nn.relu(conv1)
pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# hidden layer 2
W_conv2 = weight_variable([7,7,16,32])
b_conv2 = bias_variable([32])
conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1, W_conv2, strides=[1, 2, 2, 1], padding='SAME'), b_conv2)
relu2 = tf.nn.relu(conv2)
pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#hidden layer 3
W_conv3 = weight_variable([7,7,32,64])
b_conv3 = bias_variable([64])
conv3 = tf.nn.bias_add(tf.nn.conv2d(pool2, W_conv3, strides=[1, 2, 2, 1], padding='SAME'), b_conv3)
relu3 = tf.nn.relu(conv3)
pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#全连接层
W_fc1 = tf.Variable(xavier_init(7 * 7 * 64, 512))
b_fc1 = bias_variable([512])

h_pool3_flat = tf.reshape(pool3, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob = keep_probability)
#输出层
W_fc2 = tf.Variable(xavier_init(512, 17))
b_fc2 = bias_variable([17])


y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

y_ = tf.placeholder(tf.float32, [None,17])
cross_entropy = tf.reduce_sum((tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y_conv)))
tv = tf.trainable_variables() #得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
regularization_cost = 0.001 * tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ]) #0.001是lambda超参数
cost = regularization_cost + cross_entropy
summary_loss = tf.summary.scalar('cross_entropy', cost)

train_step = tf.train.AdamOptimizer(0.001).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
summary_accuracy = tf.summary.scalar('accuracy', accuracy)

filewriter_path = "cnn_output/filewriter"
merged_summary = tf.summary.merge([summary_accuracy , summary_loss])
train_writer = tf.summary.FileWriter(filewriter_path + '/train')
test_writer = tf.summary.FileWriter(filewriter_path + '/test')



training_epochs = 2000
n_trainsamples = int(1360*0.8)
batch_size = 128
total_batch = int(n_trainsamples / batch_size)

# 分割+读取数据集
# input_oxflowers17data.Xy_train_test_split()

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
            X_data, y_data = sess.run([img_batch, label_batch])
            _train_step, _accuracy, merged_sum = sess.run([train_step, accuracy, merged_summary],
                                                     feed_dict={x: X_data, y_: y_data, keep_probability: 0.75})
            print(i + 1, _accuracy)
            train_writer.add_summary(merged_sum, epoch * total_batch + i + 1)

        test_accuracy,merged_sum = sess.run([accuracy, merged_summary], feed_dict={x: X_test, y_: y_test, keep_probability: 1.0})
        print("Epoch " + str(epoch + 1) + ", test accuracy " + str(test_accuracy))
        test_writer.add_summary(merged_sum, (epoch + 1) * total_batch)
    # When done, ask the threads to stop.
    coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)


