# -*- coding: utf -8 -*-'
import numpy as np

import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
from tensorflow.contrib import rnn
from input_sentimentdata import get_sentimentdata

# data head=[sentence, sentiment values, vec, length, label]
X, XLen, y = get_sentimentdata()
X_train, X_test, XLen_train, XLen_test, y_train, y_test = train_test_split(X, XLen, y, test_size=0.2)
print ('train size %d ,test size %d' % (len(X_train), len(X_test)))

batch_offset = 0
epochs_completed = 0

# get batch data
def get_random_block_from_data(batch_size):
    global batch_offset, epochs_completed, X_train, XLen_train, y_train
    start = batch_offset
    batch_offset += batch_size
    if batch_offset > X_train.shape[0]:
        # Shuffle the data
        perm = np.arange(X_train.shape[0])
        np.random.shuffle(perm)
        X_train = X_train[perm]
        XLen_train = XLen_train[perm]
        y_train = y_train[perm]
        # Start next epoch
        start = 0
        batch_offset = batch_size

    end = batch_offset

    return X_train[start:end], XLen_train[start:end], y_train[start:end]



if __name__ == '__main__':


#################### training parameters setting

    train_params = {'initial_lr': 0.001,
                    'lr_decay': 0.6,
                    'batch_size': 64,
                    'emdedding_dim':50,
                    'hidden_neural_size':64,
                    'hidden_layer_num':4,
                    'max_len':40,
                    'valid_num':100,
                    'class_num':5,
                    'keep_prob':0.6,
                    'num_epoch':100,
                    'n_steps':34,
                    'dispaly_epoch':50
                    }
    ### build lstm model
    n_steps = train_params['n_steps']
    n_inputs = train_params['emdedding_dim']
    n_classes = train_params['class_num']
    n_hidden = train_params['hidden_neural_size']
    lr = train_params['initial_lr']
    batch_size = train_params['batch_size']
    num_epoch = train_params['num_epoch']
    display_step = train_params['dispaly_epoch']
    total_batch = int(len(X_train) / batch_size)

    x = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
    y = tf.placeholder(tf.int64, [None])

    weights = {
        # Hidden layer weights => 2*n_hidden because of forward + backward cells
         'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
         }
    biases = {'out': tf.Variable(tf.random_normal([n_classes]))}

    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    rnn_inputs = tf.unstack(x, axis=1)
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, rnn_inputs,dtype = tf.float32)
    pred = tf.matmul(outputs[-1], weights['out']) + biases['out']

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    correct_pred = tf.equal(tf.argmax(pred, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []

        for epoch in range(num_epoch):
            train_acc = 0
            train_loss = 0
            for i in range(total_batch):
                X_, XLen_, Y_ = get_random_block_from_data(batch_size)
                optimizer_, training_loss_, acc_ = sess.run([optimizer, loss, accuracy], feed_dict={x: X_, y: Y_})
                train_acc = train_acc + acc_
                train_loss = train_loss + training_loss_
            print("accuracy:" + str(train_acc/total_batch) + " loss: " + str(train_loss/total_batch))
            loss_, c_ = sess.run([loss, accuracy],feed_dict={x: X_test, y: y_test})
            print(str(i + 1) + ": ********* epoch " + str(epoch + 1) + " ********* test accuracy:" + str(c_) + " test loss: " + str(loss_))