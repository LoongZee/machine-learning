# encoding=utf-8

import time
import numpy as np
from input_sentimentdata import get_sentimentdata
from sklearn.model_selection import train_test_split
from LayerNormalizedLSTMCell import LayerNormalizedLSTMCell
import tensorflow as tf

# data head=[sentence, sentiment values, vec, length, label]
X, XLen, y = get_sentimentdata()
X_train, X_test, XLen_train, XLen_test, y_train, y_test = train_test_split(X, XLen, y, test_size=0.2)
print ('train size %d ,test size %d' % (len(X_train), len(X_test)))

# set para
num_steps= X.shape[1]      #56
batch_size= 64
state_size= 50           # cell size
keep_prob = 0.7           # dropout
num_classes = 5
training_epochs = 2000
total_batch = int(len(X_train) / batch_size)
batch_offset = 0
epochs_completed = 0

# parameter init method
def xavier_init(fan_in, fan_out, constant=-1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

def cell_init():
    cell = LayerNormalizedLSTMCell(num_units=state_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
    return cell
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

# get batch data
def get_test_batchdata(batch_size):
    start = 0
    num = int(len(X_test) / batch_size)
    test_data = []
    for i in range(num):
        test_data.append([X_test[start:start+batch_size], XLen_test[start:start+batch_size], y_test[start:start+batch_size]])
        start = start + batch_size
    return test_data

def build_final_graph(
        state_size=state_size,
        num_classes=num_classes,
        batch_size=batch_size,
        num_steps=num_steps,
        num_layers=3):
    x = tf.placeholder(tf.float32, [batch_size, num_steps, 50], name='x')
    y = tf.placeholder(tf.int64, [batch_size], name='y')
    seq = tf.placeholder(tf.int64, [batch_size], name='x')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # [batch_size, num_steps, state_size]
    # embeddings = tf.get_variable(name='embedding_matrix', shape=[num_classes, state_size])
    # rnn_inputs = tf.nn.embedding_lookup(params=embeddings, ids=x)

    # cell = LayerNormalizedLSTMCell(num_units=state_size, state_is_tuple=True)
    layers = [cell_init() for _ in range(num_layers)]
    cell = tf.nn.rnn_cell.MultiRNNCell(cells=layers, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    # cell = tf.nn.rnn_cell.MultiRNNCell(cells=layers, state_is_tuple=True)
    # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    # cell = cell_init()
    init_state = cell.zero_state(batch_size, tf.float32)
    # '''dynamic_rnn'''
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                                 inputs=x,
                                                 sequence_length=seq,
                                                 initial_state = init_state)

    # kernel_init = tf.random_uniform_initializer(-0.1, 0.1)
    # cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=state_size)
    # cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=state_size)
    # initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
    # initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', initializer=xavier_init(state_size, num_classes))
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    output_flattened = tf.reshape(rnn_outputs, [-1, state_size])
    output_logits = tf.matmul(output_flattened, W) + b
    output_reshaped = tf.reshape(output_logits,[-1,num_steps,num_classes])
    #logits = tf.gather(tf.transpose(output_reshaped, [1, 0, 2]), num_steps - 1)
    logits = []
    for i in range(batch_size):
        logits.append(tf.expand_dims(tf.gather(tf.gather(output_reshaped, i),seq[i] - 1),0))
    logits = tf.concat(logits,0)
    logits_softmax = tf.nn.softmax(logits)

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
    train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(total_loss)

    correct_prediction = tf.equal(tf.argmax(logits_softmax, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return dict(
        x=x,
        seq=seq,
        y=y,
        keep_prob=keep_prob,
        logits = logits,
        batch_size = batch_size,
        total_loss=total_loss,
        train_step=train_step,
        preds=accuracy,
        saver=tf.train.Saver()
    )



start_time = time.time()
g = build_final_graph(state_size=state_size,
                     num_classes=num_classes,
                     batch_size=batch_size,
                     num_steps=num_steps,
                     num_layers=3)
print('gragh cost time:', time.time()-start_time)

with tf.Session() as sess:
    filewriter_path = "output/filewriter"
    train_writer = tf.summary.FileWriter(filewriter_path + '/train')
    test_writer = tf.summary.FileWriter(filewriter_path + '/test')

    sess.run(tf.global_variables_initializer())
    test_data = get_test_batchdata(batch_size)

    for epoch in range(training_epochs):
        train_acc = 0
        train_loss = 0
        for i in range(total_batch):
            X_, XLen_, Y_ = get_random_block_from_data(batch_size)
            feed_dict = {g['x']: X_, g['seq']: XLen_, g['y']: Y_, g['keep_prob']: keep_prob}
            _, training_loss_, acc_ = sess.run([g['train_step'],g['total_loss'],g['preds']],feed_dict=feed_dict)
            train_acc = train_acc + acc_
            train_loss = train_loss + training_loss_
        print("accuracy:" + str(train_acc/total_batch) + " loss: " + str(train_loss/total_batch))
        summary1 = tf.Summary(value=[
            tf.Summary.Value(tag="train_acc", simple_value=train_acc/total_batch),
            tf.Summary.Value(tag="train_loss", simple_value=train_loss/total_batch)
        ])
        train_writer.add_summary(summary1, epoch+1)
        test_acc = 0
        test_loss = 0
        for i in range(len(test_data)):
            feed_dict = {g['x']: test_data[i][0], g['seq']: test_data[i][1], g['y']: test_data[i][2], g['keep_prob']: 1.0}
            loss, c_ = sess.run([g['total_loss'],g['preds']],feed_dict=feed_dict)
            test_acc = test_acc + c_
            test_loss = test_loss + loss
        print(str(i + 1) + ": ********* epoch " + str(epoch + 1) + " ********* test accuracy:" + str(test_acc/len(test_data)) + " test loss: " + str(test_loss/len(test_data)))
        summary2 = tf.Summary(value=[
            tf.Summary.Value(tag="test_acc", simple_value=test_acc/len(test_data)),
            tf.Summary.Value(tag="test_loss", simple_value=test_loss/len(test_data))
        ])
        test_writer.add_summary(summary2, epoch+1)

# tensorboard --logdir=run1:"test",run2:"train"















