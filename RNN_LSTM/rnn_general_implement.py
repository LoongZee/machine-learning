# -*- coding: utf-8 -*-
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
batch_size= 512
state_size= 128           # cell size
keep_prob = 0.8           # dropout
num_classes = 5
training_epochs = 2000
learning_rate = 0.001
total_batch = int(len(X_train) / batch_size)
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



def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()   # 重设计算图


'''使用list的方式,static_rnn'''
def build_basic_rnn_graph_with_list(
    state_size = state_size,
    num_classes = num_classes,
    batch_size = batch_size,
    num_steps = num_steps,
    num_layers = 3,
    learning_rate = learning_rate):

    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='x')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    x_one_hot = tf.one_hot(x, num_classes)   # (batch_size, num_steps, num_classes)
    '''这里按第二维拆开num_steps*(batch_size, num_classes)'''
    rnn_inputs = [tf.squeeze(i,squeeze_dims=[1]) for i in tf.split(x_one_hot, num_steps, 1)]
    '''dropout rnn inputs'''
    rnn_inputs = [tf.nn.dropout(rnn_input, keep_prob) for rnn_input in rnn_inputs]
    
    cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    init_state = cell.zero_state(batch_size, tf.float32)
    '''使用static_rnn方式'''
    rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell=cell, inputs=rnn_inputs, 
                                                        initial_state=init_state)
    '''dropout rnn outputs'''
    rnn_outputs = [tf.nn.dropout(rnn_output, keep_prob) for rnn_output in rnn_outputs]
    #rnn_outputs, final_state = tf.nn.rnn(cell, rnn_inputs, initial_state=init_state) # tensorflow 1.0的方式
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]

    y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(y, num_steps, 1)]

    #loss_weights = [tf.ones([batch_size]) for i in range(num_steps)]
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_as_list, 
                                                  logits=logits)
    #losses = tf.nn.seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)  # tensorflow 1.0的方式
    total_loss = tf.reduce_mean(losses)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = x,
        y = y,
        keep_prob = keep_prob,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step
    )


'''使用dynamic_rnn方式
   - 之前我们自己实现的cell和static_rnn的例子都是将得到的tensor使用list存起来，这种方式构建计算图时很慢
   - dynamic可以在运行时构建计算图
'''
def build_multilayer_lstm_graph_with_dynamic_rnn(
    state_size = state_size,
    num_classes = num_classes,
    batch_size = batch_size,
    num_steps = num_steps,
    num_layers = 3,
    learning_rate = learning_rate
    ):
    reset_graph()
    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='x')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='y')
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    embeddings = tf.get_variable(name='embedding_matrix', shape=[num_classes, state_size])
    '''这里的输入是三维的[batch_size, num_steps, state_size]
        - embedding_lookup(params, ids)函数是在params中查找ids的表示， 和在matrix中用array索引类似,
          这里是在二维embeddings中找二维的ids, ids每一行中的一个数对应embeddings中的一行，所以最后是[batch_size, num_steps, state_size]
    '''
    rnn_inputs = tf.nn.embedding_lookup(params=embeddings, ids=x)
    
    cell = tf.nn.rnn_cell.LSTMCell(num_units=state_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell]*num_layers, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=keep_prob)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    '''使用dynamic_rnn方式'''
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=rnn_inputs, 
                                                 initial_state=init_state)    
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])   # 转成二维的矩阵
    y_reshape = tf.reshape(y, [-1])
    logits = tf.matmul(rnn_outputs, W) + b                    # 进行矩阵运算
    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshape))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    
    return dict(x = x,
                y = y,
                keep_prob = keep_prob,
                init_state = init_state,
                final_state = final_state,
                total_loss = total_loss,
                train_step = train_step)

'''使用scan实现dynamic_rnn的效果'''
def build_multilayer_lstm_graph_with_scan(
    state_size = state_size,
    num_classes = num_classes,
    batch_size = batch_size,
    num_steps = num_steps,
    num_layers = 3,
    learning_rate = learning_rate
    ):
    reset_graph()
    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='x')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='y')
    embeddings = tf.get_variable(name='embedding_matrix', shape=[num_classes, state_size])
    '''这里的输入是三维的[batch_size, num_steps, state_size]
    '''
    rnn_inputs = tf.nn.embedding_lookup(params=embeddings, ids=x)
    '''构建多层的cell, 先构建一个cell, 然后使用MultiRNNCell函数构建即可'''
    cell = tf.nn.rnn_cell.LSTMCell(num_units=state_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell]*num_layers, state_is_tuple=True)  
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    '''使用tf.scan方式
       - tf.transpose(rnn_inputs, [1,0,2])  是将rnn_inputs的第一个和第二个维度调换，即[num_steps,batch_size, state_size],
           在dynamic_rnn函数有个time_major参数，就是指定num_steps是否在第一个维度上，默认是false的,即不在第一维
       - tf.scan会将elems按照第一维拆开，所以一次就是一个step的数据（和我们static_rnn的例子类似）
       - a的结构和initializer的结构一致，所以a[1]就是对应的state，cell需要传入x和state计算
       - 每次迭代cell返回的是一个rnn_output(batch_size,state_size)和对应的state,num_steps之后的rnn_outputs的shape就是(num_steps, batch_size, state_size)
       - 每次输入的x都会得到的state(final_states)，我们只要的最后的final_state
    '''
    def testfn(a, x):
        return cell(x, a[1])
    rnn_outputs, final_states = tf.scan(fn=testfn, elems=tf.transpose(rnn_inputs, [1,0,2]),
                                        initializer=(tf.zeros([batch_size,state_size]),init_state)
                                        )
    '''或者使用lambda的方式'''
    #rnn_outputs, final_states = tf.scan(lambda a,x: cell(x, a[1]), tf.transpose(rnn_inputs, [1,0,2]),
                                        #initializer=(tf.zeros([batch_size, state_size]),init_state))
    final_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
        tf.squeeze(tf.slice(c, [num_steps-1,0,0], [1,batch_size,state_size])),
        tf.squeeze(tf.slice(h, [num_steps-1,0,0], [1,batch_size,state_size]))) for c, h in final_states])

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshape = tf.reshape(y, [-1])
    logits = tf.matmul(rnn_outputs, W) + b
    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshape))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    
    return dict(x = x,
                y = y,
                init_state = init_state,
                final_state = final_state,
                total_loss = total_loss,
                train_step = train_step)


'''最终的整合模型，
   - 普通RNN，GRU，LSTM
   - dropout
   - BN
'''
def build_final_graph(
    cell_type = None,
    state_size = state_size,
    num_classes = num_classes,
    batch_size = batch_size,
    num_steps = num_steps,
    num_layers = 3,
    build_with_dropout = False,
    learning_rate = learning_rate):
    
    reset_graph()
    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='x')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
    if cell_type == 'GRU':
        cell = tf.nn.rnn_cell.GRUCell(state_size)
    elif cell_type == 'LSTM':
        cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    elif cell_type == 'LN_LSTM':
        cell = LayerNormalizedLSTMCell(state_size)
    else:
        cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
    if build_with_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
        
    init_state = cell.zero_state(batch_size, tf.float32)
    '''dynamic_rnn'''
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])
    logits = tf.matmul(rnn_outputs, W) + b

    predictions = tf.nn.softmax(logits)

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = x,
        y = y,
        keep_prob = keep_prob,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step,
        preds = predictions,
        saver = tf.train.Saver()
    )    

                


'''1、构建图和训练'''
g = build_basic_rnn_graph_with_list()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    training_losses = []

    for epoch in range(training_epochs):
        for i in range(total_batch):
            # g['batch_size'] = 512
            X_, XLen_, Y_ = get_random_block_from_data(batch_size)
            feed_dict = {g['x']: X_, g['seq']: XLen_, g['y']: Y_, g['keep_prob']: keep_prob}
            training_loss_, acc_, output_, logits_ = sess.run([g['total_loss'],g['preds'],g['output_reshaped'],g['logits']],feed_dict=feed_dict)
            print(str(i + 1) + ": accuracy:" + str(acc_) + " loss: " + str(training_loss_))
        feed_dict = {g['x']: X_test, g['seq']: XLen_test, g['y']: y_test, g['keep_prob']: 1.0}
        loss, c_ = sess.run([g['total_loss'],g['preds']],feed_dict=feed_dict)
        print(str(i + 1) + ": ********* epoch " + str(epoch + 1) + " ********* test accuracy:" + str(c_) + " test loss: " + str(loss))