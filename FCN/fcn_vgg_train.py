
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from fcn_vgg import fcn_vgg
import input_tfrecords
import TensorflowUtils as utils

"""
Configuration settings
"""
# Learning params
learning_rate = 0.001
# Network params
dropout_rate = 0.5
num_classes = 2
display_step = 10
# Get the number of training/validation steps per epoch
num_epochs = 1000
batch_size = 8
data_size = 137
train_batches_per_epoch = np.floor(data_size / batch_size).astype(np.int16)


# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "finetune_vggnet/filewriter"
checkpoint_path = "finetune_vggnet/checkpoint"

# Create parent path if it doesn't exist
if not os.path.isdir(filewriter_path):
    os.makedirs(filewriter_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)


# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.int64, [None, 224, 224, 1])
keep_prob = tf.placeholder(tf.float32)
# Initialize model
# model bal: fcn32 fcn16 fcn8
model = fcn_vgg(x, keep_prob, num_classes, model = 'fcn16')
# Link variable to model output
logits = model.logits
# Op for calculating the loss
with tf.name_scope("cross_entropy"):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = tf.squeeze(y, [3])))
# Train optimizer
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# Add the loss to summary
summary_loss = tf.summary.scalar('cross_entropy', loss)
# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    output = tf.argmax(logits, 3)
    correct_pred = tf.equal(output, tf.squeeze(y, [3]))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Add the accuracy to the summary
summary_accuracy = tf.summary.scalar('accuracy', accuracy)

# Initialize the FileWriter
train_writer = tf.summary.FileWriter(filewriter_path + '/train')
test_writer = tf.summary.FileWriter(filewriter_path + '/test')
# Merge all summaries together
model_summary = tf.summary.merge([summary_accuracy , summary_loss])

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Start Tensorflow session
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    # Load the pretrained weights into the non-trainable layer

    print("{} Start training...".format(datetime.now()))

    train_batch, train_label_batch = input_tfrecords.get_train_batch("train.tfrecords", batch_size, imagewh = 224)
    test_batch, test_label_batch = input_tfrecords.get_test_data("test.tfrecords", 10, imagewh = 224)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Loop over number of epochs
    for epoch in range(num_epochs):
        print ("{} Epoch number: {}".format(datetime.now(), epoch + 1))
        step = 1
        while step <= train_batches_per_epoch:
            # Get a batch of images and labels
            X_data, y_data = sess.run([train_batch, train_label_batch])
            # And run the training op
            _train_op, _loss, _acc, _s = sess.run([optimizer, loss, accuracy, model_summary],
                                                    feed_dict={x: X_data, y: y_data, keep_prob: dropout_rate})
            train_writer.add_summary(_s, epoch * train_batches_per_epoch + step)
            print('train_step:' + str(step) + ',loss:' + str(_loss) + ',acc:' + str(_acc))
            step += 1

        if (epoch + 1) % display_step == 0:
            #test
            X_data, y_data = sess.run([test_batch, test_label_batch])
            _loss, _acc, _out, _s = sess.run([loss, accuracy, output, model_summary],
                                       feed_dict={x: X_data, y: y_data, keep_prob: 1.})

            test_writer.add_summary(_s, epoch * train_batches_per_epoch + step)
            print('test_step:' + str(step) + ',loss:' + str(_loss) + ',acc:' + str(_acc))
            for i, img, label, pre in zip(range(X_data.shape[0]), X_data, y_data, _out):
                img = img.astype(np.uint8)
                label = np.concatenate([label * 255, np.zeros([224, 224, 2])], 2).astype(np.uint8)
                pre = np.concatenate([np.zeros([224, 224, 1]), np.expand_dims(pre * 255, 2), np.zeros([224, 224, 1])], 2).astype(np.uint8)
                utils.save_image(img, 'output/img', str(i + 1))
                utils.save_image(label, 'output/label', str(i + 1))
                utils.save_image(pre, 'output/pre', str(i + 1))

            # save checkpoint of the model
            # print("{} Saving checkpoint of model...".format(datetime.now()))
            # checkpoint_name = os.path.join(checkpoint_path, 'model_save' + str(epoch+1) + '.ckpt')
            # save_path = saver.save(sess, checkpoint_name)
            # print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

    coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)

    test_writer.close()
    train_writer.close()
