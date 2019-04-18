import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet import AlexNet
import input_oxflowers17data



# Learning params
learning_rate = 0.01
num_epochs = 10000
batch_size = 128
train_size = int(1360*0.8)
# Network params
dropout_rate = 0.5
num_classes = 17
train_layers = ['fc8', 'fc7']

# How often we want to write the tf.summary data to disk
display_step = 1

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "finetune_alexnet/filewriter"
checkpoint_path = "finetune_alexnet/checkpoint"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)


# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [None, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = score, labels = y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))
  
    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary  
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary  
for var in var_list:
    tf.summary.histogram(var.name, var)
  
# Add the loss to summary
summary_loss = tf.summary.scalar('cross_entropy', loss)
  

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  
# Add the accuracy to the summary
summary_accuracy = tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()
merged_summary1 = tf.summary.merge([summary_accuracy , summary_loss])

# Initialize the FileWriter
train1_writer = tf.summary.FileWriter(filewriter_path + '/train1')
train2_writer = tf.summary.FileWriter(filewriter_path + '/train2')
test_writer = tf.summary.FileWriter(filewriter_path + '/test')

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_size / batch_size).astype(np.int16)
# split and read data
#input_oxflowers17data.Xy_train_test_split()
# Start Tensorflow session
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                    filewriter_path))

    img_batch, label_batch = input_oxflowers17data.get_train_batch("train.tfrecords", batch_size)
    X_test, y_test = input_oxflowers17data.get_test_data("test.tfrecords")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    _X_test, _y_test = sess.run([X_test, y_test])


    # Loop over number of epochs
    for epoch in range(num_epochs):

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        _summary, acc = sess.run([merged_summary1, accuracy], feed_dict={x: _X_test,
                                            y: _y_test,
                                            keep_prob: 1.})
        test_writer.add_summary(_summary, (epoch) * train_batches_per_epoch)
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), acc))

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        step = 1
        
        while step <= train_batches_per_epoch:
            
            # Get a batch of images and labels
            X_data, y_data = sess.run([img_batch, label_batch])
            
            # And run the training op
            _train_op, _loss, _acc, _s1  = sess.run([train_op, loss, accuracy, merged_summary], feed_dict={x: X_data, y: y_data,keep_prob: dropout_rate})
            train1_writer.add_summary(_s1, epoch * train_batches_per_epoch + step)
            print('step:' + str(step) + ',loss:' + str(_loss) + ',acc:' + str(_acc))

            #Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                _s2 = sess.run(merged_summary1, feed_dict={x: X_data,
                                                        y: y_data,
                                                        keep_prob: 1.})
                train2_writer.add_summary(_s2, epoch*train_batches_per_epoch + step)

            step += 1


    if (epoch + 1) % 50 == 0:
        # save checkpoint of the model
        print("{} Saving checkpoint of model...".format(datetime.now()))
        checkpoint_name = os.path.join(checkpoint_path, 'model_save' + str(epoch+1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)
        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

    coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)

    test_writer.close()
    train1_writer.close()
    train2_writer.close()
# cd F:\deeplearning\oxflowers17\AlexNet_oxflowers17\finetune_alexnet\filewriter
# tensorboard --logdir=run1:"test",run2:"train1",run3:"train2"

        
