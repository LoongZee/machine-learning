#encoding=utf-8

import os
import tensorflow as tf
from PIL import Image


def write_data(filename, X_data_dir, y_datadir):
    x_filelist = sorted(os.listdir(X_data_dir))
    y_filelist = sorted(os.listdir(y_datadir))
    writer = tf.python_io.TFRecordWriter(filename)
    for x_file, y_file in zip(x_filelist, y_filelist):
        if x_file.split('.')[0] == y_file.split('.')[0]:
            img_path = os.path.join(X_data_dir , x_file)
            label_path = os.path.join(y_datadir , y_file)
            img = Image.open(img_path)
            img_x = img.tobytes()
            img = Image.open(label_path)
            img_y = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "img_x": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_x])),
                'img_y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_y]))
            }))
            writer.write(example.SerializeToString())
        else:
            print (x_file, y_file, 'error! image name is different.')
    writer.close()


def read_and_decode(filename, imagewh):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={
    'img_x': tf.FixedLenFeature([], tf.string),'img_y' : tf.FixedLenFeature([], tf.string)})

    img_x = tf.decode_raw(features['img_x'], tf.uint8)
    img_x = tf.reshape(img_x, [1500, 1500, 3])

    img_y = tf.decode_raw(features['img_y'], tf.uint8)
    img_y = tf.reshape(img_y, [1500, 1500, 3])

    #crop image x and y
    crop_img_x, crop_img_y = crop_img(img_x, img_y, imagewh)

    return crop_img_x, crop_img_y

def crop_img(img_x, img_y, imagewh):
    img = tf.concat([img_x, img_y], axis=2)
    crop_img = tf.random_crop(img, [imagewh, imagewh, 6])
    img_x = tf.gather(crop_img, [0,1,2], axis = 2)
    img_y = tf.gather(crop_img, [3,4,5], axis = 2)
    return img_x, img_y


def get_train_batch(data, batch_size, imagewh):
    img, label = read_and_decode(data, imagewh)
    img = tf.cast(tf.reshape(img, [imagewh, imagewh, 3]), dtype=tf.float32)
    label = tf.cast(tf.expand_dims(tf.unstack(tf.reshape(label, [imagewh, imagewh, 3]) / 255, axis = 2)[0], axis= 2), dtype=tf.int64)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=batch_size,
                                                    capacity=2000,
                                                    min_after_dequeue=1000)

    return img_batch, label_batch

def get_test_data(data, batch_size, imagewh):
    img, label = read_and_decode(data, imagewh)
    img = tf.cast(tf.reshape(img, [imagewh, imagewh, 3]),dtype=tf.float32)
    label = tf.cast(tf.expand_dims(tf.unstack(tf.reshape(label, [imagewh, imagewh, 3]) / 255, axis = 2)[0], axis= 2), dtype=tf.int64)
    img_batch, label_batch = tf.train.batch([img, label],
                                                    batch_size=batch_size,
                                                    capacity=2000)
    return img_batch, label_batch


#generate train and test tfrecords
def train_test_generatetfrecords():
    train_image_dir = "./Dataset/UTBuilding/train/input/"
    train_label_dir = "./Dataset/UTBuilding/train/target/"
    write_data("train.tfrecords", train_image_dir, train_label_dir)

    train_image_dir = "./Dataset/UTBuilding/test/input/"
    train_label_dir = "./Dataset/UTBuilding/test/target/"
    write_data("test.tfrecords", train_image_dir, train_label_dir)

train_test_generatetfrecords()






