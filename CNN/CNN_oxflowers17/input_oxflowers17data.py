#encoding=utf-8


import pandas as pd
from sklearn.model_selection  import train_test_split
import tensorflow as tf
from PIL import Image
from scipy import misc
import numpy as np
import random




def write_data(filename, X_data, y_data):
    writer = tf.python_io.TFRecordWriter(filename)
    for data_path, label in zip(X_data, y_data):
        img_path = '17flowers/jpg/' + data_path
        img = Image.open(img_path)
        img = img.resize((224, 224),Image.ANTIALIAS)
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())  #序列化为字符串
    writer.close()


def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={
    'label': tf.FixedLenFeature([], tf.int64),'img_raw' : tf.FixedLenFeature([], tf.string)})

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # 对图片进行归一化

    # color_deal_tag = random.randint(0, 24)
    # if color_deal_tag!= 24:
    #     distort_color_index = np.arange(4)
    #     np.random.shuffle(distort_color_index)
    #     img = distort_color(img, distort_color_index[0])
    #     img = distort_color(img, distort_color_index[1])
    #     img = distort_color(img, distort_color_index[2])
    #     img = distort_color(img, distort_color_index[3])

    img = distort_flip(img, random.randint(0, 2))
    img = distort_rotate(img, random.randint(0, 1))
    # img = distort_color(img, random.randint(0, 2))

    # Finally, rescale to [-1,1] instead of [0, 1)
    img = tf.subtract(img, 0.5)
    img = tf.multiply(img, 2.0)

    label = tf.cast(features['label'], tf.int64)
    return img, label

def get_train_batch(data, batch_size):
    img, label = read_and_decode(data)
    img = tf.reshape(img, [224, 224, 3])
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=batch_size,
                                                    capacity=2000,
                                                    min_after_dequeue=1000)
    label_batch = tf.one_hot(label_batch, 17, 1, 0)
    return img_batch, label_batch

def get_test_data(data):
    filename_queue = tf.train.string_input_producer([data])
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={
    'label': tf.FixedLenFeature([], tf.int64),'img_raw' : tf.FixedLenFeature([], tf.string)})

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # 对图片进行归一化
    # Finally, rescale to [-1,1] instead of [0, 1)
    img = tf.subtract(img, 0.5)
    img = tf.multiply(img, 2.0)

    label = tf.cast(features['label'], tf.int64)

    X_test, y_test = tf.train.batch([img, label],
                                    batch_size=int(1360 * 0.2),
                                    capacity=2000)
    y_test = tf.one_hot(y_test, 17, 1, 0)
    return X_test, y_test


def Xy_train_test_split():
    file_name = pd.read_table("17flowers/jpg/files.txt", error_bad_lines=False, names=['imagenames'])
    data_path = file_name['imagenames']
    y = [ int(i/80) for i in range(len(data_path))]
    X_train, X_test, y_train, y_test = train_test_split(data_path, y, test_size = 0.2)

    write_data("train.tfrecords",X_train,y_train)
    write_data("test.tfrecords",X_test,y_test)




# 通过调整亮度、对比度、饱和度、色相的顺序随机调整图像的色彩
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    else:
        pass
    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image



def distort_flip(image, flip_ordering=0):
    if flip_ordering == 0:
        image = tf.image.random_flip_left_right(image)
    elif flip_ordering == 1:
        image = tf.image.random_flip_up_down(image)
    else:
        pass
    return image


def random_rotate_image_func(image):
    # 旋转角度范围
    angle = np.random.uniform(low=-30.0, high=30.0)
    return misc.imrotate(image, angle, 'bicubic')

def distort_rotate(image, rotate_ordering=0):
    if rotate_ordering == 0:
        image = tf.py_func(random_rotate_image_func, [image], tf.float32)
    else:
        pass
    return image






