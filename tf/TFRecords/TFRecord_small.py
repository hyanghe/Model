import numpy as np
import tensorflow as tf


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array

image_small_shape = (250, 250, 3)
number_of_images_small = 100
images_small = np.random.randint(low=0, high=256, size=(number_of_images_small, *image_small_shape), dtype=np.int16)
print(images_small.shape)

labels_small = np.random.randint(low=0, high=5, size=(number_of_images_small, 1))
print(labels_small.shape)
print(labels_small[:10])

def parse_single_image(image, label):
    data = {
        'height': _int64_feature(image.shape[0]),
        'width': _int64_feature(image.shape[1]),
        'depth': _int64_feature(image.shape[2]),
        'raw_image': _bytes_feature(serialize_array(image)),
        'label': _int64_feature(label)
    }

    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out

def write_images_to_tfr_short(images, labels, filename:str="images"):
    filename= filename + ".tfrecords"
    writer = tf.io.TFRecordWriter(filename)
    count = 0

    for index in range(len(images)):
        current_image = images[index]
        current_label = labels[index]
        out = parse_single_image(image = current_image, label = current_label)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f'Wrote {count} elements to TFRecord')
    return count

# count = write_images_to_tfr_short(images_small, labels_small, filename="small_images")

def parse_tfr_element(element):
    data = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'raw_image': tf.io.FixedLenFeature([], tf.string),
        'depth': tf.io.FixedLenFeature([], tf.int64),

    }

    content = tf.io.parse_single_example(element, data)

    height = content['height']
    width = content['width']
    depth = content['depth']
    label = content['label']
    raw_image = content['raw_image']

    feature = tf.io.parse_tensor(raw_image, out_type=tf.int16)
    feature = tf.reshape(feature, shape=[height, width, depth])
    return (feature, label)

def get_dataset_small(filename):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_tfr_element)
    return dataset

dataset = get_dataset_small("./small_images.tfrecords")
for sample in dataset.take(1):
    print(sample[0].shape)
    print(sample[1].shape)