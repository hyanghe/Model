import tensorflow as tf
import numpy as np
import librosa

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


images_shape = (256, 256, 3)
size = 100
images_combined = np.random.randint(low=0, high = 256, size = (100, *images_shape), dtype=np.int16)
print(images_combined.shape)


def create_dummy_text_dataset_combined(size:int=100):
    text_data = []
    labels = []

    for i in range(size):
        if i%2 == 0:
            text = 'This image shows a wooden bridge. It connects South Darmian with the norther parts of Frenklund.'
            label = 0
        if i%3 == 0:
            text = 'This image shows a sun flower. It\'s leaves are green, the petals are of strong yellow'
            label = 1
        if i%5 == 0:
            text = 'This image shows five children playing in the sandbox. They are laughing'
            label = 2
        if i%7 == 0:
            text = 'This image shows a house on a cliff. The house is painted in red and brown tones.'
            label = 3
        else:
            text = 'This image shows a horse and a zebra. They come from a CycleGAN.'
            label = 4

        text_data.append(text)
        labels.append(label)
    return text_data, labels

text, text_labels = create_dummy_text_dataset_combined()

def create_dummy_audio_dataset():
    files = []
    labels = []

    for i in range(100):
        if i%2 == 0:
            filename = librosa.ex('fishin')
            labels.append(0)
        if i%3 == 0:
            filename = librosa.ex('brahms')
            labels.append(1)
        if i%5 == 0:
            filename = librosa.ex('nutcracker')
            labels.append(2)
        if i%7 == 0:
            filename = librosa.ex('trumpet')
            labels.append(3)
        else:
            filename = librosa.ex('vibeace')
            labels.append(4)

        y, sr = librosa.load(filename)
        files.append([y, sr])
    return files, labels

audio, audio_labels = create_dummy_audio_dataset()

def parse_combined_data(image, text, text_label, audio, audio_label):
    data = {
        'height': _int64_feature(image.shape[0]),
        'width': _int64_feature(image.shape[1]),
        'depth': _int64_feature(image.shape[2]),
        'raw_image': _bytes_feature(serialize_array(image)),
        'text': _bytes_feature(serialize_array(text)),
        'text_label': _int64_feature(text_label),
        'sr': _int64_feature(audio[1]),
        'len': _int64_feature(len(audio[0])),
        'y': _bytes_feature(serialize_array(audio[0])),
        'audio_label': _int64_feature(audio_label)
    }

    out = tf.train.Example(features=tf.train.Features(feature = data))
    return out

def write_combined_data_to_tfr(images, text_data, text_labels, audio_data, audio_labels, filename:str='combined'):
    filename = filename + '.tfrecords'
    writer = tf.io.TFRecordWriter(filename)
    count = 0

    for index in range(len(images)):
        current_image = images[index]
        current_text = text_data[index]
        current_text_label = text_labels[index]
        current_audio = audio_data[index]
        current_audio_label = audio_labels[index]
        out = parse_combined_data(image= current_image,\
                                  text = current_text,\
                                  text_label = current_text_label,\
                                  audio = current_audio,\
                                  audio_label = current_audio_label)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f'Wrote {count} elements to TFRecord')
    return count

# write_combined_data_to_tfr(images=images_combined, \
#                            text_data=text, text_labels=text_labels,\
#                            audio_data=audio, audio_labels=audio_labels)


def parse_combined_tfr_element(element):

    data = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'raw_image': tf.io.FixedLenFeature([], tf.string),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'text': tf.io.FixedLenFeature([], tf.string),
        'text_label': tf.io.FixedLenFeature([], tf.int64),
        'sr': tf.io.FixedLenFeature([], tf.int64),
        'len': tf.io.FixedLenFeature([], tf.int64),
        'y': tf.io.FixedLenFeature([], tf.string),
        'audio_label': tf.io.FixedLenFeature([], tf.int64),
    }

    content = tf.io.parse_single_example(element, data)

    height = content['height']
    width = content['width']
    depth = content['depth']
    raw_image = content['raw_image']

    image_feature = tf.io.parse_tensor(raw_image, out_type=tf.int16)
    image_feature = tf.reshape(image_feature, shape=[height, width, depth])

    sr = content['sr']
    len = content['len']
    y = content['y']
    audio_label = content['audio_label']

    audio_feature = tf.io.parse_tensor(y, out_type = tf.float32)
    audio_feature = tf.reshape(audio_feature, shape=[len])

    text = content['text']
    text_label = content['text_label']

    text_feature = tf.io.parse_tensor(text, out_type=tf.string)

    return image_feature, text_feature, text_label, audio_feature, audio_label

def get_combined_dataset(filename):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_combined_tfr_element)
    return dataset

ds = get_combined_dataset('./combined.tfrecords')
print(next(iter(ds)))