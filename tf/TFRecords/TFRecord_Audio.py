import librosa
import tensorflow as tf

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

audios, labels = create_dummy_audio_dataset()

def parse_single_audio_file(audio, label):
    data = {
        'sr': _int64_feature(audio[1]),
        'len': _int64_feature(len(audio[0])),
        'y': _bytes_feature(serialize_array(audio[0])),
        'label': _int64_feature(label)
    }

    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out

def write_audio_to_tfr(audios, labels, filename:str='audio'):
    filename = filename + '.tfrecords'
    writer = tf.io.TFRecordWriter(filename)
    count = 0
    for index in range(len(audios)):
        current_audio = audios[index]
        current_label = labels[index]

        out = parse_single_audio_file(audio=current_audio, label=current_label)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f'Wrote {count} elements to TRFecord')
    return count

# write_audio_to_tfr(audios, labels)

def parse_tfr_audio_element(element):
    data = {
        'sr': tf.io.FixedLenFeature([], tf.int64),
        'len': tf.io.FixedLenFeature([], tf.int64),
        'y': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    content = tf.io.parse_single_example(element, data)

    sr = content['sr']
    len = content['len']
    y = content['y']
    label = content['label']

    feature = tf.io.parse_tensor(y, out_type = tf.float32)
    feature = tf.reshape(feature, shape=[len])

    return (feature, label)

def get_audio_dataset(filename):
    dataset = tf.data.TFRecordDataset(filename)

    dataset = dataset.map(parse_tfr_audio_element)
    return dataset

dataset_audio = get_audio_dataset('./audio.tfrecords')
for sample in dataset_audio.take(1):
    print(sample[0].shape)
    print(sample[1].shape)