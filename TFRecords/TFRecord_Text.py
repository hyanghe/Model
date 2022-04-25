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


def create_dummy_text_dataset(size:int=100):
    text_data = []
    labels = []

    for i in range(size):
        if i%2 == 0:
            text = 'Hey, this is a sample text. We can use many different symbols.'
            label = 0
        else:
            text = 'a point is exactly what the folks think of it; after Gauss.'
            label = 1

        text_data.append(text)
        labels.append(label)
    return text_data, labels

text, labels = create_dummy_text_dataset()
print(text[:5])

def parse_single_text_data(text, label):
    data = {
        'text': _bytes_feature(serialize_array(text)),
        'label': _int64_feature(label)
    }

    out = tf.train.Example(features = tf.train.Features(feature=data))
    return out

def write_text_to_tfr(text_data, labels, filename:str='text'):
    filename = filename + '.tfrecords'
    writer = tf.io.TFRecordWriter(filename)
    count = 0
    for index in range(len(text_data)):
        current_text = text_data[index]
        current_label = labels[index]
        out = parse_single_text_data(text=current_text, label=current_label)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f'Wrote {count} elements to TFRecord')
    return count

# write_text_to_tfr(text_data=text, labels = labels)

def parse_tfr_text_element(element):
    data = {
        'text': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),

    }

    content = tf.io.parse_single_example(element, data)

    text = content['text']
    label = content['label']

    feature = tf.io.parse_tensor(text, out_type=tf.string)
    return (feature, label)

def get_text_dataset(filename):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_tfr_text_element)
    return dataset

text_dataset = get_text_dataset('./text.tfrecords')
for sample in text_dataset.take(2):
    print(sample[0].numpy())
    print(sample[1])