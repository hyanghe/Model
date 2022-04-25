from tqdm import tqdm
import tensorflow as tf
import numpy as np
import glob

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

def parse_tfr_element(element):
    data = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'raw_image': tf.io.FixedLenFeature([], tf.string)
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

image_large_shape = (400, 750, 3)
number_of_images_large = 500
images_large = np.random.randint(low=0, high=256, size=(number_of_images_large, *(image_large_shape)), dtype=np.int16)
labels_large = np.random.randint(low=0, high=5, size=(number_of_images_large, 1))

def write_images_to_tfr_long(images, labels, filename:str='large_images', max_files:int=10, out_dir:str="./content/"):
    splits = (len(images)//max_files) + 1
    if len(images)%max_files == 0:
        splits -= 1
    print(f'\nUsing {splits} shard(s) for {len(images)} files, with up to {max_files} samples per shard')
    file_count = 0

    for i in tqdm(range(splits)):
        current_shard_name = f"{out_dir}{i+1}_{splits}{filename}.tfrecords"
        writer = tf.io.TFRecordWriter(current_shard_name)

        current_shard_count = 0
        while current_shard_count < max_files:
            index = 1*max_files+current_shard_count
            if index == len(images):
                break
            current_image = images[index]
            current_label = labels[index]

            out = parse_single_image(image=current_image, label=current_label)

            writer.write(out.SerializeToString())
            current_shard_count +=1
            file_count += 1
        writer.close()
    print(f"\nwrote {file_count} elements to TFRecord")
    return file_count

# write_images_to_tfr_long(images_large, labels_large, max_files=30)

def get_dataset_large(tfr_dir:str='./content/', pattern:str='*large_images.tfrecords'):
    files = glob.glob(tfr_dir+pattern, recursive=False)
    dataset = tf.data.TFRecordDataset(files)

    dataset = dataset.map(parse_tfr_element)
    return dataset

dataset_large = get_dataset_large()
for sample in dataset_large.take(1):
    print(sample[0].shape)
    print(sample[1].shape)
