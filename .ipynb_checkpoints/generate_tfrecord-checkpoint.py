'''
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord_old.py --csv_input=data/train_labels.csv  --output_path=training/train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=training/test.record
  
  Updated:
  python3 generate_tfrecord.py --csv_input=/home/arnold/clouds_detection/obj_detection/data/train_labels.csv --output_path=/home/arnold/clouds_detection/obj_detection/data/train.record --image_dir=/home/arnold/clouds_detection/obj_detection/images/train/

  python3 generate_tfrecord.py --csv_input=/home/arnold/clouds_detection/obj_detection/data/test_labels.csv --output_path=/home/arnold/clouds_detection/obj_detection/data/test.record --image_dir=/home/arnold/clouds_detection/obj_detection/images/test/

After need to grab config ssd_...config and model tar 
tar -xzf ssd_mobilenet_v1_coco_11_06_2017.tar.gz
in config change PATH_TO_BE_CONFIG, num_classes (3 currently), train_config batch size, checkpoit name/path (fine_tune_checkpoint: "ssd_mobilenet_v1_coco_11_06_2017/model.ckpt" we download this from models) and label_map_path: "training/object-detect.pbtxt"
laast update input_path for data/ train and test .record 

NOTE: object-detect.pbtxt needs to be maunally created:
'
item {
     id:1
     name: 'scattered_clouds'
}
item {
     id:2
     name: 'clear'
}
item {
     id:3
     name'overcast'
}
'
COPY dataa, images, downloaded model folder, training and .config to models/object_detection
"cp -r data images training ssd_mobilenet_v1_coco_11_06_2017 /home/arnold/clouds_detection/modelsV2/research/object_detection/legacy"
Finally can train:
From within models/object_detection:
UPDATE : code is now in models research object detection legacy 
python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config

if have tf2.0 wont work need pip install tensorflow-gpu==1.15

want to get avg loss below 1 (if above 2 model iss trash)
'''



from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS


# need to create if for every cat we have
def class_text_to_int(row_label):
    if row_label == 'scattered_clouds':
        return 1
    if row_label == 'clear':
        return 2
    if row_label == 'overcast':
        return 3
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()