"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Author: Jin Yamanaka
Github: https://github.com/jiny2001/dcscn-image-super-resolution

Create Augmented training images

Put your images under data/[your dataset name]/ and specify [your dataset name] for --dataset.

--augment_level 2-8: will generate flipped / rotated images

"""
import numpy as np
import os
import tensorflow as tf

from helper import args, utilty as util

args.flags.DEFINE_integer("augment_level", 4, "Augmentation level. 4:+LR/UD/LR-UD flipped, 7:+rotated")

FLAGS = args.get()


def main(not_parsed_args):
    if len(not_parsed_args) > 1:
        print("Unknown args:%s" % not_parsed_args)
        exit()

    print("Building x%d augmented data." % FLAGS.augment_level)

    training_filenames = util.get_files_in_directory("/media/data3/ww/sr_data/DIV2K_train_HR/")
    target_dir = "/media/data3/ww/sr_data/DIV2K_train_HR" + ("_%d/" % FLAGS.augment_level)
    util.make_dir(target_dir)

    writer = tf.python_io.TFRecordWriter("DIV2K_org.tfrecords")
    writer2 = tf.python_io.TFRecordWriter("DIV2K_aug.tfrecords")
    for file_path in training_filenames:
        org_image = util.load_image(file_path)
        org_raw = org_image.tobytes()#convert image to bytes

        train_object = tf.train.Example(features=tf.train.Features(feature={
            'org_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[org_raw]))}))
        writer.write(train_object.SerializeToString())

        ud_image = np.flipud(org_image)
        ud_raw = ud_image.tobytes()  # convert image to bytes

        train_object2 = tf.train.Example(features=tf.train.Features(feature={
            'org_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ud_raw]))}))
        writer2.write(train_object2.SerializeToString())
    writer.close()


if __name__ == '__main__':
    tf.app.run()
