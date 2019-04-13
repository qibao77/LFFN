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

    # print("Building x%d augmented data." % FLAGS.augment_level)
    #
    # training_filenames = util.get_files_in_directory("/media/data1/ww/sr_data/291/291_HR")
    # target_dir = "/media/data3/ww/sr_data/DIV2K_train_HR" + ("_%d/" % FLAGS.augment_level)
    # util.make_dir(target_dir)
    #
    # writer = tf.python_io.TFRecordWriter("DIV2K_org.tfrecords")
    # writer2 = tf.python_io.TFRecordWriter("DIV2K_aug.tfrecords")
    # for file_path in training_filenames:
    #     org_image = util.load_image(file_path)
    #     org_raw = org_image.tobytes()#convert image to bytes
    #
    #     train_object = tf.train.Example(features=tf.train.Features(feature={
    #         'org_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[org_raw]))}))
    #     writer.write(train_object.SerializeToString())
    #
    #     ud_image = np.flipud(org_image)
    #     ud_raw = ud_image.tobytes()  # convert image to bytes
    #
    #     train_object2 = tf.train.Example(features=tf.train.Features(feature={
    #         'org_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ud_raw]))}))
    #     writer2.write(train_object2.SerializeToString())
    # writer.close()
    #
    # #generate data
    # print("Building x%d augmented data." % FLAGS.augment_level)

    training_filenames = util.get_files_in_directory("/media/data1/ww/sr_data/DIV2K_aug2/DIV2K_train_HR")
    # target_dir_x2 = "/media/data1/ww/sr_data/DIV2K_aug2/291_LR_bicubic_X2/291_LR_bicubic/X2"
    target_dir_x3 = "/media/data1/ww/sr_data/DIV2K_aug2/DIV2K_train_LR_bicubic_X3/DIV2K_train_LR_bicubic/X3"
    target_dir_x4 = "/media/data1/ww/sr_data/DIV2K_aug2/DIV2K_train_LR_bicubic_X4/DIV2K_train_LR_bicubic/X4"
    # util.make_dir(target_dir_x2)
    util.make_dir(target_dir_x3)
    util.make_dir(target_dir_x4)
    for file_path in training_filenames:
        org_image = util.load_image(file_path)
        filename = os.path.basename(file_path)
        filename, extension = os.path.splitext(filename)
        # new_filename_x2 = target_dir_x2 + '/' +filename + 'x{}'.format(2)
        new_filename_x3 = target_dir_x3 + '/' + filename + 'x{}'.format(3)
        new_filename_x4 = target_dir_x4 + '/' + filename + 'x{}'.format(4)

        # bicubic_image_x2 = util.resize_image_by_pil(org_image, 1 / 2)
        bicubic_image_x3 = util.resize_image_by_pil(org_image, 1 / 3)
        bicubic_image_x4 = util.resize_image_by_pil(org_image, 1 / 4)
        # util.save_image(new_filename_x2 + extension, bicubic_image_x2)
        util.save_image(new_filename_x3 + extension, bicubic_image_x3)
        util.save_image(new_filename_x4 + extension, bicubic_image_x4)


    # for file_path in training_filenames:
    # 	org_image = util.load_image(file_path)
    #
    # 	filename = os.path.basename(file_path)
    # 	filename, extension = os.path.splitext(filename)
    #
    # 	new_filename = target_dir + filename
    # 	util.save_image(new_filename + extension, org_image)
    #
    # 	if FLAGS.augment_level >= 2:
    # 		ud_image = np.flipud(org_image)
    # 		util.save_image(new_filename + "_v" + extension, ud_image)
    # 	if FLAGS.augment_level >= 3:
    # 		lr_image = np.fliplr(org_image)
    # 		util.save_image(new_filename + "_h" + extension, lr_image)
    # 	if FLAGS.augment_level >= 4:
    # 		lr_image = np.fliplr(org_image)
    # 		lrud_image = np.flipud(lr_image)
    # 		util.save_image(new_filename + "_hv" + extension, lrud_image)
    #
    # 	if FLAGS.augment_level >= 5:
    # 		rotated_image1 = np.rot90(org_image)
    # 		util.save_image(new_filename + "_r1" + extension, rotated_image1)
    # 	if FLAGS.augment_level >= 6:
    # 		rotated_image2 = np.rot90(org_image, -1)
    # 		util.save_image(new_filename + "_r2" + extension, rotated_image2)
    #
    # 	if FLAGS.augment_level >= 7:
    # 		rotated_image1 = np.rot90(org_image)
    # 		ud_image = np.flipud(rotated_image1)
    # 		util.save_image(new_filename + "_r1_v" + extension, ud_image)
    # 	if FLAGS.augment_level >= 8:
    # 		rotated_image2 = np.rot90(org_image, -1)
    # 		ud_image = np.flipud(rotated_image2)
    # 		util.save_image(new_filename + "_r2_v" + extension, ud_image)


if __name__ == '__main__':
    tf.app.run()
