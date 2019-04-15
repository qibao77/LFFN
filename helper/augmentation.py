"""
Paper: "Lightweight Feature Fusion Network for Single Image Super-Resolution"
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

if __name__ == '__main__':
    tf.app.run()
