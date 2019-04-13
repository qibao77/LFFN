"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Ver: 2

functions for sharing arguments and their default values
"""

import sys

import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# Model (network) Parameters
flags.DEFINE_integer("scale", 2, "Scale factor for Super Resolution (should be 2 or more)")
flags.DEFINE_integer("layers", 4, "Number of spindle modules")
flags.DEFINE_boolean("depth_wise_convolution", True, "Use depth wise convolution in spindle block")
flags.DEFINE_integer("self_ensemble", 1, "Number of using self ensemble method. [1 - 8], 1 means no self ensemble")

# Training Parameters
flags.DEFINE_float("clipping_norm", 5, "Norm for gradient clipping. If it's <= 0 we don't use gradient clipping.")
flags.DEFINE_string("initializer", "he", "Initializer for weights can be [uniform, stddev, xavier, he, identity, zero]")
flags.DEFINE_float("weight_dev", 0.01, "Initial weight stddev (won't be used when you use he or xavier initializer)")
flags.DEFINE_string("optimizer", "adam", "Optimizer can be [gd, momentum, adadelta, adagrad, adam, rmsprop]")
flags.DEFINE_float("beta1", 0.9, "Beta1 for adam optimizer")
flags.DEFINE_float("beta2", 0.999, "Beta2 for adam optimizer")
flags.DEFINE_float("momentum", 0.9, "Momentum for momentum optimizer and rmsprop optimizer")
flags.DEFINE_integer("batch_num", 16, "Number of mini-batch images for training")
flags.DEFINE_integer("batch_image_size", 32, "Image size for mini-batch")
flags.DEFINE_boolean("no_augment", False, "do not use data augmentation")
flags.DEFINE_integer("training_images", 30000, "Number of training on each epoch")
flags.DEFINE_integer("n_threads", 8, "number of threads for data loading")
flags.DEFINE_boolean("cpu", False, "use cpu only")
flags.DEFINE_integer("test_every", 1000, "do test per every N batches")
flags.DEFINE_string("test_dir_train", "/media/data1/ww/sr_data/benchmark/Set5/HR", "Test dataset dir in training stage")

# Learning Rate Control for Training
flags.DEFINE_float("initial_lr", 0.0008, "Initial learning rate")
flags.DEFINE_float("lr_decay", 0.5, "Learning rate decay rate")
flags.DEFINE_integer("lr_decay_epoch", 15, "After this epochs are completed, learning rate will be decayed by lr_decay.")
flags.DEFINE_float("end_lr", 0.00004, "Training end learning rate. If the current learning rate gets lower than this value, then training will be finished.")

# Dataset or Others
flags.DEFINE_string("train_dir", "/media/data1/ww/sr_data", "Training dataset dir.")
flags.DEFINE_string("data_train", "DATA291_aug4", "Training dataset.")
flags.DEFINE_string("scale_bin", "X4/bin", "bin filename.")
flags.DEFINE_string("ext", "X4/bin", "dataset file extension.")
flags.DEFINE_boolean("test_only", True, "set this option to test the model")
flags.DEFINE_string("test_dir", "/media/data1/ww/sr_data/test/", "Test dataset dir.")
flags.DEFINE_string("test_dataset", "all", "Directory for test dataset [set5, set14, bsd100, urban100, all]")
flags.DEFINE_integer("tests", 1, "Number of training sets")
flags.DEFINE_string("data_range", "1-1450/1-5", "train/test data range")
flags.DEFINE_boolean("do_benchmark", True, "Evaluate the performance for set5, set14 and bsd100 after the training.")


# Image Processing
flags.DEFINE_float("max_value", 255, "For normalize image pixel value")
flags.DEFINE_integer("channels", 3, "Number of image channels used. Now it should be 3. using rgb image.")
flags.DEFINE_integer("psnr_calc_border_size", -1, "Cropping border size for calculating PSNR. if < 0, use 2 + scale for default.")

# Environment (all directory name should not contain '/' after )
flags.DEFINE_string("checkpoint_dir", "/home/ww/programme/LFFN/models/LFFN_x2_B4M4_depth_div2k/models", "Directory for checkpoints")
flags.DEFINE_string("graph_dir", "/home/ww/programme/LFFN/models/LFFN_x2_B4M4_depth_div2k/graphs", "Directory for graphs")
flags.DEFINE_string("output_dir", "/home/ww/programme/LFFN/models/LFFN_x2_B4M4_depth_div2k", "Directory for output test images")
flags.DEFINE_string("tf_log_dir", "/home/ww/programme/LFFN/models/LFFN_x2_B4M4_depth_div2k/tf_log", "Directory for tensorboard log")
flags.DEFINE_string("log_filename", "/home/ww/programme/LFFN/models/LFFN_x2_B4M4_depth_div2k/log.txt", "log filename")
flags.DEFINE_string("model_name", "x2_B4M4_depth_div2k", "model name for save files and tensorboard log")
flags.DEFINE_string("load_model_name", "lffn_model", "Filename of model loading before start [filename or 'default']")

# Debugging or Logging
flags.DEFINE_boolean("initialize_tf_log", True, "Clear all tensorboard log before start")
flags.DEFINE_boolean("save_loss", True, "Save loss")
flags.DEFINE_boolean("save_weights", True, "Save weights and biases")
flags.DEFINE_boolean("save_images", False, "Save CNN weights as images")
flags.DEFINE_boolean("save_meta_data", True, "")


def get():
	print("Python Interpreter version:%s" % sys.version[:3])
	print("tensorflow version:%s" % tf.__version__)
	print("numpy version:%s" % np.__version__)

	# check which library you are using
	# np.show_config()
	return FLAGS
