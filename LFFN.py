"""
Paper: "Lightweight Feature Fusion Network for Single Image Super-Resolution"
"""

import logging
import math
import os
import time
import numpy as np
import tensorflow as tf

from helper import tf_graph, utilty as util, compute_psnr_ssim as eva

BICUBIC_METHOD_STRING = "bicubic"


class SuperResolution(tf_graph.TensorflowGraph):
    def __init__(self, flags, model_name=""):

        super().__init__(flags)

        # Model Parameters
        self.scale = flags.scale
        self.layers = flags.layers
        self.depth_wise_convolution = flags.depth_wise_convolution
        self.resampling_method = BICUBIC_METHOD_STRING
        self.self_ensemble = flags.self_ensemble

        # Training Parameters
        self.optimizer = flags.optimizer
        self.beta1 = flags.beta1
        self.beta2 = flags.beta2
        self.momentum = flags.momentum
        self.batch_num = flags.batch_num
        self.batch_image_size = flags.batch_image_size
        self.clipping_norm = flags.clipping_norm

        # Learning Rate Control for Training
        self.initial_lr = flags.initial_lr
        self.lr_decay = flags.lr_decay
        self.lr_decay_epoch = flags.lr_decay_epoch

        # Dataset or Others
        self.training_images = int(math.ceil(flags.training_images / flags.batch_num) * flags.batch_num)

        # Image Processing Parameters
        self.max_value = flags.max_value
        self.channels = flags.channels
        self.output_channels = flags.channels
        self.psnr_calc_border_size = flags.psnr_calc_border_size
        if self.psnr_calc_border_size < 0:
            self.psnr_calc_border_size = 2 + self.scale

        # initialize variables
        self.name = self.get_model_name(model_name)
        self.total_epochs = 0
        lr = self.initial_lr
        while lr > flags.end_lr:
            self.total_epochs += self.lr_decay_epoch
            lr *= self.lr_decay

        # initialize environment
        util.make_dir(self.checkpoint_dir)
        util.make_dir(flags.graph_dir)
        util.make_dir(self.tf_log_dir)
        if flags.initialize_tf_log:
            util.clean_dir(self.tf_log_dir)
        util.set_logging(flags.log_filename, stream_log_level=logging.INFO, file_log_level=logging.INFO,
                         tf_log_level=tf.logging.WARN)
        logging.info("\nLFFN-------------------------------------")
        logging.info("%s [%s]" % (util.get_now_date(), self.name))

        self.init_train_step()

    def get_model_name(self, model_name, name_postfix=""):
        if model_name is "":
            name = "LFFN%d" % self.layers
            if self.scale != 2:
                name += "_Sc%d" % self.scale
            if name_postfix is not "":
                name += "_" + name_postfix
        else:
            name = "LFFN_%s" % model_name

        return name

    # depth-wise convolution
    def split_conv(self, input_layer, name, input_channel, output_channel):
        if self.depth_wise_convolution:
            x = self.depth_conv2d_layer(name=name + '2', input_tensor=input_layer, kernel_size1=3, kernel_size2=3,
                                 input_feature_num=input_channel, output_feature_num=output_channel, activator=None)
        else:
            x = self.conv2d_layer(name=name + '2', input_tensor=input_layer, kernel_size1=3, kernel_size2=3,
                                 input_feature_num=input_channel, output_feature_num=output_channel, activator=None)

        return x

    def res_block(self, input_layer, input_channel, block_num, reuse=False):
        with tf.variable_scope('res_block_%d' % (block_num), reuse=reuse):
            input = self.conv2d_layer(name='conv1', input_tensor=input_layer, kernel_size1=3, kernel_size2=3,
                                      input_feature_num=input_channel, output_feature_num=input_channel,
                                      activator='relu')
            x = self.conv2d_layer(name='conv2', input_tensor=input, kernel_size1=3, kernel_size2=3,
                                  input_feature_num=input_channel, output_feature_num=input_channel, activator=None)
            x = tf.add(input_layer, x, name='block%d_add' % (block_num))
            return x

    def path_conv(self, input, input_channel, output_channel, block_num, path_num, path_depth, is_linear=False):
        path_tmp = input
        if is_linear:
            for i in range(path_depth):
                path_tmp = self.split_conv(path_tmp, name='block%d_path%d_conv%d' % (block_num, path_num, i),
                                           input_channel=input_channel, output_channel=output_channel)
        else:
            for i in range(path_depth):
                path_tmp = self.split_conv(path_tmp, name='block%d_path%d_conv%d' % (block_num, path_num, i),
                                           input_channel=input_channel, output_channel=output_channel)
                path_tmp = self.build_activator(path_tmp, output_channel, "prelu",
                                                base_name='block%d_path%d_act%d' % (block_num, path_num, i))
        return path_tmp

    #proposed spindle block
    def spindle_block(self, input_layer, input_channel, block_num, is_res=True, reuse=False):
        with tf.variable_scope('res_block_v2_%d' % (block_num), reuse=reuse):
            input_list = []
            input = self.conv2d_layer(name='expand', input_tensor=input_layer, kernel_size1=1, kernel_size2=1,
                                      input_feature_num=input_channel, output_feature_num=64, activator=None)
            for i in range(4):
                input_list.append(
                    tf.slice(input, begin=[0, 0, 0, i * 16], size=[-1, -1, -1, 16], name="squeeze_%d" % (i)))
            path_list = []

            path_list.append(self.path_conv(input_list[0], 16, 16, block_num, 1, 1, is_linear=True))
            path_list.append(self.path_conv(input_list[1], 16, 16, block_num, 2, 1, is_linear=False))
            path_list.append(self.path_conv(input_list[2], 16, 16, block_num, 3, 2, is_linear=False))
            path_list.append(self.path_conv(input_list[3], 16, 16, block_num, 4, 3, is_linear=False))

            x = tf.concat(path_list, 3, name='block%d_concat' % (block_num))
            x = self.conv2d_layer(name='block%d_conv_final' % (block_num), input_tensor=x, kernel_size1=1,
                                  kernel_size2=1, input_feature_num=64,
                                  output_feature_num=input_channel, activator=None)
            if is_res:
                x = tf.add(input_layer, x, name='block%d_add' % (block_num))
            return x

    # proposed sffm module
    def SFFM_module(self,input_list,module_name,channel_num,reuse=False):
        with tf.variable_scope(module_name, reuse=reuse):
            middle_mask=[]
            layer_num = len(input_list)
            for i in range(layer_num):
                middle_temp = tf.reduce_mean(input_list[i], reduction_indices=[1, 2])
                middle_temp = tf.reshape(tf.layers.dense(inputs=middle_temp, use_bias=False,
                                                         units=channel_num,name='dense%d' % (i)), [-1, 1, 1, 48])
                middle_mask.append(middle_temp)
            channel_con = tf.concat(middle_mask, 1, name="concat1")
            channel_list = tf.split(value=channel_con, num_or_size_splits=channel_num, axis=3)
            for i in range(channel_num):
                channel_fla = tf.layers.Flatten()(channel_list[i])
                channel_sof = tf.nn.softmax(channel_fla)  # denotes the weight of the middle layer in channel i
                channel_list[i] = tf.reshape(channel_sof, [-1, layer_num, 1, 1])
            channel_con = tf.concat(channel_list, 3, name="concat2")
            #self.softmax_map = channel_con
            attention_list = tf.split(value=channel_con, num_or_size_splits=layer_num, axis=1)
            for i in range(layer_num):
                input_list[i] = input_list[i] * attention_list[i]
            return input_list

    # proposed spindle module
    def mul_module(self, input_layer, in_channel, out_channel, layer_name, block_num, reuse=False):
        with tf.variable_scope(layer_name, reuse=reuse):
            Block_list = []
            block_input = input_layer
            for i in range(block_num):
                block_input = self.spindle_block(block_input, in_channel, i, is_res=True, reuse=reuse)
                # block_input = self.res_block_v1(block_input, 64, 64, i, reuse=False)
                Block_list.append(block_input)

            block_concat = tf.concat(Block_list, 3, name="block_concat")
            block_output = self.conv2d_layer(name='fuse_conv', input_tensor=block_concat, kernel_size1=1,
                                             kernel_size2=1, input_feature_num=block_num*in_channel,
                                             output_feature_num=out_channel, activator=None)

            block_output = tf.add(block_output, input_layer, name="res_add")
        return block_output

    def build_graph(self):

        #input
        self.x = tf.placeholder(tf.float32, shape=[None, None, None, self.channels], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None, None, None, self.output_channels], name="y")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        input_channels = 48
        # first layer
        input_tensor = self.conv2d_layer(name='feature_conv1', input_tensor=self.x,
                                         kernel_size1=3, kernel_size2=3, input_feature_num=3,
                                         output_feature_num=48, activator=None)
        temp_in = input_tensor

        self.middle_layer = []
        for i in range(self.layers):
            input_tensor = self.mul_module(input_tensor, 48, 48, 'module%d' % (i), 4)
            self.middle_layer.append(input_tensor)

        # fuse features with sffm module
        middle_fuse = self.SFFM_module(self.middle_layer, "middle_fuse", 48)
        self.layer_concat = tf.add_n(middle_fuse)

        input_tensor = self.conv2d_layer(name='feature_conv3', input_tensor=self.layer_concat, kernel_size1=1,
                                         kernel_size2=1, input_feature_num=48,
                                         output_feature_num=48, activator=None)
        input_tensor = tf.add(input_tensor, temp_in, name='add3')

        # building upsampling layer
        if self.scale == 4:
            up_tensor = self.build_pixel_shuffler_layer("Up-PS", input_tensor, 2, input_channels)
            up_tensor = self.build_pixel_shuffler_layer("Up-PS2", up_tensor, 2, input_channels)
        else:
            up_tensor = self.build_pixel_shuffler_layer("Up-PS", input_tensor, self.scale, input_channels)

       # last layer
        self.y_ = self.conv2d_layer(name='R-CNN1', input_tensor=up_tensor, kernel_size1=1,
                          kernel_size2=1, input_feature_num=input_channels,
                          output_feature_num=self.output_channels, activator=None)

        logging.info("The FSR model Complexity:%s Receptive Fields:%d" % (
            "{:,}".format(self.complexity), self.receptive_fields))

    def build_optimizer(self):
        """
        Build loss function.
        """
        self.lr_input = tf.placeholder(tf.float32, shape=[], name="LearningRate")
        diff = self.y_ - self.y

        self.mae = tf.reduce_mean(tf.abs(diff), name="mse")
        self.loss = self.mae

        if self.save_loss:
            tf.summary.scalar("loss/" + self.name, self.loss)

        self.training_optimizer = self.add_optimizer_op(self.loss, self.lr_input)

        util.print_num_of_total_parameters(output_detail=True)

    def add_optimizer_op(self, loss, lr_input):

        if self.optimizer == "gd":
            optimizer = tf.train.GradientDescentOptimizer(lr_input)
        elif self.optimizer == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(lr_input)
        elif self.optimizer == "adagrad":
            optimizer = tf.train.AdagradOptimizer(lr_input)
        elif self.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(lr_input, beta1=self.beta1, beta2=self.beta2)
        elif self.optimizer == "momentum":
            optimizer = tf.train.MomentumOptimizer(lr_input, self.momentum)
        elif self.optimizer == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(lr_input, momentum=self.momentum)
        else:
            print("Optimizer arg should be one of [gd, adadelta, adagrad, adam, momentum, rmsprop].")
            return None

        if self.clipping_norm > 0:
            trainables = tf.trainable_variables()
            grads = tf.gradients(loss, trainables)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.clipping_norm)
            grad_var_pairs = zip(grads, trainables)

            training_optimizer = optimizer.apply_gradients(grad_var_pairs)
        else:
            training_optimizer = optimizer.minimize(loss)

        return training_optimizer

    def init_epoch_index(self):
        self.batch_true = self.batch_num * [None]
        self.training_psnr_sum = 0
        self.training_mse_sum = 0
        self.training_step = 0

    def train_batch(self, loader):

        for batch in enumerate(loader.loader_train):
            lr, hr = batch[1][0], batch[1][1]

            feed_dict = {self.x: lr.numpy(), self.y: hr.numpy(),
                         self.lr_input: self.lr, self.is_training: 1}

            _, mse = self.sess.run([self.training_optimizer, self.mae], feed_dict=feed_dict)

            self.training_mse_sum += mse
            self.training_psnr_sum += util.get_psnr(mse, max_value=self.max_value)
            self.training_step += 1
            self.step += 1

    def log_to_tensorboard(self, test_filename, psnr, save_meta_data=True):

        # todo
        save_meta_data = False

        org_image = util.set_image_alignment(util.load_image(test_filename, print_console=False), self.scale)

        if len(org_image.shape) >= 3 and org_image.shape[2] == 3 and self.channels == 1:
            org_image = util.convert_rgb_to_y(org_image)

        input_image = util.resize_image_by_pil(org_image, 1.0 / self.scale, resampling_method=self.resampling_method)

        feed_dict = {self.x: input_image.reshape([1, input_image.shape[0], input_image.shape[1], input_image.shape[2]]),
                     self.y: org_image.reshape([1, org_image.shape[0], org_image.shape[1], org_image.shape[2]]),
                     self.is_training: 0}

        if save_meta_data:
            run_metadata = tf.RunMetadata()
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            summary_str, _ = self.sess.run([self.summary_op, self.mae], feed_dict=feed_dict, options=run_options,
                                           run_metadata=run_metadata)
            self.test_writer.add_run_metadata(run_metadata, "step%d" % self.epochs_completed)

            filename = self.checkpoint_dir + "/" + self.name + "_metadata.txt"
            with open(filename, "w") as out:
                out.write(str(run_metadata))

            tf.contrib.tfprof.model_analyzer.print_model_analysis(
                tf.get_default_graph(), run_meta=run_metadata,
                tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)

        else:
            summary_str, _ = self.sess.run([self.summary_op, self.mae], feed_dict=feed_dict)

        self.train_writer.add_summary(summary_str, self.epochs_completed)
        util.log_scalar_value(self.train_writer, 'training_PSNR', self.training_psnr_sum / self.training_step,
                              self.epochs_completed)
        util.log_scalar_value(self.train_writer, 'LR', self.lr, self.epochs_completed)
        self.train_writer.flush()

        util.log_scalar_value(self.test_writer, 'PSNR', psnr, self.epochs_completed)
        self.test_writer.flush()

    def update_epoch_and_lr(self):

        self.epochs_completed_in_stage += 1
        if self.epochs_completed_in_stage >= self.lr_decay_epoch:
            # set new learning rate
            self.lr *= self.lr_decay
            self.epochs_completed_in_stage = 0
            return True
        else:
            return False

    def print_status(self, mse, psnr, log=False):

        if self.step == 0:
            logging.info("Initial MSE:%f PSNR:%f" % (mse, psnr))
        else:
            processing_time = (time.time() - self.start_time) / self.step
            line_a = "%s Step:%s MSE:%f PSNR:%f (Training PSNR:%0.3f)" % (
                util.get_now_date(), "{:,}".format(self.step), mse, psnr, self.training_psnr_sum / self.training_step)
            estimated = processing_time * (self.total_epochs - self.epochs_completed) * (
                self.training_images // self.batch_num)
            h = estimated // (60 * 60)
            estimated -= h * 60 * 60
            m = estimated // 60
            s = estimated - m * 60
            line_b = "Epoch:%d LR:%f (%2.3fsec/step) Estimated:%d:%d:%d" % (
                self.epochs_completed, self.lr, processing_time, h, m, s)
            if log:
                logging.info(line_a)
                logging.info(line_b)
            else:
                print(line_a)
                print(line_b)

    def evaluate(self, test_filenames):
        total_mse = total_psnr = 0
        if len(test_filenames) == 0:
            return 0, 0

        for filename in test_filenames:
            mse, psnr_predicted, ssim_predicted = self.do_for_evaluate(filename, print_console=False)
            total_mse += mse
            total_psnr += psnr_predicted

        return total_mse / len(test_filenames), total_psnr / len(test_filenames)

    def do(self, input_image):

        h, w = input_image.shape[:2]
        ch = input_image.shape[2] if len(input_image.shape) > 2 else 1

        if self.max_value != 255.0:
            input_image = np.multiply(input_image, self.max_value / 255.0)  # type: np.ndarray

        spend_time=0.0
        if self.self_ensemble > 1:
            output = np.zeros([self.scale * h, self.scale * w, 3])

            for i in range(self.self_ensemble):
                image = util.flip(input_image, i)
                start_time = time.time()
                y = self.sess.run(self.y_, feed_dict={self.x: image.reshape(1, image.shape[0], image.shape[1], ch),
                                                      self.is_training: 0})
                spend_time = time.time() - start_time
                restored = util.flip(y[0], i, invert=True)
                output += restored

            output /= self.self_ensemble
        else:
            start_time = time.time()
            y = self.sess.run(self.y_, feed_dict={self.x: input_image.reshape(1, input_image.shape[0], input_image.shape[1], ch)/255.-0.5,
                                                  self.is_training: 0})
            spend_time = time.time() - start_time
            output = (y[0]+0.5)*255.

        if self.max_value != 255.0:
            hr_image = np.multiply(output, 255.0 / self.max_value)
        else:
            hr_image = output

        return hr_image, spend_time

    def do_for_evaluate_with_output(self, file_path, output_directory, print_console=False):

        filename, extension = os.path.splitext(file_path)
        output_directory += "/" + self.name + "/"
        util.make_dir(output_directory)

        true_image = util.set_image_alignment(util.load_image(file_path, print_console=False), self.scale)

        if true_image.shape[2] == 3 and self.channels == 3:

            # for color images
            input_image = util.build_input_image(true_image, scale=self.scale, alignment=self.scale)
            input_bicubic_image = util.resize_image_by_pil(input_image, self.scale,
                                                           resampling_method=self.resampling_method)

            output_image,spend_time = self.do(input_image)  # SR

            SR_y = eva.convert_rgb_to_y(output_image)
            HR_y = eva.convert_rgb_to_y(true_image)
            psnr_predicted = eva.PSNR(np.uint8(HR_y), np.uint8(SR_y), shave_border=self.psnr_calc_border_size)
            ssim_predicted = eva.compute_ssim(np.squeeze(HR_y), np.squeeze(SR_y))

            mse = util.compute_mse(HR_y, SR_y, border_size=self.psnr_calc_border_size)
            loss_image = util.get_loss_image(HR_y, SR_y, border_size=self.psnr_calc_border_size)

            util.save_image(output_directory + file_path[29:], true_image)
            util.save_image(output_directory + filename[28:] + "_input" + extension, input_image)
            util.save_image(output_directory + filename[28:] + "_input_bicubic" + extension, input_bicubic_image)
            util.save_image(output_directory + filename[28:] + "_sr" + extension, output_image)
            util.save_image(output_directory + filename[28:] + "_loss" + extension, loss_image)

        elif true_image.shape[2] == 1 and self.channels == 1:
            # for monochrome images
            input_image = util.build_input_image(true_image, scale=self.scale, alignment=self.scale)
            output_image,spend_time = self.do(input_image)

            psnr_predicted = eva.PSNR(np.uint8(true_image), np.uint8(output_image),
                                      shave_border=self.psnr_calc_border_size)
            ssim_predicted = eva.compute_ssim(np.squeeze(true_image), np.squeeze(output_image))

            mse = util.compute_mse(true_image, output_image, border_size=self.psnr_calc_border_size)
            util.save_image(output_directory + file_path, true_image)
            util.save_image(output_directory + filename + "_sr" + extension, output_image)
        else:
            psnr_predicted = 0.0
            ssim_predicted = 0.0
            mse = 0.0
            spend_time =0.0

        if print_console:
            print("[%s] psnr:%f, ssim:%f, time:%f" % (filename, psnr_predicted, ssim_predicted, spend_time))

        return mse, psnr_predicted, ssim_predicted, spend_time

    def do_for_evaluate(self, file_path, print_console=False):

        true_image = util.set_image_alignment(util.load_image(file_path, print_console=False), self.scale)

        if true_image.shape[2] == 3 and self.channels == 3:
            # for color images
            input_image = util.build_input_image(true_image, scale=self.scale,alignment=self.scale)
            output_image = self.do(input_image)

            SR_y = eva.convert_rgb_to_y(output_image)
            HR_y = eva.convert_rgb_to_y(true_image)
            # SR_y = SR_y.reshape((1,image_size[0],image_size[1]))
            # HR_y = HR_y.reshape((1,image_size[0],image_size[1]))
            psnr_predicted = eva.PSNR(np.uint8(HR_y), np.uint8(SR_y), shave_border=self.psnr_calc_border_size)
            ssim_predicted = eva.compute_ssim(np.squeeze(HR_y), np.squeeze(SR_y))
            mse = util.compute_mse(HR_y, SR_y, border_size=self.psnr_calc_border_size)

        elif true_image.shape[2] == 1 and self.channels == 1:
            # for monochrome images
            input_image = util.build_input_image(true_image, scale=self.scale, alignment=self.scale)
            output_image = self.do(input_image)

            psnr_predicted = eva.PSNR(np.uint8(true_image), np.uint8(output_image),
                                      shave_border=self.psnr_calc_border_size)
            ssim_predicted = eva.compute_ssim(np.squeeze(true_image), np.squeeze(output_image))
            mse = util.compute_mse(true_image, output_image, border_size=self.psnr_calc_border_size)
        else:
            psnr_predicted = 0
            ssim_predicted = 0
            mse = 0

        if print_console:
            print("MSE:%f, PSNR:%f, SSIM:%f" % (mse, psnr_predicted, ssim_predicted))

        return mse, psnr_predicted, ssim_predicted

    def init_train_step(self):
        self.lr = self.initial_lr
        self.epochs_completed = 0
        self.epochs_completed_in_stage = 0
        self.min_validation_mse = -1
        self.min_validation_epoch = -1
        self.step = 0

        self.start_time = time.time()

    def end_train_step(self):
        self.total_time = time.time() - self.start_time

    def print_steps_completed(self, output_to_logging=False):
        if self.step == 0:
            return

        processing_time = self.total_time / self.step
        h = self.total_time // (60 * 60)
        m = (self.total_time - h * 60 * 60) // 60
        s = (self.total_time - h * 60 * 60 - m * 60)

        status = "Finished at Total Epoch:%d Steps:%s Time:%02d:%02d:%02d (%2.3fsec/step) %d x %d x %d patches" % (
            self.epochs_completed, "{:,}".format(self.step), h, m, s, processing_time,
            self.batch_image_size, self.batch_image_size, self.training_images)

        if output_to_logging:
            logging.info(status)
        else:
            print(status)
