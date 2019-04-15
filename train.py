"""
Paper: "Lightweight Feature Fusion Network for Single Image Super-Resolution"
Author: Wei Wang
Github: https://github.com/qibao77/LFFN-master.git

training functions.
Testing Environment: Python 3.6.1, tensorflow >= 1.4.0
"""

import logging
import sys
import tensorflow as tf
import os
import LFFN
import data
from helper import args, utilty as util

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
FLAGS = args.get()

def main(not_parsed_args):
    if len(not_parsed_args) > 1:
        print("Unknown args:%s" % not_parsed_args)
        exit()

    model = LFFN.SuperResolution(FLAGS, model_name=FLAGS.model_name)
    model.build_graph()
    model.build_optimizer()
    model.build_summary_saver()

    logging.info("\n" + str(sys.argv))
    logging.info("Test Data:" + FLAGS.test_dataset + " Training Data:" + FLAGS.train_dir+FLAGS.data_train)
    util.print_num_of_total_parameters(output_to_logging=True)

    total_psnr = total_mse = 0

    for i in range(FLAGS.tests):
        mse = train(model, FLAGS, i)  # begin train
        psnr = util.get_psnr(mse, max_value=FLAGS.max_value)
        total_mse += mse
        total_psnr += psnr

        logging.info("\nTrial(%d) %s" % (i, util.get_now_date()))
        model.print_steps_completed(output_to_logging=True)
        logging.info("MSE:%f, PSNR:%f\n" % (mse, psnr))

    if FLAGS.tests > 1:
        logging.info("\n=== Final Average [%s] MSE:%f, PSNR:%f ===" % (
        FLAGS.test_dataset, total_mse / FLAGS.tests, total_psnr / FLAGS.tests))

    model.copy_log_to_archive("archive")


def train(model, flags, trial):
    test_filenames = util.get_files_in_directory(FLAGS.test_dir_train)

    model.init_all_variables()
    if flags.load_model_name != "":
        model.load_model(flags.load_model_name, output_log=True)

    model.init_train_step()
    model.init_epoch_index()
    model_updated = True
    mse = 0

    loader = data.Data(FLAGS)
    while model.lr > flags.end_lr:

        #model.build_input_batch()
        model.train_batch(loader)

        if model.training_step * model.batch_num >= model.training_images:
            # training epoch finished
            model.epochs_completed += 1
            mse, psnr = model.evaluate(test_filenames)
            model.print_status(mse, psnr, log=model_updated)
            model.log_to_tensorboard(test_filenames[0], psnr, save_meta_data=model_updated)

            model_updated = model.update_epoch_and_lr()
            model.init_epoch_index()
            model.save_model(trial=trial, output_log=True)

    model.end_train_step()
    model.save_model(trial=trial, output_log=True)

    # outputs result
    # test(model, flags.test_dataset)

    if FLAGS.do_benchmark:
        for test_data in ['set5', 'Manga109', 'bsd100', 'Urban100']:
            test(model, test_data)

    return mse


def test(model, test_data):
    test_filenames = util.get_files_in_directory(FLAGS.test_dir + test_data)
    total_psnr = total_ssim = total_mse = 0

    for filename in test_filenames:
        mse, psnr_predicted, ssim_predicted = model.do_for_evaluate_with_output(filename,output_directory=FLAGS.output_dir,print_console=False)
        total_mse += mse
        # total_psnr += util.get_psnr(mse, max_value=FLAGS.max_value)
        total_psnr += psnr_predicted
        total_ssim += ssim_predicted

    logging.info("\n=== [%s] MSE:%f, PSNR:%f , SSIM:%f===" % (
        test_data, total_mse / len(test_filenames), total_psnr / len(test_filenames), total_ssim / len(test_filenames)))


if __name__ == '__main__':
    tf.app.run()
