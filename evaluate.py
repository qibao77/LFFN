"""
Paper: "Lightweight Feature Fusion Network for Single Image Super-Resolution"
Author: Wei Wang
Github:

Functions for evaluating model performance
"""

import logging
import tensorflow as tf
import os
import LFFN
from helper import args, utilty as util
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

args.flags.DEFINE_boolean("save_results", True, "Save result, bicubic and loss images")
FLAGS = args.get()

def create_str_to_txt(path_file, str_data):
    if not os.path.exists(path_file):
        with open(path_file, "w") as f:
            print(f)
    with open(path_file,"a") as f:
        f.write(str_data)

def main(not_parsed_args):
    if len(not_parsed_args) > 1:
        print("Unknown args:%s" % not_parsed_args)
        exit()

    model = LFFN.SuperResolution(FLAGS, model_name=FLAGS.model_name)
    model.build_graph()
    model.build_summary_saver()
    model.init_all_variables()

    # test_list = ['Manga109', 'bsd100', 'Urban100', 'Set5']
    # test_list = ['bsd100', 'set5']
    # test_list = ['video_demo']
    test_list = ['Set5']
    for i in range(FLAGS.tests):
        model.load_model(FLAGS.load_model_name, trial=i, output_log=True if FLAGS.tests > 1 else False)
        for test_data in test_list:
            test(model, test_data)


def test(model, test_data):
    test_filenames = util.get_files_in_directory(FLAGS.test_dir + test_data)
    total_psnr = total_ssim = total_mse = total_time = 0

    path_file = FLAGS.output_dir + "/" + test_data + ".txt"

    for filename in test_filenames:
        mse, psnr_predicted, ssim_predicted , spend_time = model.do_for_evaluate_with_output(filename,output_directory=FLAGS.output_dir, print_console=False)
        total_mse += mse
        # total_psnr += util.get_psnr(mse, max_value=FLAGS.max_value)
        total_psnr += psnr_predicted
        total_ssim += ssim_predicted
        total_time += spend_time
        create_str_to_txt(path_file,test_data + ":" + "\n")
        create_str_to_txt(path_file,"PSNR:"+str(psnr_predicted)+" , SSIM:"+str(ssim_predicted)+" , Time:"+str(spend_time)+"\n")

    logging.info("\n=== [%s] MSE:%f, PSNR:%f , SSIM:%f , Time:%f===" % (
        test_data, total_mse / len(test_filenames), total_psnr / len(test_filenames), total_ssim / len(test_filenames), total_time / len(test_filenames)))
    create_str_to_txt(path_file, "================================================\n"+
                      "PSNR:" + str(total_psnr / len(test_filenames)) +
                      " , SSIM:" + str(total_ssim / len(test_filenames)) +
                      " , Time:" + str(total_time / len(test_filenames)) +
                      "\n================================================\n")

if __name__ == '__main__':
    tf.app.run()
