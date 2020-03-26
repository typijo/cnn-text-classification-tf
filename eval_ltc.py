"""TF 2.0 compat"""

import tensorflow.compat.v1 as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import ltcdata
from text_cnn import TextCNN
import csv
import pprint

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# ltc Parameters
tf.flags.DEFINE_boolean("use_BERT_tokenizer", False, "Whether use tokenizer of BERT")
tf.flags.DEFINE_boolean("united_sid", False, "Make a united classifier")
tf.flags.DEFINE_boolean("global_sid", False, "Make a classifier of global sid. This works only when united_sid=True")
tf.flags.DEFINE_string("data_dir", "data_ltc/separated", "Directory of datasets")
tf.flags.DEFINE_string("base_dir", ".", "Directory of dataset / model. Select gs bucket when running on colab.")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")

FLAGS = tf.flags.FLAGS

path_accdata = os.path.join(FLAGS.base_dir, ltcdata.make_str_of_setting(
    FLAGS.united_sid, FLAGS.global_sid, FLAGS.use_BERT_tokenizer, FLAGS.batch_size, FLAGS.num_epochs, FLAGS.embedding_dim) + ".txt")

with tf.io.gfile.GFile(path_accdata, "w") as f_accdata:
    for sid in ltcdata.get_sid_list(FLAGS.united_sid):
        data_dir = os.path.join(FLAGS.base_dir, FLAGS.data_dir)
        data = ltcdata.load_data(data_dir, with_eval=True,
                is_united=FLAGS.united_sid, is_global=FLAGS.global_sid,
                use_BERT_tokenizer=FLAGS.use_BERT_tokenizer, sid=sid, with_idrange=True)
        x_test, y_test = data["eval"]

        print("\nEvaluating... sid: %d\n" % sid)
        f_accdata.write("**sid: %d**\n" % sid)

        # Evaluation
        # ==================================================
        name_cp_dir = ltcdata.make_name_outdir(
            FLAGS.united_sid, FLAGS.global_sid, FLAGS.use_BERT_tokenizer, sid, FLAGS.batch_size, FLAGS.num_epochs, FLAGS.embedding_dim)
        cp_dir = os.path.join(FLAGS.base_dir, name_cp_dir, "checkpoints")
        checkpoint_file = tf.train.latest_checkpoint(cp_dir)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                # Generate batches for one epoch
                batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

                # Probs for global prediction
                probs = graph.get_operation_by_name("output/scores").outputs[0]

                # Collect the predictions and probs here
                all_predictions = []
                all_probs = []

                for x_test_batch in batches:
                    batch_predictions, batch_probs = sess.run(
                        [predictions, probs], {input_x: x_test_batch, dropout_keep_prob: 1.0})
                    all_predictions = np.concatenate([all_predictions, batch_predictions])
                    all_probs = np.concatenate([all_probs, batch_probs])

        # Print accuracy if y_test is defined
        if y_test is not None:
            y_test = np.argmax(y_test, axis=1)
            all_predictions_mod = []
            if FLAGS.united_sid and FLAGS.global_sid:
                correct_predictions = 0
                idrange = data["idrange"]
                for yi, probi in zip(y_test, all_probs):
                    id_from, id_to = idrange[yi]
                    predi = id_from + np.argmax(probi[id_from:id_to+1])
                    if yi == predi:
                        correct_predictions += 1
                    all_predictions_mod.append(predi)
                
                correct_predictions = float(correct_predictions)
                all_predictions_mod = np.array(all_predictions_mod)
            else:
                correct_predictions = float(sum(all_predictions == y_test))
                all_predictions_mod = all_predictions
            print("Total number of test examples: {}".format(len(y_test)))
            print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

            f_accdata.write("Total number of test examples: {}\n".format(len(y_test)))
            f_accdata.write("Accuracy: {:g}\n".format(correct_predictions/float(len(y_test))))

            cm = [[0 for _ in range(4)] for _ in range(4)]

            for yi, pi in zip(y_test, all_predictions_mod):
                yi = int(yi)
                pi = int(pi)
                cm[yi][pi] += 1
            
            f_accdata.write("ans|pred,#1,#2,#3,#4\n")
            for i, cmi in enumerate(cm):
                f_accdata.write("#%d,%s\n" % (
                    i, ",".join([str(cmij) for cmij in cmi])))
            
            pprint.pprint(cm)

