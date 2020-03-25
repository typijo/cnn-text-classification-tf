"""TF 2.0 compat"""

import tensorflow.compat.v1 as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import ltcdata
from text_cnn import TextCNN

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("show_traininfo_every", 10, "Show train loss/acc after this many steps (default: 10)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# ltc Parameters
tf.flags.DEFINE_boolean("use_BERT_tokenizer", False, "Whether use tokenizer of BERT")
tf.flags.DEFINE_boolean("united_sid", False, "Make a united classifier")
tf.flags.DEFINE_boolean("global_sid", False, "Make a classifier of global sid. This works only when united_sid=True")
tf.flags.DEFINE_string("data_dir", "data_ltc", "Directory of datasets")
tf.flags.DEFINE_string("base_dir", ".", "Directory of dataset / model. Select gs bucket when running on colab.")
tf.flags.DEFINE_integer("sid_from", 0, "Ltc from which it trains")
tf.flags.DEFINE_integer("max_examples", 500000, "Max number of examples")

FLAGS = tf.flags.FLAGS

def train(data_train, data_dev, vocab_size, out_dir="."):
    # Training
    # ==================================================
    x_train, y_train = data_train
    x_dev, y_dev = data_dev

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=vocab_size,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.join(out_dir, "checkpoints")
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            print("checkpoints are saved to", checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % FLAGS.show_traininfo_every == 0:
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                if writer:
                    writer.add_summary(summaries, step)
                
                return loss, accuracy, step

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            print("do %d epochs" % FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    batches_dev = data_helpers.batch_iter(
                        list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
                    
                    losses = []
                    accuracies = []
                    for batch_dev in batches_dev:
                        loss_this, accuracy_this, step = dev_step(
                            x_dev, y_dev, writer=dev_summary_writer)
                        losses.append(loss_this)
                        accuracies.append(accuracy_this)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(
                        time_str, step,
                        sum(losses)/len(losses), sum(accuracies)/len(accuracies)))

                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
            
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Finally saved model checkpoint to {}\n".format(path))

def main(argv=None):
    for sid in ltcdata.get_sid_list(FLAGS.united_sid):
        if sid < FLAGS.sid_from:
            continue
        
        data = ltcdata.load_data(
            os.path.join(
                FLAGS.base_dir, ltcdata.make_name_indir(FLAGS.united_sid)),
            with_train=True, with_dev=True,
            is_united=FLAGS.united_sid, is_global=FLAGS.global_sid,
            use_BERT_tokenizer=FLAGS.use_BERT_tokenizer, sid=sid,
            max_examples=FLAGS.max_examples)

        name_cp_dir = ltcdata.make_name_outdir(
            FLAGS.united_sid, FLAGS.global_sid, FLAGS.use_BERT_tokenizer, sid)
        cp_dir = os.path.join(FLAGS.base_dir, name_cp_dir)
        
        train(data["train"], data["dev"], data["num_vocab"], out_dir=cp_dir)

if __name__ == '__main__':
    tf.app.run()
