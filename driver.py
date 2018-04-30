import config
import utils
from functools import partial
import tensorflow as tf
import numpy as np
import cPickle
import os
import json
import time
from collections import deque

try:
    from .model import BiEncoderModel
except:
    from model import BiEncoderModel

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_input(example):
    """

    :return: dict
    """
    features = tf.parse_single_example(example, features={
                                           'context': tf.FixedLenFeature([config.MAX_SENTENCE_LEN], tf.int64),
                                           'utterance': tf.FixedLenFeature([config.MAX_SENTENCE_LEN], tf.int64),
                                           'context_len': tf.FixedLenFeature([], tf.int64),
                                           'utterance_len': tf.FixedLenFeature([], tf.int64),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                   })
    try:
        with tf.variable_scope('embedding', reuse=True):
            embeddings_mat = tf.get_variable("word_embeddings")
            logging.info("Reused embedding")
    except ValueError:
        logging.info("Getting embedding for the first time")
        with open(os.path.join(BASE_DIR, config.VOCABULARY)) as fh:
            vocabulary = json.load(fh)

        with tf.variable_scope('embedding'):
            embeddings_matrix = utils.build_embedding_matrix(os.path.join(BASE_DIR, config.EMBED_FILE),
                                                             vocabulary=vocabulary,
                                                             embed_len=config.EMBED_LEN)
            embeddings_mat = tf.get_variable("word_embeddings", trainable=False, initializer=embeddings_matrix)
    print("shape of embedding matrix", embeddings_mat.shape)

    features['context'] = tf.nn.embedding_lookup(embeddings_mat, features['context'])
    features['utterance'] = tf.nn.embedding_lookup(embeddings_mat, features['utterance'])
    print("shape of train context", features['context'].shape)
    print("shape of train context len", features['context_len'].shape)
    return features['context'], features['utterance'], features['context_len'], features['utterance_len'], features['label']


def build_input_pipeline(in_files, batch_size, num_epochs=None, mode='train'):
    """
    Build an input pipeline with the DataSet API
    :param in_files list of tfrecords filenames
    :return dataset iterator (use get_next() method to get the next batch of data from the dataset iterator
    """
    dataset = tf.contrib.data.TFRecordDataset(in_files)
    dataset = dataset.map(parse_input, num_threads=12,
                          output_buffer_size=10 * batch_size)  # Parse the record to tensor

    if mode is 'train':  # we only want to shuffle for training dataset
        dataset = dataset.shuffle(buffer_size=4 * batch_size)

    dataset = dataset.batch(batch_size)

    if num_epochs:
        dataset = dataset.repeat(num_epochs)
    else:
        dataset = dataset.repeat()  # Repeat the input indefinitely.
    iterator = dataset.make_initializable_iterator()
    return iterator


def train():
    """
    Builds the graph and runs the graph in a session
    :return:
    """
    train_files = config.TRAIN_FILES
    validation_files = config.VALIDATION_FILES

    with tf.Graph().as_default():

        logging.info("Building train input pipeline")
        input_train_iter = build_input_pipeline(train_files,
                                                config.TRAIN_BATCH_SIZE,
                                                num_epochs=None)  # change num_epochs to None in production

        logging.info("Building validation input pipeline")
        input_validation_iter = build_input_pipeline(validation_files,
                                                     config.VALIDATION_BATCH_SIZE,
                                                     mode='valid')

        model = BiEncoderModel(input_train_iter, input_validation_iter)

        logging.info("Building graph")
        train_op = model.build_graph()  # for training
        valid_op = model.get_validation_probabilities()  # for model selection

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(input_train_iter.initializer)
        sess.run(input_validation_iter.initializer)

        saver = tf.train.Saver()

        #  starting the training
        logging.info("Training starts...")
        batch = 0
        epoch_step = 0
        num_batches_train = int(1000000/config.TRAIN_BATCH_SIZE)
        num_batches_valid = int(195600/config.VALIDATION_BATCH_SIZE)
        evaluation_metric_old = 0.0
        evaluation_metric_history = deque([0.0, 0.0, 0.0, 0.0, 0.0], 5)  # used for early stopping
        try:
            while True:
                # here calculate accuracy and/or training loss
                _, loss = sess.run([train_op, model.get_loss()])

                if batch % 100 == 0:
                    logger.info("Epoch step: {0} Train step: {1} loss = {2}".format(epoch_step, batch, loss))
                
                batch += 1

                # evaluate the model with the validation set every 1/4 of the epoch
                if batch % int(0.25 * num_batches_train) == 0:  # change 0.01 to 0.25 in production

                    # evaluate the model on validation dataset
                    logger.info("Evaluating on the validation dataset...")
                    probs = []
                    for b in range(num_batches_valid):
                        probabilities = sess.run(valid_op)
                        probs.extend(probabilities.flatten().tolist())

                        if b % 100 == 0:
                            logger.info("Epoch step: {0} Train step: {1} Valid step: {2}".format(epoch_step,
                            batch, b))

                        #if b == 300:  # remove this break condition in production
                        #    break

                    evaluation_metric_new = utils.get_recall_values(probs)[0]  # returns a tuple of list with
                    # Recall@1,2 and 5 and model_responses
                    evaluation_metric_history.append(evaluation_metric_new[2])  # adds to the right side of the queue
                    logger.info("Epoch step: {0} Train step: {1} Valid step: {2} Evaluation_Metric = {3}".format(
                        epoch_step, batch, b, evaluation_metric_new))
                    logger.info("Epoch step: {0} Train step: {1} Valid step: {2} Evaluation_Metric_History = {3}".format(
                        epoch_step, batch, b, list(evaluation_metric_history)))

                    # save a model checkpoint if evaluated metric is better than the previous one
                    if evaluation_metric_new[2] > evaluation_metric_old:
                        best_steps = [epoch_step, batch]
                        logger.info("Epoch step: {0} Train step: {1} Saving checkpoint".format(epoch_step, batch))
                        saver.save(sess, './checkpoints/best_model.ckpt')
                        evaluation_metric_old = evaluation_metric_new[2]

                    # stop the training if the evaluation_metric_history is in an monotonically decreasing order
                    if utils.is_monotonically_decreasing(list(evaluation_metric_history)):
                        logger.info("Best model at Epoch step: {0} Train step: {1}".format(best_steps[0], best_steps[1]))
                        logger.info("Training completed.")
                        break

                if batch % num_batches_train == 0:  # completes an epoch examples/batch_size
                    logger.info("Completed epoch {0}".format(epoch_step))
                    batch = 0
                    epoch_step += 1

        except tf.errors.OutOfRangeError:
            logging.info('Done training for {0} epochs, {1} steps.'.format(epoch_step, batch))

        sess.close()


if __name__ == "__main__":
    """
    ip = build_input_pipeline('./dataset/train.tfrecords', 50, 1)
    data = ip.get_next()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(ip.initializer)
    data_res = sess.run(data)
    print(data_res[0], data_res[1], data_res[2])
    sess.close()
    """

    """
    #  load embedding matrix
    m = build_embedding_matrix(config.VOCAB_PROCESSOR, config.EMBED_FILE)
    print("m shape", m.shape)
    print(m)
    """
    train()
    #review = "It nicely predicted the conditioning of human minds with that of the patient. " \
     #       "We all believe that what we believe is true with our own point of view and we want " \
     #       "to solve all the problem accordingly."
    #infer(review)


