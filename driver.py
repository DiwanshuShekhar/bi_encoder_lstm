import config
import utils
import tensorflow as tf
import os
import json
import sys
import csv

try:
    from .model import BiEncoderModel
    #from .model_19 import BiEncoderModel
    #from .model_20 import BiEncoderModel
    #from .model_21 import BiEncoderModel
    #from .model_22 import BiEncoderModel
    #from .model_23 import BiEncoderModel
    #from .model_24 import BiEncoderModel
    #from .model_rnn import BiEncoderModel
    #from .model_dual import BiEncoderModel
    #from .model_dual_rnn import BiEncoderModel
except:
    from model import BiEncoderModel
    #from model_19 import BiEncoderModel
    #from model_20 import BiEncoderModel
    #from model_21 import BiEncoderModel
    #from model_22 import BiEncoderModel
    #from model_23 import BiEncoderModel
    #from model_24 import BiEncoderModel
    #from model_rnn import BiEncoderModel
    #from model_dual import BiEncoderModel
    #from model_dual_rnn import BiEncoderModel

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_csv_iter(filename_ndarray):
  """
  Returns an iterator over a CSV file. Skips the header.
  """
  filename = filename_ndarray[0].decode()
  with open(os.path.join(BASE_DIR, filename)) as csvfile:
    reader = csv.reader(csvfile)
    # Skip the header
    next(reader)
    for row in reader:
      yield row[0], row[1], row[2]


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
                                                             embed_len=config.EMBED_LEN,
                                                             random=False)  # True for random embedding
            embeddings_mat = tf.get_variable("word_embeddings", trainable=False, initializer=embeddings_matrix)
    print("shape of embedding matrix", embeddings_mat.shape)

    features['context'] = tf.nn.embedding_lookup(embeddings_mat, features['context'])
    features['utterance'] = tf.nn.embedding_lookup(embeddings_mat, features['utterance'])
    print("shape of train context", features['context'].shape)  # should be [max_sentence_length, word_embed_size]
    print("shape of train context len", features['context_len'].shape)
    return features['context'], features['utterance'], features['context_len'], features['utterance_len'], features['label']


def parse_input_elmo(context, utterance, label):
    """
    features = tf.parse_single_example(example, features={
        'context': tf.FixedLenFeature([config.MAX_SENTENCE_LEN], tf.int64),
        'utterance': tf.FixedLenFeature([config.MAX_SENTENCE_LEN], tf.int64),
        'context_len': tf.FixedLenFeature([], tf.int64),
        'utterance_len': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    })

    bilm_vocab_object = utils.build_bilm_vocab(os.path.join(BASE_DIR, config.TOKEN), 20)

    tokenized_context = bilm_vocab_object.decode(features['context'].tolist()).split()
    tokenized_utterance = bilm_vocab_object.decode(features['utterance'].tolist()).split()
    """

    def strip_and_split(sentence):
        if sys.version_info[0] != 2:
            sentence = sentence.decode()

        sentence_tokenized = sentence.strip().split()
        sentence_length = len(sentence_tokenized)
        return sentence_tokenized, sentence_length

    #  context
    context_tokenized, context_tokenized_len = tf.py_func(strip_and_split, [context], [tf.string, tf.int64])
    context_embedded = tf.py_func(utils.get_bilm_embedding,
                       [os.path.join(BASE_DIR, 'data/options_test.json'),
                       os.path.join(BASE_DIR, 'data/lm_weights_test.hdf5'),
                       50, context_tokenized], tf.float32)
    context_embedded.set_shape([20, 32])  # ToDo: match shape to bilm embedding
    #context_embedded = tf.reshape(context_embedded, [None, 32])
    context_tokenized_len.set_shape([])

    #  utterance
    utterance_tokenized, utterance_tokenized_len = tf.py_func(strip_and_split, [utterance], [tf.string, tf.int64])
    utterance_embedded = tf.py_func(utils.get_bilm_embedding,
                       [os.path.join(BASE_DIR, 'data/options_test.json'),
                       os.path.join(BASE_DIR, 'data/lm_weights_test.hdf5'),
                       50, utterance_tokenized], tf.float32)
    utterance_embedded.set_shape([20, 32])  # ToDo: match shape to bilm embedding
    #utterance_embedded = tf.reshape(utterance_embedded, [None, 32])
    utterance_tokenized_len.set_shape([])

    return context_embedded, utterance_embedded, context_tokenized_len, utterance_tokenized_len, tf.cast(label, tf.int64)


def build_input_pipeline(in_files, batch_size, num_epochs=None, mode='train', use_elmo=False):
    """
    Build an input pipeline with the DataSet API
    :param in_files list of tfrecords filenames
    :return dataset iterator (use get_next() method to get the next batch of data from the dataset iterator
    """
    if use_elmo:
        dataset = tf.data.Dataset.from_generator(create_csv_iter,
                                                 (tf.string, tf.string, tf.int64),
                                                 args=[tf.constant(in_files, tf.string)])
    else:
        dataset = tf.data.TFRecordDataset(in_files)

    if use_elmo:
        dataset = dataset.map(parse_input_elmo, num_parallel_calls=12)
    else:
        dataset = dataset.map(parse_input, num_parallel_calls=12)

    if mode is 'train':  # we only want to shuffle for training dataset
        dataset = dataset.shuffle(buffer_size=4 * batch_size)

    dataset = dataset.batch(batch_size)

    if num_epochs:
        dataset = dataset.repeat(num_epochs)
    else:
        dataset = dataset.repeat()  # Repeat the input indefinitely.
    #iterator = dataset.make_initializable_iterator()
    #return iterator
    return dataset


def train():
    """
    Builds the graph and runs the graph in a session
    :return:
    """
    train_files = ['data/train_fake.csv']
    validation_files = ['data/valid_fake.csv']

    with tf.Graph().as_default():

        logging.info("Building train input pipeline")
        training_dataset = build_input_pipeline(train_files,
                                                10,
                                                num_epochs=None,
                                                mode='train',  # change num_epochs to None in production
                                                use_elmo=True)

        logging.info("Building validation input pipeline")
        validation_dataset = build_input_pipeline(validation_files,
                                                  10,
                                                  mode='valid',
                                                  use_elmo=True)

        # A feedable iterator is defined by a handle placeholder and its structure. We
        # could use the `output_types` and `output_shapes` properties of either
        # `training_dataset` or `validation_dataset` here, because they have
        # identical structure.
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle, training_dataset.output_types, training_dataset.output_shapes)
        next_batch = iterator.get_next()
        print(next_batch)

        # You can use feedable iterators with a variety of different kinds of iterator
        # (such as one-shot and initializable iterators).
        training_iterator = training_dataset.make_initializable_iterator()
        validation_iterator = validation_dataset.make_initializable_iterator()

        model = BiEncoderModel()

        logging.info("Building graph")
        logits = model.inference(next_batch)
        tf.add_to_collection('logits_tensor', logits)
        loss_op = model.create_loss()
        train_op = model.create_optimizer()  # for training

        # other ops for visualization, evaluation etc
        probabilities_op = model.get_validation_probabilities(logits)

        sess_conf = tf.ConfigProto()
        sess_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_conf)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(training_iterator.initializer)
        sess.run(validation_iterator.initializer)
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

        # adding summaries
        tf.summary.scalar('cross_entropy_loss', loss_op)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('checkpoints/train', sess.graph)
        validation_writer = tf.summary.FileWriter('checkpoints/validation', sess.graph)

        saver = tf.train.Saver(max_to_keep=1)

        #  starting the training
        logging.info("Training starts...")
        batch = 0
        epoch_step = 0
        num_batches_train = int(config.NUM_EXAMPLES_TRAIN/config.TRAIN_BATCH_SIZE)
        num_batches_valid = int(config.NUM_EXAMPLES_VALID/config.VALIDATION_BATCH_SIZE)
        evaluation_metric_old = 0.0
        early_stop_counter = 0
        global_step_counter = 0
        try:
            while True:
                # here calculate accuracy and/or training loss
                op_result, loss, summary = sess.run([train_op, loss_op, merged], feed_dict={handle: training_handle})

                if batch % 100 == 0:
                    logger.info("Epoch step: {0} Train step: {1} loss = {2}".format(epoch_step, batch, loss))
                    train_writer.add_summary(summary, global_step_counter)

                batch += 1

                # evaluate the model with the validation set every 1/4 of the epoch
                if batch % int(0.25 * num_batches_train) == 0:  # change 0.01 to 0.25 in production

                    # evaluate the model on validation dataset
                    logger.info("Evaluating on the validation dataset...")
                    probabilities = []
                    losses = []
                    for b in range(num_batches_valid):
                        probabilities_batch, loss_batch = sess.run([probabilities_op, loss_op],
                                                                   feed_dict={handle: validation_handle})

                        probabilities.extend(probabilities_batch.flatten().tolist())
                        losses.append(loss_batch)

                        if b % 100 == 0:
                            logger.info("Epoch step: {0} Train step: {1} Valid step: {2}".format(epoch_step,
                            batch, b))

                        #if b == 300:  # remove this break condition in production
                        #   break

                    evaluation_metric_new = utils.get_recall_values(probabilities)[0]  # returns a tuple of list with
                    # Recall@1,2 and 5 and model_responses
                    logger.info("Epoch step: {0} Train step: {1} Valid step: {2} Evaluation_Metric = {3}".format(
                        epoch_step, batch, b, evaluation_metric_new))
                    for i, k in enumerate([1,2,5]):
                        summary = tf.Summary()
                        summary.value.add(tag='recall_{}'.format(k), simple_value=evaluation_metric_new[i])
                        validation_writer.add_summary(summary, global_step_counter)

                    average_loss_validation = float(sum(losses))/len(losses)
                    summary = tf.Summary()
                    summary.value.add(tag="cross_entropy_loss", simple_value=average_loss_validation)
                    validation_writer.add_summary(summary, global_step_counter)

                    # save a model checkpoint if the new evaluated metric is better than the previous one
                    if evaluation_metric_new[2] > evaluation_metric_old:
                        best_steps = [epoch_step, batch]
                        logger.info("Epoch step: {0} Train step: {1} Saving checkpoint".format(epoch_step, batch))
                        path = os.path.join(config.CHECKPOINT_PATH, 'model_{0}_{1}.ckpt'.format(epoch_step, batch))
                        saver.save(sess, path)
                        evaluation_metric_old = evaluation_metric_new[2]
                        early_stop_counter = 0
                    else:
                        early_stop_counter = early_stop_counter + 1

                    # stop the training if the early_stop_counter > 10
                    if early_stop_counter > 10:
                        logger.info("Best model at Epoch step: {0} Train step: {1}".format(best_steps[0], best_steps[1]))
                        logger.info("Training completed.")
                        break

                if batch % num_batches_train == 0:  # completes an epoch examples/batch_size
                    logger.info("Completed epoch {0}".format(epoch_step))
                    batch = 0
                    epoch_step += 1

                global_step_counter = global_step_counter + 1

        except tf.errors.OutOfRangeError:
            logging.info('Done training for {0} epochs, {1} steps.'.format(epoch_step, batch))

        sess.close()


def test(checkpoint_file):
    """
    """
    test_files = config.TEST_FILES

    with tf.Graph().as_default():
        logging.info("Building test input pipeline")
        test_dataset = build_input_pipeline(test_files,
                                            config.TEST_BATCH_SIZE,
                                            num_epochs=1,
                                            mode='valid',
                                            use_elmo=False)

        test_iterator = test_dataset.make_initializable_iterator()
        next_batch = test_iterator.get_next()

        model = BiEncoderModel()

        logging.info("Building graph")
        logits = model.inference(next_batch)

        # other ops for visualization, evaluation etc
        probabilities_op = model.get_validation_probabilities(logits)

        sess_conf = tf.ConfigProto()
        sess_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_conf)

        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)

        logging.info("Uninitialized variables: {}".format(sess.run(tf.report_uninitialized_variables())))
        sess.run(test_iterator.initializer)

        #  starting model evaluation on test set
        logging.info("Evaluation starts...")
        num_batches_test = int(config.NUM_EXAMPLES_TEST/config.TEST_BATCH_SIZE)
        probabilities = []

        for b in range(num_batches_test):
            probabilities_batch = sess.run(probabilities_op)
            probabilities.extend(probabilities_batch.flatten().tolist())

            if b % 100 == 0:
                logger.info("Test step: {}".format(b))

                #if b == 300:  # remove this break condition in production
                #   break

        evaluation_metric = utils.get_recall_values(probabilities)  # returns a tuple of list with
        # Recall@1,2 and 5 and model_responses
        logger.info("Evaluation_Metric = {0}".format(evaluation_metric[0]))
        logger.info("Model prediction on examples {}".format(evaluation_metric[1]))

        sess.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage_text = """Usage:
                        To train:
                        python driver.py --train
                        To test:
                        python driver.py --test checkpoints/my_model.ckpt"""
        print(usage_text)

    if sys.argv[1] == '--train':
        train()

    if sys.argv[1] == '--test':
        test(sys.argv[2])


