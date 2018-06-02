import tensorflow as tf
import logging
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('model')


class BiEncoderModel(object):

    def __init__(self):

        # hyper-parameters
        self.n_neurons = config.HIDDEN_SIZE
        self.learning_rate = 0.001

    def _get_next_batch(self):
        data = self.data_iterator.get_next()
        self.context_embedded, self.utterance_embedded = data[0], data[1]
        self.context_len, self.utterance_len, self.labels = data[2], data[3], data[4]

    def inference(self, data):
        self.context_embedded, self.utterance_embedded = data[0], data[1]
        self.context_len, self.utterance_len, self.labels = data[2], data[3], data[4]

        with tf.variable_scope('rnn_context'):
            cell_context = tf.nn.rnn_cell.LSTMCell(
                self.n_neurons,
                forget_bias=2.0,
                use_peepholes=True,
                state_is_tuple=True)

            # Run the utterance and context through the RNN
            outputs_contexts, encoding_context = tf.nn.dynamic_rnn(cell_context,
                                                                   self.context_embedded,
                                                                   dtype=tf.float32,
                                                                   sequence_length=self.context_len)
        with tf.variable_scope("rnn_response"):
            cell_response = tf.nn.rnn_cell.LSTMCell(
                self.n_neurons,
                forget_bias=2.0,
                use_peepholes=True,
                state_is_tuple=True)

            outputs_responses, encoding_utterance = tf.nn.dynamic_rnn(cell_response,
                                                                      self.utterance_embedded,
                                                                      dtype=tf.float32,
                                                                      sequence_length=self.utterance_len)

        encoding_context = encoding_context.h
        encoding_utterance = encoding_utterance.h
        print("context encoded shape: {0}, utterance encoded shape {1}".format(encoding_context.shape,
                                                                               encoding_utterance.shape)
              )
        M = tf.diag([1.0] * self.n_neurons)
        print ("Shape of M {}".format(M.shape))
        logits_list = []
        poly_kernel_pow_max = 3
        for poly_kernel_pow in range(1, poly_kernel_pow_max + 1):
            bias = tf.get_variable("B_" + str(poly_kernel_pow), shape=None,
                                    initializer=0.0)


            # "Predict" a  response: c * M
            generated_response = tf.matmul(encoding_context, M)
            #generated_response = tf.expand_dims(generated_response, 2)
            print ("Shape of gen res {}".format(generated_response.shape))
            #encoding_utterance = tf.expand_dims(encoding_utterance, 2)
            print ("Shape of enc utt {}".format(encoding_utterance.shape))

            # Dot product between generated response and actual response
            # (c * M) * r
            res1 = tf.reduce_sum(tf.multiply(generated_response, encoding_utterance), axis=1)
            res1 = tf.add(res1, bias)
            print ("res1 shape", res1.shape)
            logits = tf.pow(res1, poly_kernel_pow)
            logits = tf.add(logits, bias)
            logits_list.append(logits)


        logits = tf.add_n(logits_list)
        self.logits = tf.reshape(logits, [-1, 1])

        print ("Shape of logits at inference {}".format(self.logits.shape))
        return self.logits

    def create_loss(self):
        # logits and labels must have the shape (?, 1)
        logging.info("Shape of logits {0}".format(self.logits.shape))
        logits = tf.reshape(self.logits, [-1, 1])
        labels = tf.reshape(self.labels, [-1, 1])

        with tf.name_scope('loss'):
            self.losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(labels), logits=logits)

        return tf.reduce_mean(self.losses)

    def create_optimizer(self):
        logging.info("Shape of losses {0}".format(self.losses.shape))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(tf.reduce_mean(self.losses))
        return train_op

    def get_predictions(self, logits):
        probabilities = tf.sigmoid(logits)
        self.predicted_labels = tf.greater_equal(probabilities, 0.5)
        self.predicted_labels = tf.cast(self.predicted_labels, tf.int64)
        return self.predicted_labels

    def _create_accuracy(self):
        # labels and predicted labels must have the shape (?, 1)
        predicted_labels = tf.reshape(self.predicted_labels, [-1, 1])
        labels = tf.reshape(self.labels, [-1, 1])

        truth_values = tf.equal(predicted_labels, labels)
        truth_values = tf.cast(truth_values, tf.float64)
        accuracy = tf.reduce_mean(truth_values, axis=0)[0]

        return accuracy

    def get_validation_probabilities(self, logits):
        return tf.sigmoid(logits)
