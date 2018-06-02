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

        def process_hidden_states(outputs, seq_len, batch_size, max_time):
           new_outputs = []
           for i in range(batch_size):
               if seq_len[i] is not tf.to_int64(max_time):
                   states_non_zero = outputs[i, :tf.to_int32(seq_len[i]), :]
                   states_zero = outputs[i, tf.to_int32(seq_len[i]):, :]
                   states = tf.concat([states_zero, states_non_zero], 0)
                   new_outputs.append(states)
               else:
                   states = outputs[i]
                   new_outputs.append(states)
           return tf.stack(new_outputs, axis=0)

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
                                                                   sequence_length=self.context_len,
                                                                   time_major=False)
        with tf.variable_scope("rnn_response"):
            cell_response = tf.nn.rnn_cell.LSTMCell(
                self.n_neurons,
                forget_bias=2.0,
                use_peepholes=True,
                state_is_tuple=True)

            outputs_responses, encoding_utterance = tf.nn.dynamic_rnn(cell_response,
                                                                      self.utterance_embedded,
                                                                      dtype=tf.float32,
                                                                      sequence_length=self.utterance_len,
                                                                      time_major=False)

        # process the hidden states so that it doesn't have zero states a max_time
        outputs_contexts = process_hidden_states(outputs_contexts, self.context_len,
                                                 config.TRAIN_BATCH_SIZE, 160)
        outputs_responses = process_hidden_states(outputs_responses, self.utterance_len,
                                                  config.TRAIN_BATCH_SIZE , 160)

        # reshape the hidden states from [batch_size, max_time, embed_size] to
        # [max_time, batch_size, embed_size]
        outputs_contexts = tf.transpose(outputs_contexts, perm=[1, 0, 2])
        print ("outputs contexts shape ".format(outputs_contexts.shape))
        outputs_responses = tf.transpose(outputs_responses, perm=[1, 0, 2])
        print ("outputs responses shape ".format(outputs_responses.shape))


        with tf.variable_scope("trainable_parameters"):
            bias = tf.get_variable("B", shape=None, trainable=True, initializer=0.0)

        alpha = []
        for i in range(160):
            #  to check the sanity of the implementation
            #if i + 1 == self.context:
            #    alpha.append([0.0] * config.TRAIN_BATCH_SIZE)
            #elif i + 1 == self.context_len - 1:
            #    alpha.append([1.0] * config.TRAIN_BATCH_SIZE)
            #elif i + 1 == config.context_len - 2:
            #    alpha.append([0.0] * config.TRAIN_BATCH_SIZE)
            #else:
            #    alpha.append([0.0] * config.TRAIN_BATCH_SIZE)
            alpha.append([(i + 1.0)**2 / float(160 * 160)] * config.TRAIN_BATCH_SIZE)

        alpha = tf.constant(alpha)
        print ("shape of alpha", alpha.shape) # shape: [max_time, batch_size]

        res1 = tf.multiply(outputs_contexts, outputs_responses)   # will multiply element-wise
        res2 = tf.reduce_sum(res1, axis=2)  # will sum the third dimension
        print ("shape of res2", res2.shape) # [max_time, batch_size]
        res3 = tf.multiply(res2, alpha)  # will multiply element-wise with scalar matrix
        res4 = tf.reduce_sum(res3, axis=0)  # will sum the first dimension

        logits = tf.reshape(res4, [-1, 1])
        logits = tf.add(logits, bias)

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
