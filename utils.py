import cPickle
import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('utils')


def get_vocab(vocab_file):
    """
    :param vocab_file: string path to a json file of word and its id
    :return: dict of word and id
    """
    with open(vocab_file, 'r') as fh:
        vocabulary = json.load(fh)

    return vocabulary


def get_ids_from_string(review, max_sentence_length, vocabulary):
    """
    converts a user provided review to a list of ids
    :param review: string
    :return: tuple of list of ids and length of user provided review
    """
    ids = [1] * max_sentence_length
    words = review.split()
    for i, word in enumerate(words):
        if i > max_sentence_length - 1:
            break
        ids[i] = vocabulary.get(word, 0)
    return ids, len(words)


def build_embedding_matrix(embeb_file,
                           vocabulary=None,
                           embed_len=None):
    """
    :param embed_file: string path to embedding file
    :param vocabulary: dictionary of word and integer id
    :return: tensor shape = [MAX_SENTENCE_LEN, embedding_dimension]
    """
    embeddings_mat = np.random.uniform(-0.25, 0.25, (len(vocabulary), embed_len)).astype("float32")

    embed_dict = {}
    with open(embeb_file, 'r') as fh:
        for line in fh:
            tokens = line.split(" ")
            embed_word, embed_vector = tokens[0], tokens[1:]
            embed_dict[embed_word] = embed_vector

    for word, id in vocabulary.items():
        if word in embed_dict:
            embeddings_mat[id] = embed_dict[word]

    del embed_dict
    return embeddings_mat


def is_monotonically_decreasing(a_list):
    prev_item = a_list[0]
    for idx, item in enumerate(a_list):
        if idx == 0:
            continue

        if prev_item < item:
            return False

        prev_item = item

    return True


def recall_at_k(k, probabilities_matrix):

    # logging.info("Given probabilities_matrix {}".format(probabilities_matrix))
    index_matrix = np.argsort(probabilities_matrix, axis=1)  # index_matrix sorts the input matrix in ascending order
    # logging.info("Given index_matrix {}".format(index_matrix))

    def my_func(array_slice):
        array_slice = array_slice[::-1]  # reverses the array
        if 0 in array_slice[:k]:
            return True
        else:
            return False

    bool_array = np.apply_along_axis(my_func, 1, index_matrix)
    return np.mean(bool_array), bool_array.astype(int)


def get_recall_values(probabilities_list, k=[1, 2, 5]):
    a = np.array(probabilities_list)
    cols = 10
    a = a.reshape((-1, cols))
    logging.info("Reshaped the probabilities list to {}".format(a.shape))
    recalls = []
    example_predictions = []
    for i in k:
        results = recall_at_k(i, a)
        recalls.append(results[0])
        example_predictions.append(results[1])
        print_to_file(results[1], 'recall_{}.txt'.format(i))
    return recalls, example_predictions


def print_to_file(example_prediction, file_name):
    """
    example_prediction is an ndarray of 0 and 1
    0 means the model did not predict the example within the given recall@k
    1 means the model predicted the example within the given recall@k
    the resultant file will be used for error analysis of the model
    """
    pred_list = example_prediction.tolist()
    with open(file_name, 'w') as fh:
        fh.write(','.join(str(e) for e in pred_list) + '\n')


if __name__ == "__main__":
    print(is_monotonically_decreasing([10, 9, 8, 7, 6]))  # True
    print(is_monotonically_decreasing([10, 9, 3, 7, 6]))  # False
    print(is_monotonically_decreasing([10, 10, 10, 10, 10]))  # True
    print(is_monotonically_decreasing([10, 10, 9, 7, 10]))  # False

    a1 = [0.2, 0.3, 0.1, 0.4, 0.7, 0.8, 0.3, 0.67, 0.64, 0.15]
    results = get_recall_values(a1)
    print("recalls ", results[0])
    print("examples ", results[1])

    prob_mat = np.random.rand(100, 10)
    result = get_recall_values(prob_mat)
    print("new recall", result)


