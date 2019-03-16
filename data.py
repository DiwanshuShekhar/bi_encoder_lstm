import os
import csv
import tensorflow as tf
import pandas as pd
import numpy as np
import functools

import h5py
from bilm import dump_bilm_embeddings

tf.flags.DEFINE_integer(
  "min_word_frequency", 5, "Minimum frequency of words in the vocabulary")

tf.flags.DEFINE_integer("max_sentence_len", 160, "Maximum Sentence Length")

tf.flags.DEFINE_string(
  "input_dir", os.path.abspath("./data"),
  "Input directory containing original CSV data files (default = './data')")

tf.flags.DEFINE_string(
  "output_dir", os.path.abspath("./data"),
  "Output directory for TFrEcord files (default = './data')")

FLAGS = tf.flags.FLAGS

TRAIN_PATH = os.path.join(FLAGS.input_dir, "train.csv")
VALIDATION_PATH = os.path.join(FLAGS.input_dir, "valid.csv")
TEST_PATH = os.path.join(FLAGS.input_dir, "test.csv")


def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)


def create_csv_iter(filename):
  """
  Returns an iterator over a CSV file. Skips the header.
  """
  with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    # Skip the header
    next(reader)
    for row in reader:
      yield row


def create_vocab(input_iter, min_frequency):
  """
  Creates and returns a VocabularyProcessor object with the vocabulary
  for the input iterator.
  """
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      FLAGS.max_sentence_len,
      min_frequency=min_frequency,
      tokenizer_fn=tokenizer_fn)
  vocab_processor.fit(input_iter)
  return vocab_processor


def transform_sentence(sequence, vocab_processor):
  """
  Maps a single sentence into the integer vocabulary. Returns a python array.
  """
  return next(vocab_processor.transform([sequence])).tolist()


def create_example_train(row, vocab):
  """
  Creates a training example for the Ubuntu Dialog Corpus dataset.
  Returnsthe a tensorflow.Example Protocol Buffer object.
  """
  context, utterance, label = row
  context_transformed = transform_sentence(context, vocab)
  utterance_transformed = transform_sentence(utterance, vocab)
  context_len = len(next(vocab._tokenizer([context])))
  utterance_len = len(next(vocab._tokenizer([utterance])))
  label = int(float(label))

  # New Example
  example = tf.train.Example()
  example.features.feature["context"].int64_list.value.extend(context_transformed)
  example.features.feature["utterance"].int64_list.value.extend(utterance_transformed)
  example.features.feature["context_len"].int64_list.value.extend([context_len])
  example.features.feature["utterance_len"].int64_list.value.extend([utterance_len])
  example.features.feature["label"].int64_list.value.extend([label])
  return example


def create_example_train_elmo(row):
  """
  Creates a training example for the Ubuntu Dialog Corpus dataset.
  Returnsthe a tensorflow.Example Protocol Buffer object.
  """
  context, utterance, label = row
  context_tokenized = context.strip().split()
  utterance_tokenized = utterance.strip().split()
  context_len = len(context_tokenized)
  utterance_len = len(utterance_tokenized)
  label = int(float(label))

  # New Example
  example = tf.train.Example()
  example.features.feature["context"].string_list.value.extend(context_tokenized)
  example.features.feature["utterance"].string_list.value.extend(utterance_tokenized)
  example.features.feature["context_len"].int64_list.value.extend([context_len])
  example.features.feature["utterance_len"].int64_list.value.extend([utterance_len])
  example.features.feature["label"].int64_list.value.extend([label])
  return example


def create_example_test(row, vocab):
  """
  Creates a test/validation example for the Ubuntu Dialog Corpus dataset.
  Returns a tensorflow.Example Protocol Buffer object.
  """
  context, utterance = row[:2]
  distractors = row[2:]
  context_len = len(next(vocab._tokenizer([context])))
  utterance_len = len(next(vocab._tokenizer([utterance])))
  context_transformed = transform_sentence(context, vocab)
  utterance_transformed = transform_sentence(utterance, vocab)

  examples = []

  # New Good Example
  example = tf.train.Example()
  example.features.feature["context"].int64_list.value.extend(context_transformed)
  example.features.feature["utterance"].int64_list.value.extend(utterance_transformed)
  example.features.feature["context_len"].int64_list.value.extend([context_len])
  example.features.feature["utterance_len"].int64_list.value.extend([utterance_len])
  example.features.feature["label"].int64_list.value.extend([1])

  examples.append(example)

  # Distractor sequences
  for i, distractor in enumerate(distractors):
    # Distractor Length Feature
    dis_len = len(next(vocab._tokenizer([distractor])))

    # Distractor Text Feature
    dis_transformed = transform_sentence(distractor, vocab)

    example = tf.train.Example()
    example.features.feature["context"].int64_list.value.extend(context_transformed)
    example.features.feature["utterance"].int64_list.value.extend(dis_transformed)
    example.features.feature["context_len"].int64_list.value.extend([context_len])
    example.features.feature["utterance_len"].int64_list.value.extend([dis_len])
    example.features.feature["label"].int64_list.value.extend([0])

    examples.append(example)

  return examples


def create_tfrecords_file(input_filename, output_filename, example_fn):
  """
  Creates a TFRecords file for the given input data and
  example transofmration function
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  print("Creating TFRecords file at {}...".format(output_filename))
  for i, row in enumerate(create_csv_iter(input_filename)):
    x = example_fn(row)  # check if x is one single example or a list of examples
    if isinstance(x, list):
        for e in x:
            writer.write(e.SerializeToString())
    else:
        writer.write(x.SerializeToString())
  writer.close()
  print("Wrote to {}".format(output_filename))


def write_vocabulary(vocab_processor, outfile):
  """
  Writes the vocabulary to a file, one word per line.
  """
  vocab_size = len(vocab_processor.vocabulary_)
  with open(outfile, "w") as vocabfile:
    for id in range(vocab_size):
      word =  vocab_processor.vocabulary_._reverse_mapping[id]
      vocabfile.write(word + "\n")
  print("Saved vocabulary to {}".format(outfile))


def get_length_stats(it):
  """

  :param it: iterator
  :return:
  """
  context_length = []
  response_length = []
  for row in it:
      context_length.append(len(row[0].split(" ")))
      response_length.append(len(row[1].split(" ")))

  print(context_length[:10])
  print(response_length[:10])
  data = np.array([context_length, response_length]).T
  print(data)

  df = pd.DataFrame(data=data, columns=['con_len', 'res_len'])
  print(df.head(10))
  return df.describe(percentiles=[0.3, 0.6, 0.9])


def dump_train_dataset_for_elmo(input_iter, output_path):
    with open(output_path, 'w') as fh:
        for row in input_iter:
            fh.write(row + "\n")


def dump_hdf5_elmo(vocab_file, dataset_file, options_file,
                   weight_file, embedding_file="elmo_embeddings.hdf5"):

    dump_bilm_embeddings(vocab_file, dataset_file, options_file, weight_file, embedding_file)


def dump_elmo_embedding(elmo_embedding_hdf5, elmo_dataset, output_file):

    sentence_to_tokens = {}
    word_to_embedding = {}

    with open(elmo_dataset, 'r') as fh:
        for idx, line in enumerate(fh):
            sentence_to_tokens[str(idx)] = line.split()

    with h5py.File(elmo_embedding_hdf5, 'r') as fh:
        for idx, _ in enumerate(fh):
            embedding_all = fh[str(idx)]
            weighted_embedding = np.sum(embedding_all, axis=0)
            print(weighted_embedding.shape)
            print(len(sentence_to_tokens[str(idx)]))

            for idx, word in enumerate(sentence_to_tokens[str(idx)]):
                word_to_embedding[word] = list(weighted_embedding[idx])

    print(word_to_embedding)

    with open(output_file, 'w') as fh:
        for k, v in word_to_embedding.items():
            value_str = " ".join([str(e) for e in v])
            fh.write(k + " " + value_str + "\n")


def modify_validation_format_to_train_format(itr):
    """
    Modify valid.csv of UDCv2 to match train.csv format
    train.csv contains context, utterance and label
    valid.csv contains Context,Ground Truth Utterance,
    Distractor_0,Distractor_1,Distractor_2,Distractor_3,
    Distractor_4,Distractor_5,Distractor_6,Distractor_7,Distractor_8
    Return an iterator such that one row from original valid.txt
    generates 10 rows:
    Context, Ground Truth Utterance, 1

    :param itr:
    :return:
    """
    for row in itr:
        yield (row[0], row[1], 1)
        yield (row[0], row[2], 0)
        yield (row[0], row[3], 0)
        yield (row[0], row[4], 0)
        yield (row[0], row[5], 0)
        yield (row[0], row[6], 0)
        yield (row[0], row[7], 0)
        yield (row[0], row[8], 0)
        yield (row[0], row[9], 0)
        yield (row[0], row[10], 0)


def itr_to_csv(itr, output_file_name):
    with open(output_file_name, 'w', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerows(itr)

if __name__ == "__main__":
  """
  print("Creating vocabulary...")
  input_iter = create_csv_iter(TRAIN_PATH)
  input_iter = (x[0] + " " + x[1] for x in input_iter)
  vocab = create_vocab(input_iter, min_frequency=FLAGS.min_word_frequency)
  print("Total vocabulary size: {}".format(len(vocab.vocabulary_)))

  # Create vocabulary.txt file
  #write_vocabulary(
  #  vocab, os.path.join(FLAGS.output_dir, "vocabulary.txt"))

  # Save vocab processor
  #vocab.save(os.path.join(FLAGS.output_dir, "vocab_processor.bin"))

  # Create validation.tfrecords
  create_tfrecords_file(
      input_filename=VALIDATION_PATH,
      output_filename=os.path.join(FLAGS.output_dir, "validation_v1.tfrecords"),
      example_fn=functools.partial(create_example_test, vocab=vocab))

  # Create test.tfrecords
  create_tfrecords_file(
      input_filename=TEST_PATH,
      output_filename=os.path.join(FLAGS.output_dir, "test_v1.tfrecords"),
      example_fn=functools.partial(create_example_test, vocab=vocab))

  # Create train.tfrecords
  #create_tfrecords_file(
  #    input_filename=TRAIN_PATH,
  #    output_filename=os.path.join(FLAGS.output_dir, "train.tfrecords"),
  #    example_fn=functools.partial(create_example_train, vocab=vocab))
  """

  """
  input_iter = create_csv_iter(TRAIN_PATH)
  df = get_length_stats(input_iter)
  print(df)
  """
  #input_iter = create_csv_iter(TRAIN_PATH)
  #input_iter = (x[0] + " " + x[1] for x in input_iter)
  #dump_train_dataset_for_elmo(input_iter, 'data/train_dataset_elmo.txt')
  #dump_hdf5_elmo('data/vocabulary.txt', 'data/train_dataset_elmo_sample.txt',
  #               'data/options.json', 'data/lm_weights.hdf5', 'data/train_dataset_elmo.hdf5')

  #dump_elmo_embedding('data/train_dataset_elmo.hdf5', 'data/train_dataset_elmo_sample.txt', 'data/elmo.91K.32d_udcv2.txt')

  # Create train.tfrecords
  #create_tfrecords_file(
  #    input_filename=TRAIN_PATH,
  #    output_filename=os.path.join(FLAGS.output_dir, "train_elmo.tfrecords"),
  #    example_fn=functools.partial(create_example_train_elmo))

  #  modify valid.csv to match train.csv format
  itr = modify_validation_format_to_train_format(create_csv_iter('data/valid.csv'))
  itr_to_csv(itr, 'data/valid_reformatted.csv')





