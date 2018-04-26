MAX_SENTENCE_LEN = 160
EMBED_LEN = 200
EMBED_FILE = 'data/glove.6B.200d.txt' #  also change the EMBED_LEN when changing this
TRAIN_BATCH_SIZE = 256
VALIDATION_BATCH_SIZE = 10
TRAIN_FILES = ['data/train.tfrecords']
VALIDATION_FILES = ['data/validation_v1.tfrecords']
NUM_EPOCHS = 2
CHECKPOINT_FILE = 'checkpoints/390.ckpt'
VOCABULARY = 'data/vocabulary.json'