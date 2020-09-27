import glob
import utils
import pickle
import numpy as np
import tensorflow as tf

PATH = glob.glob("/data/*.txt")

# HYPER PARAMETERS
EPOCHS = 10000
BATCH_SIZE = 512
RNN_SIZE = 512
LAYERS = 3
KEEP_PROBABILITY = 0.7
EMBED_DIM = 512
SEQUENCE_LENGTH = 30
LEARNING_RATE = 0.001
OUTPUT_DIR = './output'


corpus_raw = utils.combine_books(PATH)

## PRE PROCESSING ##
token_dict = utils.token_lookup()
for token, replacement in token_dict.items():
    corpus_raw = corpus_raw.replace(token, ' {} '.format(replacement))
corpus_raw = corpus_raw.lower()
corpus_raw = corpus_raw.split()

vocab_to_int, int_to_vocab = utils.create_lookup_tables(corpus_raw)
corpus_int = [vocab_to_int[word] for word in corpus_raw]
pickle.dump((corpus_int, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))

## BUILDING NETWORK ##

train_graph = tf.Graph()
with train_graph.as_default():
    input_text = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')

    vocab_size = len(int_to_vocab)
    input_text_shape = tf.shape(input_text)

    lstm = tf.contrib.rnn.BasicLSTMCell(num_units=RNN_SIZE)
    drop_cell = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=KEEP_PROBABILITY)
    cell = tf.contrib.rnn.MultiRNNCell([drop_cell] * LAYERS)

    initialize = cell.zero_state(input_text_shpae[0], tf.float32)
    intialize = tf.identity(initialize, name='intial_state')

    embed = tf.contrib.layers.embed_sequence(input_text, vocab_size, EMBED_DIM)

    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
    final_state = tf.identity(final_state, name='final_state')

    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)
    probs = tf.nn.softmax(logits, name='probs')

    cost = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones[input_text_shape[0], input_text_shape[1]])
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(g, -1., 1.) v) for g, v in gradients if grad is not None]
    trad_op = optimizer.apply_gradients(capped_gradients)

## TRAINING ##
