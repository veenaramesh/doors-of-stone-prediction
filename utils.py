import numpy as np
import tensorflow as tf
import codecs

def combine_books(path):
    book_filenames = sorted(path)
    corpus_raw = u""
    for filename in book_filenames:
        with codecs.open(filename, 'r', 'utf-8') as book_file:
            corpus_raw += book_file.read()

    return corpus_raw

def create_lookup_tables(txt):
    vocab = set(txt)
    int_to_vocab = {key: word for key, word in enumerate(vocab)}
    vocab_to_int = {word: key for key, word in enumerate(vocab)}
    return vocab_to_int, int_to_vocab

def token_lookup():
    return {
        '.': '||period||',
        ',': '||comma||',
        '"': '||quotes||',
        ';': '||semicolon||',
        '!': '||exclamation-mark||',
        '?': '||question-mark||',
        '(': '||left-parentheses||',
        ')': '||right-parentheses||',
        '--': '||emm-dash||',
        '\n': '||return||'
    }

def get_batches(int_text, batch_size, sequence_length):
    words_per_batch = batch_size * sequence_length
    num_batches = len(int_text)//words_per_batch
    int_text = int_text[:num_batches * words_per_batch]
    y = np.array(int_text[1:] + int_text[0])
    x = np.array(int_text)

    y_batches = np.split(y.reshape(batch_size, -1), num_batches, axis=1)
    x_batches = np.split(x.reshape(batch_size, -1), num_batches, axis=1)

    batch_data = list(zip(x_batches, y_batches))

    return np.array(batch_data)
