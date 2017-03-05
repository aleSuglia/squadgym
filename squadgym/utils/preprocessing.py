import string

import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check "trunc" has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def preprocess_text(text, language="english", lower=True):
    return [token.lower() if lower else token for token in word_tokenize(text, language)]


def preprocess_question(query, language="english", lower=True):
    return [token.lower() if lower else token for token in query
            if token not in stopwords.words(language) and token not in string.punctuation]


def tokens2ids(tokens, token2id):
    return [token2id[token] for token in tokens]


def ids2tokens(ids, id2token, padding_value=None):
    if padding_value is not None:
        return [id2token[id] for id in ids if id != padding_value]
    return [id2token[id] for id in ids]
