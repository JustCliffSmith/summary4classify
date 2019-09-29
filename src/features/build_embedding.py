# -*- coding: utf-8 -*-
import sys
import logging
import pickle
from pathlib import Path
import operator

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from features.build_features import load_raw_data
from pipeline.utility import pack_data_dict, unpack_data_dict

def load_glove_embeddings_index(word_index, embed_path):
    """ Load glove embedding from txt or pkl file and create embedding matrix.

    Keyword arguments:
    word_index - word_index from keras Tokenizer
    embed_path - path to embedding file, either as a .txt or .pkl
    
    Returns:
    embeddings_index - embeddings index
    """
   
    if embed_path.endswith('.txt'):
        print("Loading glove embedding from .txt file.")
        def get_coefs(word, *arr): 
            return word, np.asarray(arr, dtype='float32')[:300]
        embeddings_index = dict((get_coefs(*o.split(" ")) for o in open(embed_path)))
        print('Dumping pickled copy of embeddings indices for faster loading.')
        pickle_path = embed_path[:-3] + 'pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(embeddings_index, f, pickle.HIGHEST_PROTOCOL)
        return embeddings_index

    if embed_path.endswith('.pkl'):
        logging.info("Loading glove embedding from .pkl file.")
        with open(embed_path, 'rb') as f:    
            embeddings_index = pickle.load(f)
        return embeddings_index

    if not (embed_path.endswith('.txt') or embed_path.endswith('.pkl')):
        logging.error("Embedding path must end in .txt or .pkl")
        #raise Exception
        import sys
        sys.exit() 


def create_embedding_matrix(embeddings_index, max_features, word_index):
    """ Creates embedding matrix from embeddings index.
    
    Creates embedding matrix from embeddings index. If word is not found 
    then tries lowercase and then Titlecase version of the word.

    Keyword arguments:
    embeddings_index - embeddings index

    Returns:
    embedding_matrix - matrix of embeddings, dim=(num_words, embed_size)
    """
    logging.info("Computing mean and std of embedding indices.")
    print("Computing mean and std of embedding indices.")
    # Generate stats on embedding so OOV words do not skew mean and std.
    #all_embs = np.stack(embeddings_index.values())
    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean = np.mean(all_embs)
    emb_std = np.std(all_embs)
    logging.info(emb_mean)
    print(emb_mean)
    logging.info(emb_std)
    print(emb_std)
    # Hardcoded mean and std if I wanted to be lazy and not calculate it.
    #emb_mean, emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]

    num_words = min(max_features, len(word_index))
    # Instantiate embedding_matrix with random weights so OOV words
    # match the stats of the rest of the words.
    embedding_matrix = np.random.normal(emb_mean, 
                                        emb_std, 
                                        (num_words, embed_size))
    logging.info("Populating embedding matrix.")
    print("Populating embedding matrix.")
    for word, i in word_index.items():
        if i >= max_features: 
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.lower())
            if embedding_vector is None:
                embedding_vector = embeddings_index.get(word.title())
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
# Which approach is faster?
#    for word, i in word_index.items():
#        if i <= max_features:
#            try:
#                embedding_matrix[i] = embedding_index[word]
#            except KeyError:
#                try:
#                    embedding_matrix[i] = embedding_index[word.lower()]
#                except KeyError:
#                    try:
#                        embedding_matrix[i] = embedding_index[word.title()]
#                    except KeyError:
#                        pass
            
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    return embedding_matrix 


def build_vocab(texts):
    """ Create a dictionary

    Keyword arguments:
    texts - pandas series containing the corpus.

    Returns:
    vocab - dict, keys are words and values are number of occurences.
    """
    try:
        sentences = texts.apply(lambda x: x.split()).values
    except Exception:
        print(f"Text corpus must be in pandas series but is {type(texts)}")
        #sys.exit()
    vocab = {}
    print("Building vocabulary from the entire text corpus.")
    #for sentence in sentences:
    for sentence in texts:
        for word in sentence:
            vocab.setdefault(word, 0)
            vocab[word] += 1            
    return vocab


def check_coverage(vocab, embeddings_index):
    """ Checks percentage of vocabularly in pre-trained embedding. 

    Loops through all words in vocabularly to determine what percentage are
    in the pre-trained embedding. Percentages are printed.
    Lightly modified from Dieter's work on Kaggle.
    
    Keyword Arguments: 
    vocab - dictionary of vocabularly counts 
    embedding_index -  
    
    Return: 
    unknown_words = words not in embedding.
    """
    known_words = {}
    unknown_words = {}
    num_known_words = 0
    num_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            num_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            num_unknown_words += vocab[word]
            pass

    percentage_vocab = len(known_words) / len(vocab)
    percentage_alltext = num_known_words / (num_known_words + num_unknown_words)
    logging.info(f"Found embeddings for {100*percentage_vocab:.3f}% of vocab")
    print(f"Found embeddings for {100*percentage_vocab:.3f}% of vocab")
    logging.info(f"Found embeddings for  {100*percentage_alltext:.3f}% of all text")
    print(f"Found embeddings for  {100*percentage_alltext:.3f}% of all text")
    unknown_words = sorted(unknown_words.items(), 
                           key=operator.itemgetter(1))[::-1]
    return unknown_words


def main(project_dir):
    """ 
    """
    logger = logging.getLogger(__name__)
    data = load_raw_data(project_dir)
    X_train, y_train, X_test, y_test = unpack_data_dict(data)

    # Parameters for embedding.
    maxlen = 220
    #max_features = 100000
    max_features = 436356
    #embed_size = 300
    
    # Tokenize text.
    tokenizer = Tokenizer(num_words=max_features, filters= '', lower=False)
    logging.info('Fitting tokenizer.')
    print('Fitting tokenizer.')
    
    # Fit tokenized on both train and test and generate word_index.
    all_text = list(X_train) + list(X_test)
    tokenizer.fit_on_texts(all_text)
    word_index = tokenizer.word_index
    path = project_dir + '/data/processed/word_index.pkl'
    with open(path, 'wb') as f:
        pickle.dump(word_index, f, pickle.HIGHEST_PROTOCOL)


    # Tokenize train and test data.   
    X_train = tokenizer.texts_to_sequences(list(X_train))
    X_test = tokenizer.texts_to_sequences(list(X_test))

    # Pad sequences up to maxlen.
    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    data = pack_data_dict(X_train, y_train, X_test, y_test)
    tok_path = project_dir + '/data/processed/data_tokenize.pkl'
    print(f'Saving tokenized data to {tok_path}.')
    with open(tok_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    #embed_path = project_dir + '/data/external/glove.840B.300d.txt'
    embed_path = project_dir + '/data/external/glove.840B.300d.pkl'
    print(f"Embedding path: {embed_path}")
    embeddings_index = load_glove_embeddings_index(word_index, embed_path)
    embedding_matrix = create_embedding_matrix(embeddings_index, 
                                               max_features, 
                                               word_index)
    path = project_dir + '/data/processed/embedding_matrix.pkl'
    with open(path, 'wb') as f:
        pickle.dump(embedding_matrix, f, pickle.HIGHEST_PROTOCOL)
    vocab = build_vocab(all_text)
    unknown_words = check_coverage(vocab, embeddings_index)
   

if __name__ == '__main__':
    log_fmt = '%(asctime)s-%(levelname)s: %(message)s'
    date_fmt = '%Y%m%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)
    
    project_dir = str(Path(__file__).resolve().parents[2])

    main(project_dir)
