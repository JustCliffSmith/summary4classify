# -*- coding: utf-8 -*-
import sys
import logging
import pickle
from pathlib import Path
import operator

import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from features.build_features import load_raw_data
#from utility import pack_data_dict, unpack_data_dict

def load_glove_embeddings_index(word_index, embed_path):
    ''' Load glove embedding from a text file and creates embedding matrix.

    Keyword arguments:
    word_index - word_index from keras Tokenizer
    embed_path - path to embedding file, either as a .txt or .pkl
    
    Returns:
    embeddings_index - embeddings index
    '''
   
    if embed_path.endswith('.txt'):
        logging.info("Loading glove embedding from .txt file.")
        def get_coefs(word,*arr): 
            return word, np.asarray(arr, dtype='float32')[:300]
        embeddings_index = {(get_coefs(*o.split(" ")) for o in open(embed_path))}
        # Pickle embeddings so can do the quick way in the future?
    if embed_path.endswith('.pkl'):
        logging.info("Loading glove embedding from .pkl file.")
        with open(embed_path, 'rb') as f:    
            embeddings_index = pickle.load(f)
    else:
        logging.error("Embedding path must end in .txt or .pkl")
        #raise Exception
        import sys
        sys.exit() 

    return embeddings_index

def create_embedding_matrix(embeddings_index):
    ''' Creates embedding matrix from embeddings index.
    
    Creates embedding matrix from embeddings index. If word is not found 
    then tries lowercase and then Titlecase version of the word.

    Keyword arguments:
    embeddings_index - embeddings index

    Returns:
    embedding_matrix - matrix of embeddings, dim=(num_words, embed_size)
    '''
    logging.info("Computing mean and std of embedding indices.")
    # Generate stats on embedding so OOV words do not skew mean and std.
    all_embs = np.stack(embeddings_index.values())
    emb_mean = np.mean(all_embs)
    emb_std = np.std(all_embs)
    logging.info(emb_mean)
    logling.info(emb_std)
    emb_mean, emb_std = -0.005838499,0.48782197
    embed_size = all_embs.shape[1]

    num_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, 
                                        emb_std, 
                                        (num_words, embed_size))
    logging.info("Populating embedding matrix.")
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
            
    return embedding_matrix 


def build_vocab(texts):
    ''' Create a dictionary

    Keyword arguments:
    texts - pandas series containing the corpus.

    Returns:
    vocab - dict, keys are words and values are number of occurences.
    '''
    if type(texts) != pd.core.series.Series:
        logging.error('Texts corpus must be in pandas series!')
        sys.exit() 

    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            vocab.setdefault(word, 0)
            vocab[word] += 1            
    return vocab


def check_coverage(vocab, embeddings_index):
    ''' Checks percentage of vocabularly in pre-trained embedding. 

    Loops through all words in vocabularly to determine what percentage are
    in the pre-trained embedding. Percentages are printed.
    Lightly modified from Dieter's work on Kaggle.
    
    Keyword Arguments: 
    vocab - dictionary of vocabularly counts 
    embedding_index -  
    
    Return: 
    unknown_words = words not in embedding.
    '''
    known_words = {}
    unknown_words = {}
    num_known_words = 0
    num_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    percentage_vocab = len(known_words) / len(vocab)
    percentage_alltext = num_known_word / (num_known_words + num_unknown_words)
    logging.info(f"Found embeddings for {percentage_vocab:.3f} of vocab")
    logging.info(f"Found embeddings for  {percentage_alltext:.3f} of all text")
    unknown_words = sorted(unknown_words.items(), 
                           key=operator.itemgetter(1))[::-1]
    return unknown_words


def main(project_dir):
    """ 
    """
    logger = logging.getLogger(__name__)
    data = load_raw_data(project_dir)
    #X_train, y_train, X_test, y_test = unpack_data_dict(data)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    maxlen = 220
    max_features = 100000
    #embed_size = 300
    tokenizer = Tokenizer(num_words=max_features, filters= '', lower=False)
    logging.info('Fitting tokenizer.')
    
    all_text = list(X_train) + list(X_test)
    tokenizer.fit_on_texts(all_text)
    word_index = tokenizer.word_index
   
    X_train = tokenizer.texts_to_sequences(list(X_train))
    y_train = y_train
    X_test = tokenizer.texts_to_sequences(list(X_test))
    y_test = y_test

    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)

    #data = pack_data_dict(X_train, y_train, X_test, y_test)
    data = {'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test}
    with open(project_dir + '/data/processed/data_tokenize.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    '''
    embed_path = project_dir + '/data/external/glove.840B.300d.txt'
    print(embed_path)
    embeddings_index = load_glove_embeddings_index(word_index, embed_path)
    embedding_matrix = create_embedding_matrix(embeddings_index)
    vocab = build_vocab(texts)
    unknown_words = check_coverage(vocab, embeddings_index)
    '''

if __name__ == '__main__':
    log_fmt = '%(asctime)s-%(levelname)s: %(message)s'
    date_fmt = '%Y%m%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)
    
    project_dir = str(Path(__file__).resolve().parents[2])

    main(project_dir)
