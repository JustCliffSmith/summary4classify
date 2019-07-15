# -*- coding: utf-8 -*-
import pickle
import logging
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline.utility import pack_data_dict, unpack_data_dict

def load_raw_data(project_dir):
    ''' Load raw newsgroup data.
   
    Returns:
    data - dict of training and testing data. 
    '''
    
    with open(project_dir + '/data/interim/newsgroup.pkl', 'rb') as f:
        data = pickle.load(f)   
    return data


def vectorize(data):
    ''' Vectorize text data using TF-IDF. 

    Keyword arguments:
    data - raw data.

    Returns:
    vec_data - data passed through sklearn's TF-IDF
    '''
   
    X_train, y_train, X_test, y_test = unpack_data_dict(data) 
        
    vec = TfidfVectorizer()
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    vec_data = pack_data_dict(X_train_vec, y_train, X_test_vec, y_test)

    return vec_data


def main(project_dir):
    """ 
    """
    logger = logging.getLogger(__name__)

    data = load_raw_data(project_dir)
    vec_data = vectorize(data)
    with open(project_dir + '/data/processed/TFIDFnewsgroup.pkl', 'wb') as f:
        pickle.dump(vec_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    log_fmt = '%(asctime)s-%(levelname)s: %(message)s'
    date_fmt ='%Y%m%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)

    project_dir = str(Path(__file__).resolve().parents[2])

    main(project_dir)
