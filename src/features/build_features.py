# -*- coding: utf-8 -*-
import pickle
import logging
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

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
    
    X_train = data['X_train']
    X_test = data['X_test']
        
    vec = TfidfVectorizer()
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    vec_data = {'X_train': X_train_vec,
                'y_train': data['y_train'],
                'X_test': X_test_vec,
                'y_test': data['y_test']}

    return vec_data


def main(project_dir):
    """ 
    """
    logger = logging.getLogger(__name__)

    data = load_raw_data(project_dir)
    vec_data = vectorize(data)
    with open(project_dir + '/data/processed/TFIDFnewsgroup.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    log_fmt = '%(asctime)s-%(levelname)s: %(message)s'
    date_fmt ='%Y%m%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)

    project_dir = str(Path(__file__).resolve().parents[2])

    main(project_dir)
