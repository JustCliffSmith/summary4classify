# -*- coding: utf-8 -*-
import logging
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from features.utility import load_TFIDF_data

def train(model, data):
    ''' Trains a scikit learn model.
    
    Keyword arguments:
    model - Instantiated sklearn model.
    data - Dictionary of data ready for training.

    Returns:
    model - Trained sklearn model.
    '''
    X_train = data['X_train']
    y_train = data['y_train']

    model.fit(X_train, y_train)
    return model

def main(project_dir):
    """ 
    """
    logger = logging.getLogger(__name__)
    lr = LogisticRegression()
    data = load_TFIDF_data(project_dir)
    lr = train(lr, data)
    with open(project_dir + '/models/baseline_lr.pkl', 'wb') as f:
        pickle.dump(lr, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    log_fmt = '%(asctime)s-%(levelname)s: %(message)s'
    date_fmt = '%Y%m%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)

    project_dir = str(Path(__file__).resolve().parents[2])

    main(project_dir)
