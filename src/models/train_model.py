# -*- coding: utf-8 -*-
import logging
import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from features.utility import load_TFIDF_data
from pipeline.utility import parse_arguments

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
    
    logging.info(f"Training set size: {y_train.shape[0]:,}")
    
    cv_results = cross_validate(model, 
                                X_train, 
                                y_train,
                                scoring='f1_macro',
                                n_jobs=-1,
                                verbose=1,
                                cv=5)
    logging.info(f"cv scores: {cv_results['test_score']}")
    logging.info(f"Average cv score: {np.mean(cv_results['test_score']):.6f}")
    logging.info(f"Std cv score: {np.std(cv_results['test_score']):.6f}")
    model.fit(X_train, y_train)
    return model

def main(project_dir, args):
    """ 
    """
    logger = logging.getLogger(__name__)

    lr = LogisticRegression(multi_class='auto', 
                            solver='lbfgs', 
                            max_iter=10000)
    if args.baseline == True:
        data = load_TFIDF_data(project_dir)
        lr = train(lr, data)
        with open(project_dir + '/models/baseline_lr.pkl', 'wb') as f:
            pickle.dump(lr, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    log_fmt = '%(asctime)s-%(levelname)s: %(message)s'
    date_fmt = '%Y%m%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)

    project_dir = str(Path(__file__).resolve().parents[2])

    args = parse_arguments()

    main(project_dir, args)
