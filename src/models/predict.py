# -*- coding: utf-8 -*-
import sys
import logging
from pathlib import Path


def predict(model, data):
    ''' Generate predictions from trained scikit-learn model.

    Keyword arguments:
    model - trained sklearn model
    data - dictionary containing test

    Returns:
    pred -     
    '''

    X_test = data['X_test']
    y_test = data['y_test']
    try:
        pred = model.predict_proba(y_test)
    except AttributeError:
        logging.info('predict_proba() is not implemented for this model!')
        sys.exit()
    #TODO print helpful statistics from the model!
    return pred

def main(project_dir):
    ''' 
    '''
    logger = logging.getLogger(__name__)


if __name__ == '__main__':
    log_fmt = '%(asctime)s-%(levelname)s: %(message)s'
    date_fmt = '%Y%m%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)

    project_dir = str(Path(__file__).resolve().parents[2])

    main(project_dir)
