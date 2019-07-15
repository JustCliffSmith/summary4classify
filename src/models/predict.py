# -*- coding: utf-8 -*-
import sys
import pickle
import logging
from pathlib import Path
from sklearn.metrics import f1_score

from features.utility import load_TFIDF_data
from pipeline.utility import parse_arguments, pack_data_dict


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
    logging.info(f"Test set size: {y_test.shape[0]:,}")

    pred = model.predict(X_test)

    try:
        pred_proba = model.predict_proba(X_test)
    except AttributeError:
        logging.info('predict_proba() is not implemented for this model!')
        sys.exit()
    
    f1_macro = f1_score(y_test, pred, average='macro')
    logging.info(f"Macro f1 score on test set: {f1_macro:.4f}")
    
    return pred, pred_proba

def main(project_dir, args):
    ''' 
    '''
    logger = logging.getLogger(__name__)
    if args.baseline == True:
        data = load_TFIDF_data(project_dir)
        with open(project_dir + '/models/baseline_lr.pkl', 'rb') as f:
            lr_baseline = pickle.load(f)
        pred, pred_proba = predict(lr_baseline, data)

if __name__ == '__main__':
    log_fmt = '%(asctime)s-%(levelname)s: %(message)s'
    date_fmt = '%Y%m%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)

    project_dir = str(Path(__file__).resolve().parents[2])
    args = parse_arguments()
    main(project_dir, args)
