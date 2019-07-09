# -*- coding: utf-8 -*-
import logging
import pickle
from pathlib import Path

project_dir = str(Path(__file__).resolve().parents[2])

def load_TFIDF_data(project_dir):
    ''' Load TFIDF newsgroup data. '''
    with open(project_dir + '/data/processed/TFIDFnewsgroup.pkl', 'rb') as f:
        data = pickle.load(f) 
    return data

def main(project_dir):
    """ 
    """
    logger = logging.getLogger(__name__)


if __name__ == '__main__':
    log_fmt = '%(asctime)s-%(levelname)s: %(message)s'
    date_fmt = '%Y%m%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)

    project_dir = str(Path(__file__).resolve().parents[2])  

    main(project_dir)
