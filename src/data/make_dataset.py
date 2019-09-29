# -*- coding: utf-8 -*-
import logging
import pickle
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups
from pipeline.utility import pack_data_dict


def main(project_dir):
    """ Loads data from the 20newsgroup via sklearn.
 
    Stores the pickled dict in ../../data/interim
    """
    logger = logging.getLogger(__name__)
    logger.info("Making final data set from raw data.")

    data_home = project_dir + '/data/interim'
    train = fetch_20newsgroups(data_home=data_home,
                               subset='train', 
                               remove=('headers', 'footers', 'quotes'))
    test = fetch_20newsgroups(data_home=data_home,
                              subset='test', 
                              remove=('headers', 'footers', 'quotes'))

    data = pack_data_dict(train.data, train.target, test.data, test.target) 
    
    with open(data_home  + '/newsgroup.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return data


if __name__ == '__main__':
    log_fmt = '%(asctime)s-%(levelname)s: %(message)s'
    date_fmt = '%Y%m%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt=date_fmt)

    project_dir = str(Path(__file__).resolve().parents[2])

    data = main(project_dir)
