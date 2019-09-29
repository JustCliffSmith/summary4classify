# -*- coding: utf-8 -*-
import argparse

def parse_arguments():
    """ Parses any command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", 
                        action="store_true",
                        help="Run baseline vectorizer and model.")
    return parser.parse_args()

def pack_data_dict(X_train, y_train, X_test, y_test):
    """ Packs training and testing data into a dict."""
    data = {'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test}
    return data

def unpack_data_dict(data):
    """ Removes training and testing data from a dict and returns a tuple."""
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']
