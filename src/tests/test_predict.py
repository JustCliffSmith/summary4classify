# -*- coding: utf-8 -*-
from pathlib import Path
import pytest

project_dir = str(Path(__file__).resolve().parents[2])

from features.utility import load_TFIDF_data
from models.predict import predict

# My first test! Just want to see how it works!
def test_prediction_shape():
    data = load_TFIDF_data(project_dir)
    #TODO need to implement training...
    X_test = data['X_test']
    assert shape(predict(model, data)) == (X_test.shape)
