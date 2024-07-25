import pandas as pd
import numpy as np
import pickle
import joblib

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin





class ProtoEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, stats_file='encoding_stats.pkl'):
        self.stats_file = stats_file
        self.proto_stats = None
        self.proto_overall_mean = None
    
    def fit(self, X, y=None):
        with open(self.stats_file, 'rb') as f:
            encoding_stats = pickle.load(f)
        
        self.proto_stats, self.proto_overall_mean = encoding_stats['proto']
        return self
    
    def transform(self, X):
        X = X.copy()
        X['proto_target_encoded'] = X['proto'].map(self.proto_stats)
        X['proto_target_encoded'].fillna(self.proto_overall_mean, inplace=True)
        return X

class ServiceEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, stats_file='encoding_stats.pkl'):
        self.stats_file = stats_file
        self.service_stats = None
        self.service_overall_mean = None
    
    def fit(self, X, y=None):
        with open(self.stats_file, 'rb') as f:
            encoding_stats = pickle.load(f)
        
        self.service_stats, self.service_overall_mean = encoding_stats['service']
        return self
    
    def transform(self, X):
        X = X.copy()
        X['service_target_encoded'] = X['service'].map(self.service_stats)
        X['service_target_encoded'].fillna(self.service_overall_mean, inplace=True)
        return X

class StateEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, stats_file='encoding_stats.pkl'):
        self.stats_file = stats_file
        self.state_stats = None
        self.state_overall_mean = None
    
    def fit(self, X, y=None):
        with open(self.stats_file, 'rb') as f:
            encoding_stats = pickle.load(f)
        
        self.state_stats, self.state_overall_mean = encoding_stats['state']
        return self
    
    def transform(self, X):
        X = X.copy()
        X['state_target_encoded'] = X['state'].map(self.state_stats)
        X['state_target_encoded'].fillna(self.state_overall_mean, inplace=True)
        return X

class ColumnDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        cols = ['proto', 'service', 'state', 'id']
        return X.drop(cols, axis = 1)

class FeatureScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        with open('robust_scaler_model.pkl', 'rb') as f:
            loaded_scaler = pickle.load(f)
        exclude_columns = ['is_ftp_login', 'ct_ftp_cmd', 'is_sm_ips_ports', 'label']
        columns_to_scale = [col for col in X.columns if col not in exclude_columns]
        X[columns_to_scale] = loaded_scaler.transform(X[columns_to_scale])
        return X

class Prediction(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        model = joblib.load('pretrained_model.pkl')
        pred =  model.predict(X)
        return pd.DataFrame(pred, columns = ['prediction'])
