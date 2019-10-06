#!/bin/python

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Stacker(BaseEstimator):
    def __init__(self, layer_models, blender_model, holdout_perc):
        self.layer_models = layer_models
        self.blender_model = blender_model
        self.holdout_perc = holdout_perc

    def fit(self, X, y):
        X_train, X_holdout = train_test_split(
            X, test_size=self.holdout_perc, random_state=0)
        y_train, y_holdout = train_test_split(
            y, test_size=self.holdout_perc, random_state=0)

        # Train per usual, and make predictions on the holdout set
        predictions = []
        for clf in self.layer_models:
            clf.fit(X_train, y_train)
            predictions.append(clf.predict(X_holdout))

        # Flatten and create dataframe of predictions
        pred_flat = list(zip(*predictions))  # * unzips a list
        df_pred = pd.DataFrame(pred_flat)

        # Train the blender model on the predictions
        self.blender_model.fit(df_pred, y_holdout)

    def predict(self, X_test):
        # Just follow the same sequence as above
        # No splitting is required here
        layer_test = [
            clf.predict(X_test) for clf in self.layer_models]
        df_test_pred = pd.DataFrame(list(zip(*layer_test)))

        pred = self.blender_model.predict(df_test_pred)
        return pred

    def score(self, X_test, y_test):
        accuracy = accuracy_score(y_test, self.predict(X_test))
        return accuracy
