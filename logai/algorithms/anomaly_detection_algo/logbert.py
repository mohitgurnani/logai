#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from logai.algorithms.algo_interfaces import NNAnomalyDetectionAlgo
from logai.algorithms.nn_model.logbert.configs import LogBERTConfig
from logai.algorithms.nn_model.logbert.train import LogBERTTrain, LogBERTMultiClassificationTrain
from logai.algorithms.nn_model.logbert.predict import LogBERTPredict, LogBERTMultiClassificationPredict
from logai.algorithms.factory import factory
from datasets import Dataset as HFDataset
import pandas as pd


@factory.register("detection", "logbert", LogBERTConfig)
class LogBERT(NNAnomalyDetectionAlgo):
    """Logbert model for anomaly detection of logs
    :param config: A config object for logbert model.
    """

    def __init__(self, config: LogBERTConfig):
        if config.multi_classifier:
            self.logbert_train = LogBERTMultiClassificationTrain(config=config)
            self.logbert_predict = LogBERTMultiClassificationPredict(config=config)
        else:
            self.logbert_train = LogBERTTrain(config=config)
            self.logbert_predict = LogBERTPredict(config=config)

    def fit(self, train_data: HFDataset, dev_data: HFDataset, **kwargs):
        """Fit method for training logBERT model.
        
        :param train_data: The training dataset of type huggingface Dataset object.
        :param dev_data: The development dataset of type huggingface Dataset object.
        :param kwargs: Extra data
        """
        self.logbert_train.fit(train_data, dev_data, **kwargs)

    def predict(self, test_data: HFDataset, **kwargs) -> pd.DataFrame:
        """Predict method for running inference on logBERT model.
        
        :param test_data: The test dataset of type huggingface Dataset object.
        :param kwargs: Extra data
        :return: A pandas dataframe object containing the evaluation results for each type of metric.
        """
        return self.logbert_predict.predict(test_data, **kwargs)
