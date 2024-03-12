import os
from _csv import writer
import argparse
import sys

from logai.algorithms.factory import AlgorithmFactory
from logai.algorithms.nn_model.logbert.configs import LogBERTConfig
from logai.algorithms.vectorization_algo.logbert import LogBERTVectorizerParams
from logai.applications.openset.anomaly_detection.openset_anomaly_detection_workflow import OpenSetADWorkflowConfig, validate_config_dict
from logai.dataloader.data_model import LogRecordObject, DataLoader
from logai.preprocess.mask_logs import MaskLogLine
from logai.utils.file_utils import read_file
from logai.utils.dataset_utils import split_train_dev_test_for_anomaly_detection
import logging
from logai.dataloader.data_loader import FileDataLoader
from logai.preprocess.hdfs_preprocessor import HDFSPreprocessor
from logai.information_extraction.log_parser import LogParser
from logai.preprocess.openset_partitioner import OpenSetPartitioner
from logai.analysis.nn_anomaly_detector import NNAnomalyDetector
from logai.information_extraction.log_vectorizer import LogVectorizer
from logai.utils import constants
from datasets import Dataset as HFDataset
import pandas as pd
import torch, os
from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
from torch.utils.data import Dataset


# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)


parser = argparse.ArgumentParser(description="Set some values")
parser.add_argument("--config_file", type=str, required=True, help="The value is path of your config file")
args = parser.parse_args()


config_path = args.config_file


config_parsed = read_file(config_path)
config_dict = config_parsed["workflow_config"]
config = OpenSetADWorkflowConfig.from_dict(config_dict)

processed_filepath = os.path.join(config.output_dir, 'processed.csv')
processed_filepath_2 = os.path.join(config.output_dir, 'processed_2.csv')
processed_filepath_3 = os.path.join(config.output_dir, 'processed_3.csv')
processed_filepath_4 = os.path.join(config.output_dir, 'processed_4.csv')

partitioned_filepath = os.path.join(config.output_dir, 'partitioned.csv')

train_filepath = os.path.join(config.output_dir, 'train_ml.csv')
dev_filepath = os.path.join(config.output_dir, 'dev_ml.csv')
test_filepath = os.path.join(config.output_dir, 'test_ml.csv')
test_filepath_2 = os.path.join(config.output_dir, 'test_2_ml.csv')
test_filepath_3 = os.path.join(config.output_dir, 'test_abnormal_ml.csv')
test_filepath_4 = os.path.join(config.output_dir, 'test_normal_ml.csv')


def get_features(data=None, cat=None):
  tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased", max_length=512)
  texts = []
  for text, source_file, log_level in zip(data.body[constants.LOGLINE_NAME],
                                                   data.body[constants.SOURCE_FILE],
                                                   data.severity_number[constants.LOG_SEVERITY_NUMBER]):
    texts.append(f"{text} [Log Level: {log_level}] [Source File: {source_file}]")
  encodings = tokenizer(texts, truncation=True, padding=True)
  labels_df = data.labels.reset_index(drop=True)
  data_loader = DataLoader(encodings, labels_df[constants.LABELS])
  # print("Vectorization complete for %s" % cat)
  return data_loader


def mask_logs(logline):
  masker = MaskLogLine(logline)
  nlogline = masker.mask_log_line_efficient_way(logline)
  return nlogline


def process_df(df, df_type="train"):
  #df["epoch"] = df["epoch"].astype(int)
  # has_zero_or_na_column = (df['epoch'] == 0) | df['epoch'].isna()
  # if has_zero_or_na_column.any():
  #   print("Dataset %s has corrupt epochs" % df_type)
  #   exit(1)
  # df = df.dropna(subset=['pid'])
  # print(df.shape)

  #df.drop_duplicates(subset='body', keep='first', inplace=True)
  #print("after duplicate removal")
  #print(df.shape)

  df["filename"] = df["filename"].str.replace(r'\..*', '', regex=True)
  _SEVERITY_MAP = {
    "DEBUG": 0,
    "INFO": 1,
    "WARNING": 2,
    "ERROR": 3,
    "CRITICAL": 4,
    "FATAL": 5,
  }
  df = df.rename(
    columns={"body": constants.LOGLINE_NAME, "level": constants.LOG_LEVEL}
  )
  df[constants.LOG_SEVERITY_NUMBER] = df[constants.LOG_LEVEL].map(_SEVERITY_MAP)
  df[constants.LOG_SEVERITY_NUMBER] = df[constants.LOG_SEVERITY_NUMBER].fillna(1)
  # df[constants.LOG_TIMESTAMPS] = pd.to_datetime(df["epoch"], unit='s')
  # TODO - we are not masking, will decide later
  # df[constants.LOGLINE_NAME] = df[constants.LOGLINE_NAME].apply(mask_logs)
  # print("Masking of logs is complete")
  # df[constants.SPAN_ID] = df["pid"] #df["source_file"] + "_" + df[constants.LOG_SEVERITY_NUMBER].astype(str) # df["pid"] + "_" + + logs_df["hh"].astype(str)
  return df


def divide_df(df, n):
    num_rows_to_select = len(df) // n
    selected_data = df.sample(n=num_rows_to_select)
    return selected_data


#data_types = {'Column1':str, 'Column2': str, 'Column3': str, 'Column4': int, 'Column5': str, 'Column6': int, 'Column7': str}

# if config.data_loader_config.test and os.path.exists(config.data_loader_config.test):
#   test_df = pd.read_csv(config.data_loader_config.test)#), index_col=0) #dtype=data_types, nrows=2000)
if config.data_loader_config.train and os.path.exists(config.data_loader_config.train):
  train_df = pd.read_csv(config.data_loader_config.train)#), index_col=0)#, dtype=data_types, encoding='unicode_escape')#, nrows=2000)
  normal_entries = train_df[train_df['label'] == 'normal']
  normal_entries = normal_entries.groupby('logline').head(1)
  normal_entries.reset_index(drop=True, inplace=True)
  sampled_normal_entries = normal_entries.sample(n=min(7500, len(normal_entries)))
  abnormal_entries = train_df[train_df['label'] != 'normal']
  sampled_entries = pd.concat([sampled_normal_entries, abnormal_entries])
  train_df = sampled_entries.sample(frac=1).reset_index(drop=True)
  # Get the indices of the sampled normal entries
  sampled_indices = sampled_normal_entries.index
  # Filter the normal entries DataFrame to exclude the sampled indices
  remaining_normal_entries = normal_entries[~normal_entries.index.isin(sampled_indices)]

else:
  exit(1)

train_df = process_df(train_df, "train")
train_df_2 = process_df(remaining_normal_entries, "train")
train_df_3 = process_df(abnormal_entries, "train")
train_df_4 = process_df(normal_entries, "train")
# test_df = process_df(test_df, "test")

# Perform vertical concatenation
# logs_df = pd.concat([train_df, test_df])
logs_df = train_df
logs_df_2 = train_df_2
logs_df_3 = train_df_3
logs_df_4 = train_df_4

# Reset the index
logs_df = logs_df.reset_index(drop=True)
logs_df_2 = logs_df_2.reset_index(drop=True)
logs_df_3 = logs_df_3.reset_index(drop=True)
logs_df_4 = logs_df_4.reset_index(drop=True)


# Some more pre-processing
category_counts = logs_df['label'].value_counts()
print("Category counts before splitting: %s" % category_counts)
labels = logs_df['label'].unique().tolist()
labels = [s.strip() for s in labels]
num_labels = len(labels)
id2label = {id: label for id, label in enumerate(labels)}
label2id = {label: id for id, label in enumerate(labels)}
logs_df[constants.LABELS] = logs_df.label.map(lambda x: label2id[x.strip()])
logs_df_2[constants.LABELS] = logs_df_2.label.map(lambda x: label2id[x.strip()])
logs_df_3[constants.LABELS] = logs_df_3.label.map(lambda x: label2id[x.strip()])
logs_df_4[constants.LABELS] = logs_df_3.label.map(lambda x: label2id[x.strip()])

print("num_labels", num_labels)
print("id2label", id2label)
print("label2id", label2id)

metadata = {constants.LOG_TIMESTAMPS: [constants.LOG_TIMESTAMPS],
            "body": [constants.LOGLINE_NAME, "filename"], constants.LABELS: [constants.LABELS],
            "severity_text": ["severity"], constants.LOG_SEVERITY_NUMBER: [constants.LOG_SEVERITY_NUMBER]}

logrecord = LogRecordObject.from_dataframe(logs_df, metadata)
logrecord_2 = LogRecordObject.from_dataframe(logs_df_2, metadata)
logrecord_3 = LogRecordObject.from_dataframe(logs_df_3, metadata)
logrecord_4 = LogRecordObject.from_dataframe(logs_df_3, metadata)

logrecord.save_to_csv(processed_filepath)
logrecord_2.save_to_csv(processed_filepath_2)
logrecord_3.save_to_csv(processed_filepath_3)
logrecord_4.save_to_csv(processed_filepath_4)


# partitioner = OpenSetPartitioner(config.open_set_partitioner_config)
# logrecord = partitioner.partition(logrecord)
# logrecord.save_to_csv(partitioned_filepath)

# print (logrecord.body[constants.LOGLINE_NAME])

train_data, dev_data, test_data = split_train_dev_test_for_anomaly_detection(
                logrecord,training_type=config.training_type,
                test_data_frac_neg_class=config.test_data_frac_neg,
                test_data_frac_pos_class=config.test_data_frac_pos,
                shuffle=config.train_test_shuffle
            )
category_counts = train_data.labels['labels'].value_counts().rename(index=id2label)
print("Category counts after splitting: train: %s" % category_counts)
category_counts = dev_data.labels['labels'].value_counts().rename(index=id2label)
print("Category counts after splitting: dev: %s" % category_counts)
category_counts = test_data.labels['labels'].value_counts().rename(index=id2label)
print("Category counts after splitting: test: %s" % category_counts)


train_data.save_to_csv(train_filepath)
dev_data.save_to_csv(dev_filepath)
test_data.save_to_csv(test_filepath)
logrecord_2.save_to_csv(test_filepath_2)
logrecord_3.save_to_csv(test_filepath_3)
logrecord_4.save_to_csv(test_filepath_4)


# print ('Train/Dev/Test Anomalous', len(train_data.labels[train_data.labels[constants.LABELS]==1]),
#                                    len(dev_data.labels[dev_data.labels[constants.LABELS]==1]),
#                                    len(test_data.labels[test_data.labels[constants.LABELS]==1]))
# print ('Train/Dev/Test Normal', len(train_data.labels[train_data.labels[constants.LABELS]==0]),
#                                    len(dev_data.labels[dev_data.labels[constants.LABELS]==0]),
#                                    len(test_data.labels[test_data.labels[constants.LABELS]==0]))

dev_features = get_features(dev_data, "dev")
train_features = get_features(train_data, "train")

torch.cuda.empty_cache()

anomaly_detector = NNAnomalyDetector(config=config.nn_anomaly_detection_config)
anomaly_detector.fit(train_features, dev_features, num_labels=num_labels, id2label=id2label, label2id=label2id, device=device)

del train_features, dev_features

#test_features = get_features(test_data, "test")
# For each text, we can check predict logic
test_data.labels.reset_index(drop=True, inplace=True)
predict_results = anomaly_detector.predict(test_data, num_labels=num_labels, id2label=id2label, label2id=label2id, device=device)


y_pred = [label2id[x[2]] for x in predict_results]
y_true = test_data.labels['labels'].tolist()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Assuming y_true and y_pred are your true and predicted labels
accuracy = accuracy_score(y_true, y_pred)
print("accuracy - %s" % accuracy)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)  # 'weighted' for multi-class
print("precision - %s" % precision)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
print("recall - %s" % recall)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
print("f1 score - %s" % f1)
conf_matrix = confusion_matrix(y_true, y_pred)
classification_rep = classification_report(y_true, y_pred)


#--config_file /home/ml/data/logbert_config_sm2.yaml --predict_only 2 --process_data 1 --partition_data 1 --split_needs 1 --tokenization_needed 1

