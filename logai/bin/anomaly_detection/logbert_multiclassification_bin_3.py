import os
from _csv import writer
import argparse
import sys
import json

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

partitioned_filepath = os.path.join(config.output_dir, 'partitioned.csv')

train_filepath = os.path.join(config.output_dir, 'train_ml.csv')
dev_filepath = os.path.join(config.output_dir, 'dev_ml.csv')
test_filepath = os.path.join(config.output_dir, 'test_ml.csv')


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


# #data_types = {'Column1':str, 'Column2': str, 'Column3': str, 'Column4': int, 'Column5': str, 'Column6': int, 'Column7': str}

# # if config.data_loader_config.test and os.path.exists(config.data_loader_config.test):
# #   test_df = pd.read_csv(config.data_loader_config.test)#), index_col=0) #dtype=data_types, nrows=2000)
# if config.data_loader_config.train and os.path.exists(config.data_loader_config.train):
#   train_df = pd.read_csv(config.data_loader_config.train)#), index_col=0)#, dtype=data_types, encoding='unicode_escape')#, nrows=2000)
# else:
#   exit(1)

# train_df = process_df(train_df, "train")
# # test_df = process_df(test_df, "test")

# # Perform vertical concatenation
# # logs_df = pd.concat([train_df, test_df])
# logs_df = train_df

# # Reset the index
# logs_df = logs_df.reset_index(drop=True)

# # Some more pre-processing
# category_counts = logs_df['label'].value_counts()
# print("Category counts before splitting: %s" % category_counts)
# labels = logs_df['label'].unique().tolist()
# labels = [s.strip() for s in labels]
# num_labels = len(labels)
# id2label = {id: label for id, label in enumerate(labels)}
# label2id = {label: id for id, label in enumerate(labels)}
# logs_df[constants.LABELS] = logs_df.label.map(lambda x: label2id[x.strip()])

# metadata = {constants.LOG_TIMESTAMPS: [constants.LOG_TIMESTAMPS],
#             "body": [constants.LOGLINE_NAME, "filename"], constants.LABELS: [constants.LABELS],
#             "severity_text": ["severity"], constants.LOG_SEVERITY_NUMBER: [constants.LOG_SEVERITY_NUMBER]}

# logrecord = LogRecordObject.from_dataframe(logs_df, metadata)

# logrecord.save_to_csv(processed_filepath)


# # partitioner = OpenSetPartitioner(config.open_set_partitioner_config)
# # logrecord = partitioner.partition(logrecord)
# # logrecord.save_to_csv(partitioned_filepath)

# # print (logrecord.body[constants.LOGLINE_NAME])

# train_data, dev_data, test_data = split_train_dev_test_for_anomaly_detection(
#                 logrecord,training_type=config.training_type,
#                 test_data_frac_neg_class=config.test_data_frac_neg,
#                 test_data_frac_pos_class=config.test_data_frac_pos,
#                 shuffle=config.train_test_shuffle
#             )
# category_counts = train_data.labels['labels'].value_counts().rename(index=id2label)
# print("Category counts after splitting: train: %s" % category_counts)
# category_counts = dev_data.labels['labels'].value_counts().rename(index=id2label)
# print("Category counts after splitting: dev: %s" % category_counts)
# category_counts = test_data.labels['labels'].value_counts().rename(index=id2label)
# print("Category counts after splitting: test: %s" % category_counts)


# train_data.save_to_csv(train_filepath)
# dev_data.save_to_csv(dev_filepath)
# test_data.save_to_csv(test_filepath)

# # print ('Train/Dev/Test Anomalous', len(train_data.labels[train_data.labels[constants.LABELS]==1]),
# #                                    len(dev_data.labels[dev_data.labels[constants.LABELS]==1]),
# #                                    len(test_data.labels[test_data.labels[constants.LABELS]==1]))
# # print ('Train/Dev/Test Normal', len(train_data.labels[train_data.labels[constants.LABELS]==0]),
# #                                    len(dev_data.labels[dev_data.labels[constants.LABELS]==0]),
# #                                    len(test_data.labels[test_data.labels[constants.LABELS]==0]))

# dev_features = get_features(dev_data, "dev")
# train_features = get_features(train_data, "train")

# torch.cuda.empty_cache()



# # Read the CSV file into a DataFrame
input_csv = "/home/ml/workspace/data/temp_output_n7/test_ml.csv"
metadata_file = "/home/ml/workspace/data/temp_output_n7/test_ml_metadata.json"
split_files_dir = "/home/ml/workspace/data/temp_output_n7/split_files_dir"
metadata_files_dir = "/home/ml/workspace/data/temp_output_n7/split_files_dir"

df = pd.read_csv(input_csv)

chunk_size = 100

# Calculate the number of chunks needed
num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)
# Split into chunks of size 100
chunks = []
start_index = 0
for i in range(num_chunks):
    end_index = min(start_index + chunk_size, len(df))
    chunks.append(df[start_index:end_index])
    start_index = end_index

# Create directories to store the chunk files
os.makedirs(split_files_dir, exist_ok=True)
os.makedirs(metadata_files_dir, exist_ok=True)

# Write each chunk to a separate CSV file
created_files = []
for i, chunk in enumerate(chunks, start=1):
    csv_file = f"{split_files_dir}/test_ml_{i}.csv"
    chunk.to_csv(csv_file, index=False)
    created_files.append(csv_file)
    
    # Copy metadata file
    metadata_dest = f"{metadata_files_dir}/test_ml_{i}_metadata.json"
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    with open(metadata_dest, "w") as f:
        json.dump(metadata, f, indent=4)
    created_files.append(metadata_dest)


anomaly_detector = NNAnomalyDetector(config=config.nn_anomaly_detection_config)
# anomaly_detector.fit(train_features, dev_features, num_labels=num_labels, id2label=id2label, label2id=label2id, device=device)
# del train_features, dev_features

# num_labels = 7
# id2label = {0: 'vm_operation', 1: 'high_availability', 2: 'acropolis_scheduler', 3: 'ahv_networking', 4: 'acropolis_crash', 5: 'ahv_libvirt_qemu', 6: 'gpu'}
# label2id = {'vm_operation': 0, 'high_availability': 1, 'acropolis_scheduler': 2, 'ahv_networking': 3, 'acropolis_crash': 4, 'ahv_libvirt_qemu': 5, 'gpu': 6}
# num_labels = 8
# id2label = {0: 'vm_operation', 1: 'high_availability', 2: 'acropolis_scheduler', 3: 'ahv_networking', 4: 'acropolis_crash', 5: 'ahv_libvirt_qemu', 6: 'gpu', 7: 'normal'}
# label2id = {'vm_operation': 0, 'high_availability': 1, 'acropolis_scheduler': 2, 'ahv_networking': 3, 'acropolis_crash': 4, 'ahv_libvirt_qemu': 5, 'gpu': 6, 'normal': 7}
num_labels = 8
id2label = {0: 'normal', 1: 'ahv_libvirt_qemu', 2: 'acropolis_crash', 3: 'vm_operation', 4: 'acropolis_scheduler', 5: 'gpu', 6: 'ahv_networking', 7: 'high_availability'}
label2id = {'normal': 0, 'ahv_libvirt_qemu': 1, 'acropolis_crash': 2, 'vm_operation': 3, 'acropolis_scheduler': 4, 'gpu': 5, 'ahv_networking': 6, 'high_availability': 7}


overall_output = []
accuracy_total = 0
precision_total = 0
recall_total = 0
f1_score_total = 0
total_entries = 0

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
for i in range(num_chunks):
    test_data = LogRecordObject.load_from_csv(f"{split_files_dir}/test_ml_{i+1}.csv")

    #test_features = get_features(test_data, "test")
    # For each text, we can check predict logic
    test_data.labels.reset_index(drop=True, inplace=True)
    predict_results = anomaly_detector.predict(test_data, num_labels=num_labels, id2label=id2label, label2id=label2id, device=device)
    print(predict_results)

    y_pred = [label2id[x[2]] for x in predict_results]
    y_true = test_data.labels['labels'].tolist()

    # Iterate over elements of both lists simultaneously
    print("\n############################\n")
    for i, (pred, true) in enumerate(zip(y_pred, y_true)):
        # Check for inequality
        if pred != true:
            print(test_data.body["logline"][i])
            print("pred: %s  true:%s\n"% (id2label[pred], id2label[true]))
    print("\n############################\n")

    # Assuming y_true and y_pred are your true and predicted labels
    accuracy = accuracy_score(y_true, y_pred)
    print("accuracy - %s" % accuracy)
    overall_output.append("accuracy - %s" % accuracy)
    accuracy_total += accuracy*len(predict_results)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)  # 'weighted' for multi-class
    print("precision - %s" % precision)
    overall_output.append("precision - %s" % precision)
    precision_total += precision*len(predict_results)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    print("recall - %s" % recall)
    overall_output.append("recall - %s" % recall)
    recall_total += recall*len(predict_results)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    print("f1 score - %s" % f1)
    overall_output.append("f1 score - %s" % f1)
    f1_score_total += f1*len(predict_results)
    overall_output.append("\n")
    conf_matrix = confusion_matrix(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred)
    total_entries += len(predict_results)

#--config_file /home/ml/data/logbert_config_sm2.yaml --predict_only 2 --process_data 1 --partition_data 1 --split_needs 1 --tokenization_needed 1

# Delete the newly created files
for file in created_files:
    os.remove(file)

print("\n FINAL RESULTS \n")
for line in overall_output:
    print(line)

print("accuracy: ", accuracy_total/total_entries)
print("precision: ", precision_total/total_entries)
print("recall: ", recall_total/total_entries)
print("f1_score: ", f1_score_total/total_entries)