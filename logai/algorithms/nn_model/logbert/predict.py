#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from transformers import BertForMaskedLM, DataCollatorWithPadding, BertForSequenceClassification, BertTokenizerFast
from datasets import Dataset as HFDataset
import numpy as np
import pandas as pd
from .eval_metric_utils import compute_metrics, ground_truth, prediction_score, \
    compute_auc_roc_for_metric
from .predict_utils import Predictor, PredictionLabelSmoother
from transformers import TrainingArguments
import logging
import os
from .configs import LogBERTConfig
from logai.utils import constants
from .tokenizer_utils import (
    get_special_tokens,
    get_special_token_ids,
    get_tokenizer,
    get_mask_id,
)


class LogBERTPredict:
    """Class for running inference on logBERT model for unsupervised log anomaly detection.

    :param config: config object describing the parameters of logbert model.
    """

    def __init__(self, config: LogBERTConfig):

        self.config = config

        self.model_dirpath = os.path.join(
            self.config.output_dir, self.config.model_name
        )

        self.model = None
        self.tokenizer = get_tokenizer(self.config.tokenizer_dirpath)
        self.special_tokens = get_special_tokens()
        self.special_token_ids = get_special_token_ids(
            self.tokenizer
        )  # [self.tokenizer.convert_tokens_to_ids(x) for x in special_tokens]
        self.mask_id = get_mask_id(self.tokenizer)

        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=self.config.max_token_len,
        )

        print("initialized data collator")
        self.predictor_args = TrainingArguments(
            self.model_dirpath,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            eval_accumulation_steps=self.config.eval_accumulation_steps,
            resume_from_checkpoint=self.config.resume_from_checkpoint,
        )

    def _generate_masked_input(self, examples, indices):
        input_ids = examples["input_ids"][0]
        attention_masks = examples["attention_mask"][0]
        token_type_ids = examples["token_type_ids"][0]
        index = indices[0]
        input_ids = np.array(input_ids)

        sliding_window_diag = np.eye(input_ids.shape[0])

        if sliding_window_diag.shape[0] == 0:
            return examples
        mask = 1 - np.isin(input_ids, self.special_token_ids).astype(np.int64)
        sliding_window_diag = sliding_window_diag * mask
        sliding_window_diag = sliding_window_diag[
            ~np.all(sliding_window_diag == 0, axis=1)
        ]
        num_sections = int(sliding_window_diag.shape[0] / self.config.mask_ngram)
        if num_sections <= 0:
            num_sections = sliding_window_diag.shape[0]
        sliding_window_diag = np.array_split(sliding_window_diag, num_sections, axis=0)
        diag = np.array([np.sum(di, axis=0) for di in sliding_window_diag])

        input_rpt = np.tile(input_ids, (diag.shape[0], 1))
        labels = np.copy(input_rpt)
        input_ids_masked = (input_rpt * (1 - diag) + diag * self.mask_id).astype(
            np.int64
        )
        attention_masks = np.tile(
            np.array(attention_masks), (input_ids_masked.shape[0], 1)
        )
        token_type_ids = np.tile(
            np.array(token_type_ids), (input_ids_masked.shape[0], 1)
        )
        labels[
            input_ids_masked != self.mask_id
        ] = -100  # Need masked LM loss only for tokens with mask_id
        examples = {}
        examples["input_ids"] = input_ids_masked
        examples["attention_mask"] = attention_masks
        examples["token_type_ids"] = token_type_ids
        examples["labels"] = labels
        examples["indices"] = np.array([index] * input_ids_masked.shape[0]).astype(
            np.int64
        )
        return examples

    def load_model(self):
        """Loading logbert model from the model dir path as specified in the logBERTConfig config"""
        checkpoint_dir = "checkpoint-" + str(
            max(
                [
                    int(x.split("-")[1])
                    for x in os.listdir(self.model_dirpath)
                    if x.startswith("checkpoint")
                ]
            )
        )
        model_checkpoint = os.path.abspath(
            os.path.join(self.model_dirpath, checkpoint_dir)
        )
        logging.info("Loading model from {}".format(model_checkpoint))
        self.model = BertForMaskedLM.from_pretrained(model_checkpoint)
        self.model.tokenizer = self.tokenizer  # self.vectorizer.tokenizer
        self.predictor = Predictor(
            model=self.model,
            args=self.predictor_args,
            train_dataset=None,
            eval_dataset=None,
            data_collator=self.data_collator
        )
        self.predictor.label_smoother = PredictionLabelSmoother(
            epsilon=self.predictor_args.label_smoothing_factor
        )

    def predict(self, test_dataset: HFDataset, **kwargs):
        """Method for running inference on logbert to predict anomalous loglines in test dataset.

        :param test_dataset: test dataset of type huggingface Dataset object.
        :return: dict containing instance-wise loss and scores.
        """
        if not self.model:
            self.load_model()

        test_labels = {k: v for k, v in enumerate(list(test_dataset[constants.LABELS]))}

        if constants.LOG_COUNTS in test_dataset:
            test_counts = {
                k: v for k, v in enumerate(list(test_dataset[constants.LOG_COUNTS]))
            }
        else:
            test_counts = None

        mlm_dataset_test = test_dataset.map(
            self._generate_masked_input,
            with_indices=True,
            batched=True,
            batch_size=1,
            num_proc=1,
            remove_columns=test_dataset.column_names,
        )

        # List to keep track of anomaly predictions for each instance
        num_shards = self.config.num_eval_shards
        for i in range(num_shards):
            test_masked_lm_shard = mlm_dataset_test.shard(
                num_shards=num_shards, index=i
            )
            test_results = self.predictor.predict(test_masked_lm_shard)
            logging.info(
                "test_loss: {} test_runtime: {} test_samples/s: {}".format(
                    test_results.metrics["test_loss"],
                    test_results.metrics["test_runtime"],
                    test_results.metrics["test_samples_per_second"],
                )
            )


            data_columns = [
                "indices",
                "max_loss",
                "sum_loss",
                "num_loss",
                "top6_loss",
                "top6_max_prob",
                "top6_min_logprob",
                "top6_max_entropy",
            ]
            eval_metrics_per_instance_series = pd.DataFrame(
                np.transpose(
                    np.array(
                        self.predictor.label_smoother.eval_metrics_per_instance,
                        dtype=object,
                    )
                ),
                index=range(
                    len(self.predictor.label_smoother.eval_metrics_per_instance[0])
                ),
                columns=data_columns,
            )
            logging.info(
                "number of original test instances {}".format(
                    len(eval_metrics_per_instance_series.groupby("indices"))
                )
            )

            if i % 2 == 0 and test_labels is not None:
                compute_metrics(
                    eval_metrics_per_instance_series, test_labels, test_counts, output_dir=self.config.output_dir
                )

        try:
            compute_auc_roc_for_metric(
                y=ground_truth,
                metric=prediction_score,
                metric_name_str="overall_end_scores_top6_prob",
                plot_graph=True,
                plot_histogram=True,
                output_dir=self.config.output_dir
            )
        except Exception as err:
            print(err)

        try:
            with open(os.path.join(self.config.output_dir, "outcome"), "a") as fp:
                for idx in range(len(ground_truth)):
                    if idx < len(prediction_score):
                        content = "%s,  %s" % (ground_truth[idx], prediction_score[idx])
                        fp.write(content + "\n")
        except Exception as err:
            print(err)

        del ground_truth[:], prediction_score[:]

        return eval_metrics_per_instance_series


class LogBERTMultiClassificationPredict:
    """Class for running inference on logBERT model for unsupervised log anomaly detection.

    :param config: config object describing the parameters of logbert model.
    """

    def __init__(self, config: LogBERTConfig):

        self.config = config
        self.model_dirpath = os.path.join(
            self.config.output_dir, self.config.model_name
        )

        self.model = None
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased", max_length=512)

    def load_model(self, **kwargs):
        """Loading logbert model from the model dir path as specified in the logBERTConfig config"""
        checkpoint_dir = "checkpoint-" + str(
            max(
                [
                    int(x.split("-")[1])
                    for x in os.listdir(self.model_dirpath)
                    if x.startswith("checkpoint")
                ]
            )
        )
        model_checkpoint = os.path.abspath(
            os.path.join(self.model_dirpath, checkpoint_dir)
        )
        logging.info("Loading model from {}".format(model_checkpoint))

        self.model = BertForSequenceClassification.from_pretrained(model_checkpoint,
                                                                   num_labels=kwargs.get("num_labels"),
                                                                   id2label=kwargs.get("id2label"),
                                                                   label2id=kwargs.get("label2id"))
        self.model.to(kwargs.get("device"))

    def predict(self, data, **kwargs):
        """Method for running inference on logbert to predict anomalous loglines in test dataset.

        :param test_dataset: test dataset of type huggingface Dataset object.
        :return: dict containing instance-wise loss and scores.
        """
        if not self.model:
            self.load_model(**kwargs)

        texts = []
        results = []
        for text, source_file, log_level in zip(data.body[constants.LOGLINE_NAME],
                                                data.body[constants.SOURCE_FILE],
                                                data.severity_number[constants.LOG_SEVERITY_NUMBER]):
            text = f"{text} [Log Level: {log_level}] [Source File: {source_file}]"
            texts.append(text)

            inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs.to(kwargs.get("device"))

            # Get model output (logits)
            outputs = self.model(**inputs)

            probs = outputs[0].softmax(1)
            """ Explanation outputs: The BERT model returns a tuple containing the output logits (and possibly other elements depending on the model configuration). In this case, the output logits are the first element in the tuple, which is why we access it using outputs[0].
    
            outputs[0]: This is a tensor containing the raw output logits for each class. The shape of the tensor is (batch_size, num_classes) where batch_size is the number of input samples (in this case, 1, as we are predicting for a single input text) and num_classes is the number of target classes.
    
            softmax(1): The softmax function is applied along dimension 1 (the class dimension) to convert the raw logits into class probabilities. Softmax normalizes the logits so that they sum to 1, making them interpretable as probabilities. """

            # Get the index of the class with the highest probability
            # argmax() finds the index of the maximum value in the tensor along a specified dimension.
            # By default, if no dimension is specified, it returns the index of the maximum value in the flattened tensor.
            pred_label_idx = probs.argmax()

            # Now map the predicted class index to the actual class label
            # Since pred_label_idx is a tensor containing a single value (the predicted class index),
            # the .item() method is used to extract the value as a scalar
            pred_label = self.model.config.id2label[pred_label_idx.item()]
            results.append([probs, pred_label_idx, pred_label])

        return results