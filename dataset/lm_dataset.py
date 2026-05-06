#!/usr/bin/python
# -*- coding: utf-8 -*-


from os import truncate
from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainingDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    # 我们拿到的是，jsonl里的每一行
    def __getitem__(self, index):
        sample = self.samples[index]

        # tokenizer把文本转化为input_id
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids

        # 需要加上EOS，BOS，以及PAD填充
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # 需要自行编写labels，防止PAD参与loss计算
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # 需要编写attention_mask，告诉模型哪些位置是有效的，哪些位置是PAD
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # 我们要输出的，是input_ids, attention_mask, labels
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
