from collections import namedtuple
import copy
import json
import logging
import os

import torch
import numpy as np
from configs.label_config import get_label_config
from configs.constant_parmas import InputFeatures
from processors.task_utils import truncate_maxlen_with_first_speaker, truncate_maxlen_punctuation, encode_segment_ids

from .utils import DataProcessor, get_markup

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, text_a, intent_labels, slot_labels, cur_start_idx=None, cur_end_idx=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.cur_start_idx = cur_start_idx
        self.cur_end_idx = cur_end_idx
        self.intent_labels = intent_labels
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


class NluInputExample(InputExample):
    """A single training/test example for joint nlu."""

    def __init__(self, guid, query, context, intent_labels, slot_labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.query = query
        self.context = context
        self.intent_labels = intent_labels
        self.slot_labels = slot_labels


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attent_mask, all_token_type_ids, all_len, all_intent_label_ids, all_slot_label_ids, \
        all_speaker_segment_ids = map(torch.stack, zip(*batch))

    return all_input_ids, all_attent_mask, all_token_type_ids, all_len, all_intent_label_ids, all_slot_label_ids, \
        all_speaker_segment_ids


def convert_examples_to_features(examples, tokenizer, intent_label_list=[], slot_label_list=[], max_seq_length=512,
                                 max_context_size=5, max_context_len=100, max_query_len=50, args=None, in_context=True):
    slot_label_map = args.slot_label2id

    special_tokens_dict = {'additional_special_tokens': ['[USR]', '[SYS]', '[CAR]', '[HOME]', '[REALTY]', '[FINANCE]', '[EDU]', '[LIFE]', '[PHOTO]', '[BIZ]', '[INVEST]', '[CULTURE]', '[HEALTH]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    features = []
    slot_label_ids = None
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        if args.add_usr:
            query = example.query[-max_query_len + 1:]
            query = ['[USR]'] + list(query)
        else:
            query = example.query[-max_query_len:]
        if in_context:
            # ["上下文1", "上下文2", "上下文3"]
            context = example.context["text"][-max_context_size:]
            roles = example.context["roles"][-max_context_size:]

            concate_context = []
            for text, role in zip(context, roles):
                role_special_token = "[USR]" if role == 0 else "[SYS]"
                concate_context.append(role_special_token)
                concate_context.extend(list(text))
            # 按照标点符号或者USR SYS标识截断上下文
            if args.truncate_methods == 'punctuation':
                concate_context = truncate_maxlen_punctuation(
                    concate_context, max_context_len)
            elif args.truncate_methods == 'hard_truncate':
                concate_context = truncate_maxlen_with_first_speaker(
                    concate_context, max_context_len)
            else:
                concate_context = concate_context[-max_context_len:]

            # if args.slot_decoder == "global_pointer":
            #     concate_context += ["[PAD]"] * (max_context_len - len(concate_context))
            # FIXME
            concate_context += ["[PAD]"] * \
                (max_context_len - len(concate_context))
            tokens = tokenizer(" ".join(concate_context), " ".join(list(query)), padding='max_length', truncation=True,
                               max_length=max_seq_length)
        else:
            query = example.query[-max_seq_length:]
            tokens = tokenizer(" ".join(list(query)), padding='max_length', truncation=True,
                               max_length=max_seq_length)
        input_ids = tokens["input_ids"]
        # [sequence_a_segment_id] * len(input_ids)
        segment_ids = tokens["token_type_ids"]
        input_mask = tokens["attention_mask"]

        segment_speaker_ids = []
        if args.speaker_segment:
            segment_speaker_ids = encode_segment_ids(input_ids, tokenizer)
        if args.model_type in ["bert_slot"]:
            intent_label_ids = []
        elif args.multi_label:
            intent_label_ids = [0] * len(intent_label_list)
            intent_label_ids = np.zeros(len(intent_label_list), dtype=np.uint8)
            for intent in example.intent_labels:
                intent_label_ids[intent_label_list.index(intent)] = 1
        else:
            # FIXME: 默认标签
            intent_label_ids = [1] if len(example.intent_labels) == 0 else intent_label_list.index(example.intent_labels[0])

        slot_labels = example.slot_labels[-max_query_len:]
        if args.model_type in ["bert_intent"]:
            slot_label_ids = []
        elif args.markup == "span":
            slot_label_ids = np.zeros(
                (len(slot_label_list), max_query_len, max_query_len), dtype=np.uint8)
            for span in example.slot_labels:
                if span.end >= max_query_len:
                    continue
                slot_label_ids[args.slot_label2id[span.slot_name]][span.start][span.end] = 1
        else:
            # print(f"slot_labels: {slot_labels}")
            # [USR] + ...
            if args.add_usr:
                slot_labels = example.slot_labels[-max_query_len + 1:]
                slot_labels = ["X"] + slot_labels
            slot_labels += ["X"] * (max_query_len - len(slot_labels))
            slot_label_ids = [slot_label_map.get(x) for x in slot_labels]


        input_len = len(input_ids)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if ex_index < 5:
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            logger.info("*** Example ***")
            logger.info(f"guid: {example.guid}")
            logger.info(f"tokens: {' '.join([str(x) for x in tokens])}")
            logger.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
            logger.info(f"input_mask: {' '.join([str(x) for x in input_mask])}")
            logger.info(f"segment_ids: {' '.join([str(x) for x in segment_ids])}")
            # logger.info(f"intent_label_ids: {' '.join([str(x) for x in intent_label_ids])}")
            logger.info(f"intent_label_ids: {intent_label_ids}")
            logger.info(f"slot_labels: {' '.join([str(x) for x in slot_labels])}")
            logger.info(f"slot_label_ids: {' '.join([str(x) for x in slot_label_ids])}")

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,
                                      segment_ids=segment_ids, input_len=input_len,
                                      intent_label_ids=intent_label_ids,
                                      slot_label_ids=slot_label_ids, speaker_flag=segment_speaker_ids))
    return features


class NluProcessor(DataProcessor):
    """Processor for joint nlu data set."""

    def __init__(self, task_name="default", slot_markup="bio", model_type="bert_nlu") -> None:
        super().__init__()
        self.task_name = task_name
        self.slot_markup = slot_markup
        self.model_type = model_type

    def get_train_examples(self, data_dir, in_context=True, prefix=None):
        """See base class."""
        file_name = f"{prefix}_train.json" if prefix else "train.json"
        return self._create_examples(self._read_json(os.path.join(data_dir, file_name)), "train",
                                     markup=self.slot_markup, in_context=in_context)

    def get_dev_examples(self, data_dir, in_context=True, prefix=""):
        """See base class."""
        file_name = f"{prefix}_dev.json" if prefix else "dev.json"
        return self._create_examples(self._read_json(os.path.join(data_dir, file_name)), "dev",
                                     markup=self.slot_markup, in_context=in_context)

    def get_test_examples(self, data_dir, in_context=True, prefix=""):
        """See base class."""
        file_name = f"{prefix}_test.json" if prefix else "test.json"
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test",
                                     markup=self.slot_markup, in_context=in_context)

    def get_labels(self, slot_markup="bio"):
        """See base class."""
        label_config = get_label_config(self.task_name)
        intent_labels = self.get_intent_labels(label_config.get("ind_intent_categories"))
        slot_raw_labels = label_config.get("slot_categories", [])
        slot_labels = self.get_slot_labels(slot_raw_labels, slot_markup)
        return intent_labels, slot_labels

    def get_intent_labels(self, intent_labels):
        return intent_labels

    def get_slot_labels(self, raw_labels, markup="bio"):
        markup = markup.lower()
        assert markup in ["bio", "bios", "span", "bioes"]
        scale_map = {'bio': 2, 'bios': 3, 'bioes': 4}
        if markup == "span" or self.model_type in ["bert_intent"]:
            return raw_labels
        slot_labels = ["X", "O"]
        for label in raw_labels:
            slot_labels.append(f"B-{label}")
            slot_labels.append(f"I-{label}")
            if markup == "bios":
                slot_labels.append(f"S-{label}")
            elif markup == 'bioes':
                slot_labels.append(f"S-{label}")
                slot_labels.append(f"E-{label}")
        legal_nums = len(raw_labels) * scale_map[markup]
        legal_nums += 2  # "O" and "X"
        assert legal_nums == len(slot_labels)

        return slot_labels

    def _create_examples(self, lines, set_type, markup="bio", in_context=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = f"{set_type}-{i}"
            query = line["query"]
            context = line["context"] if in_context else None
            intent_labels = line["label"]["intent"]
            slot_labels = line["label"]["slot"]
            # convert raw label to bio, bios or span type label
            slot_labels = get_markup(markup, len(query), slot_labels, self.model_type)
            examples.append(NluInputExample(guid=guid, query=query, context=context,
                                            intent_labels=intent_labels,
                                            slot_labels=slot_labels))

        return examples


class IntentProcessor(DataProcessor):
    """Processor for intent data set."""

    def __init__(self, task_name="default") -> None:
        super().__init__()
        self.task_name = task_name

    def get_train_examples(self, data_dir, in_context=True):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train", in_context=in_context)

    def get_dev_examples(self, data_dir, in_context=True):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev", in_context=in_context)

    def get_test_examples(self, data_dir, in_context=True):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test", in_context=in_context)

    def get_labels(self):
        """See base class."""
        label_config = get_label_config(self.task_name)
        return label_config["ind_intent_categories"]

    def _create_examples(self, lines, set_type, in_context=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = f"{set_type}-{i}"
            query = line["query"]
            context = line["context"] if in_context else None
            intent_labels = line["label"]["intent"]
            # convert raw label to bio, bios or span type label
            examples.append(NluInputExample(guid=guid, query=query, context=context,
                                            intent_labels=intent_labels,
                                            slot_labels=None))

        return examples


task_processors = {
    "default": NluProcessor,
    "ind_car_nlu": NluProcessor,
    "comment_intention": NluProcessor,
    "ind_car2_nlu": NluProcessor,
    "ind_car_v2_nlu": NluProcessor,
    "ind_car_slot": NluProcessor,
    "ind_car_intent": NluProcessor,
    "ind_home_nlu": NluProcessor
}

def get_processor(task_name):
    assert task_name in task_processors
    return task_processors[task_name]