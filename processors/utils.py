import csv
import json
from collections import namedtuple
import numpy as np

SpanSlot = namedtuple("SpanSlot", ["slot_name", "start", "end"])

class DataProcessor(object):
    """Base class for data converters."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_text(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines

    @classmethod
    def _read_json(self, input_file,):
        data = []
        with open(input_file, "r") as f:
            data = json.load(f)
        return data

def get_entity_bios(seq, id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    chunks = [[chunk[0], chunk[1], chunk[2]+1] for chunk in chunks] 
    return chunks


def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
        
    chunks = [[chunk[0], chunk[1], chunk[2]+1] for chunk in chunks] 
    return chunks

def get_entity_span(seq, id2label, threshold=0):
    entities = []
    for l, start, end in zip(*np.where(seq > threshold)):
        entities.append((id2label[l], start, end))

    return entities

def get_entities(seq, id2label, markup='bios', threshold=0):
    """
    :param seq:
    :param id2label:
    :param markup:
    :return:
    """
    assert markup in ['bio', 'bios', 'span']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    elif markup == 'span':
        return get_entity_span(seq, id2label, threshold=threshold)
    else:
        return get_entity_bios(seq, id2label)


def get_markup(markup, length, entities=[], model_type="bert_nlu"):
    """convert entities(start_idx, end_index, entity name) to markup(bio, bios)
    params:
        markup: 标注形式，bio, bios
        words：将json文件每一行中的文本分离出来，存储为words列表
        labels：标记文本对应的标签，存储为labels
    examples:
        words示例：['生', '生', '不', '息', 'C', 'S', 'O', 'L']
        labels示例：['O', 'O', 'O', 'O', 'B-game', 'I-game', 'I-game', 'I-game']
    """
    if model_type in ["bert_intent"]:
        return []
    assert markup in ["bio", "bios", "span", None]
    if markup == "span":
        label_list = []
        for entity_map in entities:
            name, value = list(entity_map.items())[0]
            for sub_name, sub_index in value.items():
                label_list.append(SpanSlot(name, sub_index[0], sub_index[1]))
        return label_list
    label_list = ['O'] * length
    for entity_map in entities:
        name, value = list(entity_map.items())[0]
        for sub_name, sub_index in value.items():
            start, end = sub_index[0], sub_index[1]
            if markup.lower() == "bios":
                # bios
                if start == end:
                    label_list[start] = 'S-' + name
                else:
                    label_list[start] = 'B-' + name
                    label_list[start + 1: end] = ['I-' + name] * (len(sub_name) - 1)
            else:
                # bio
                label_list[start] = 'B-' + name
                # else:
                label_list[start + 1: end] = ['I-' + name] * (len(sub_name) - 1)

    return label_list
