# -*- coding: utf-8 -*-
# Author: zhaoguangpu@bytedance.com
# Created: 2023/2/06
from collections import namedtuple


NONE_LABEL="无"
NluLoss = namedtuple("NluLoss", ("total_loss", "intent_loss", "slot_loss", "kl_loss_intent", "kl_loss_slot"))
NluLogits = namedtuple("NluLogits", ("intent_logits", "slot_logits"))
InputFeatures = namedtuple("InputFeatures", ("input_ids", "input_mask", "segment_ids", "intent_label_ids",
                                             "slot_label_ids", "input_len", "speaker_flag"))