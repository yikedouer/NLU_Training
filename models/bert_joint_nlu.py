import logging

import torch
import torch.nn as nn
from losses.focal_loss import MultiClassFocalLoss, MultiLabelFocalLoss
from losses.circle_loss import MultiLabelCircleLoss
from losses.kl_divergence import KLDivergenceLoss, MultiLabelCrossEntropyKLD
from losses.label_smoothing import LabelSmoothingCrossEntropy
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel

from models.layers.linears import IntentClassifier
from models.layers.slot_decoder import get_slot_decoder
from configs.constant_parmas import NluLogits, NluLoss



torch.set_printoptions(profile="full")

logger = logging.getLogger(__name__)

LOSS_MAP = {
    "multi_label_focal_loss": MultiLabelFocalLoss(),
    "multi_class_focal_loss": MultiClassFocalLoss(ignore_index=-1),
    "multi_label_circle_loss": MultiLabelCircleLoss(),
    "label_smooth_ce": LabelSmoothingCrossEntropy(),
    "ce": CrossEntropyLoss(ignore_index=-1),
    "bce": nn.BCEWithLogitsLoss(),
}

class BertJointNlu(BertPreTrainedModel):
    def __init__(self, config, args, num_intent_labels, num_slot_labels, do_export=False):
        super(BertJointNlu, self).__init__(config)
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.intent_loss_coef = args.intent_loss_coef
        self.slot_loss_coef = args.slot_loss_coef
        self.kl_loss_slot_coef = args.kl_loss_slot_coef
        self.kl_loss_intent_coef = args.kl_loss_intent_coef
        self.slot_loss_type = args.slot_loss_type
        self.intent_loss_type = args.intent_loss_type
        self.do_rdrop = args.do_rdrop
        self.do_export = do_export
        
        self.bert = BertModel(config)
        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_decoder = get_slot_decoder(args.slot_decoder, config.hidden_size, num_slot_labels, args.dropout_rate)
        
        self.intent_loss_fct = LOSS_MAP[self.intent_loss_type]
        if self.do_rdrop:
            self.intent_kl_fct = MultiLabelCrossEntropyKLD() if self.intent_loss_type == "multi_label_circle_loss" else KLDivergenceLoss()
            self.slot_kl_fct = MultiLabelCrossEntropyKLD() if self.slot_loss_type == "multi_label_circle_loss" else KLDivergenceLoss()
        
        self.post_init()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                intent_labels_ids=None, slot_labels_ids=None, is_train=True):
  
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output, sequence_output = bert_outputs[1], bert_outputs[0]
        slot_logits, slot_mask = sequence_output[:,102:-1,:], attention_mask[:,102:-1]
        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_decoder(slot_logits, mask=slot_mask)
        if self.do_export:
            return (intent_logits, slot_logits)

        bert_outputs2 = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) if self.do_rdrop else None
        total_loss, slot_loss, intent_loss, kl_loss_intent, kl_loss_slot = 0.0, 0.0, 0.0, 0.0, 0.0

        if intent_labels_ids is not None:
            intent_labels_ids = intent_labels_ids.float()
            intent_loss = self.intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_labels_ids.view(-1, self.num_intent_labels))
            if self.do_rdrop and is_train:
                intent_logits2 = self.intent_classifier(bert_outputs2[1])
                intent_loss2 = self.intent_loss_fct(intent_logits2.view(-1, self.num_intent_labels), intent_labels_ids.view(-1, self.num_intent_labels))
                intent_loss = 0.5 * (intent_loss + intent_loss2)
                kl_loss_intent = self.intent_kl_fct(intent_logits2, intent_logits, reduction="sum")
        
        if slot_labels_ids is not None:
            slot_loss = self.slot_decoder.compute_loss(slot_logits, slot_labels_ids, slot_mask)
            if self.do_rdrop and is_train and self.slot_decoder.name in ["softmax", "global_pointer"]:
                slot_logits2 = bert_outputs2[0][:, 102:-1, :]
                slot_logits2 = self.slot_decoder(slot_logits2, mask=slot_mask)
                slot_loss2 = self.slot_decoder.compute_loss(slot_logits2, slot_labels_ids, slot_mask)
                kl_loss_slot = self.slot_kl_fct(slot_logits2, slot_logits, reduction="sum")
                slot_loss = 0.5 * (slot_loss + slot_loss2) 
            slot_loss =  slot_loss * self.slot_loss_coef
            total_loss = intent_loss * self.intent_loss_coef + slot_loss
            if self.do_rdrop:
                kl_loss_intent *= self.kl_loss_intent_coef
                kl_loss_slot *= self.kl_loss_slot_coef
                total_loss += kl_loss_intent  + kl_loss_slot

        nlu_loss = NluLoss(total_loss=total_loss, intent_loss=intent_loss, slot_loss=slot_loss, kl_loss_intent=kl_loss_intent, kl_loss_slot=kl_loss_slot)
        nlu_logits = NluLogits(intent_logits=intent_logits, slot_logits=slot_logits)
        return (nlu_loss, nlu_logits)


    def resize_token_embeddings(self, re_token_size):
        self.bert.resize_token_embeddings(re_token_size)
