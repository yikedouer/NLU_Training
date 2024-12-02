import logging

import torch
import torch.nn as nn
from losses.focal_loss import MultiClassFocalLoss, MultiLabelFocalLoss
from losses.circle_loss import MultiLabelCircleLoss
from losses.kl_divergence import KLDivergenceLoss, MultiLabelCrossEntropyKLD
from losses.label_smoothing import LabelSmoothingCrossEntropy
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel

from models.layers.slot_decoder import get_slot_decoder
from configs.constant_parmas import NluLogits, NluLoss



torch.set_printoptions(profile="full")

logger = logging.getLogger(__name__)


class BertSlottagger(BertPreTrainedModel):
    def __init__(self, config, args, num_slot_labels, do_export=False):
        super(BertSlottagger, self).__init__(config)
        self.num_slot_labels = num_slot_labels
        self.slot_loss_coef = args.slot_loss_coef
        self.kl_loss_slot_coef = args.kl_loss_slot_coef
        self.slot_loss_type = args.slot_loss_type
        self.do_rdrop = args.do_rdrop
        self.do_export = do_export
        
        self.bert = BertModel(config)
        self.slot_decoder = get_slot_decoder(args.slot_decoder, config.hidden_size, num_slot_labels, args.dropout_rate)
        
        if self.do_rdrop:
            self.slot_kl_fct = MultiLabelCrossEntropyKLD() if self.slot_loss_type == "multi_label_circle_loss" else KLDivergenceLoss()
        
        self.post_init()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                slot_labels_ids=None, is_train=True):
  
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = bert_outputs[0]
        slot_logits, slot_mask = sequence_output[:,102:-1,:], attention_mask[:,102:-1]
        slot_logits = self.slot_decoder(slot_logits, mask=slot_mask)
        if self.do_export:
            return slot_logits

        bert_outputs2 = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) if self.do_rdrop else None
        total_loss, slot_loss, kl_loss_slot = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        
        if slot_labels_ids is not None:
            slot_loss = self.slot_decoder.compute_loss(slot_logits, slot_labels_ids, slot_mask)
            if self.do_rdrop and is_train and self.slot_decoder.name in ["softmax", "global_pointer"]:
                slot_logits2 = bert_outputs2[0][:, 102:-1, :]
                slot_logits2 = self.slot_decoder(slot_logits2, mask=slot_mask)
                slot_loss2 = self.slot_decoder.compute_loss(slot_logits2, slot_labels_ids, slot_mask)
                kl_loss_slot = self.slot_kl_fct(slot_logits2, slot_logits, reduction="sum")
                slot_loss = 0.5 * (slot_loss + slot_loss2)

            total_loss = slot_loss * self.slot_loss_coef
            if self.do_rdrop:
                kl_loss_slot *= self.kl_loss_slot_coef
                total_loss += kl_loss_slot

        nlu_loss = NluLoss(total_loss=total_loss, intent_loss=0.0, slot_loss=slot_loss, kl_loss_intent=0.0, kl_loss_slot=kl_loss_slot)
        nlu_logits = NluLogits(intent_logits=None, slot_logits=slot_logits)
        return (nlu_loss, nlu_logits)


    def resize_token_embeddings(self, re_token_size):
        self.bert.resize_token_embeddings(re_token_size)
