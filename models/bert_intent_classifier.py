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
from configs.constant_parmas import NluLogits, NluLoss


torch.set_printoptions(profile="full")

logger = logging.getLogger(__name__)

LOSS_MAP = {
    "multi_label_focal_loss": MultiLabelFocalLoss(),
    "multi_class_focal_loss": MultiClassFocalLoss(ignore_index=-1),
    "multi_label_circle_loss": MultiLabelCircleLoss(),
    "label_smooth_ce": LabelSmoothingCrossEntropy(),
    "ce": CrossEntropyLoss(),
    # "ce": CrossEntropyLoss(weight=torch.tensor([6, 5])),
    "bce": nn.BCEWithLogitsLoss(),
}

class BertIntentClassifier(BertPreTrainedModel):
    def __init__(self, config, args, num_intent_labels, do_export=False):
        super(BertIntentClassifier, self).__init__(config)
        self.num_intent_labels = num_intent_labels
        self.intent_loss_coef = args.intent_loss_coef
        self.kl_loss_intent_coef = args.kl_loss_intent_coef
        self.intent_loss_type = args.intent_loss_type
        self.do_rdrop = args.do_rdrop
        self.do_export = do_export
        
        self.bert = BertModel(config)
        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        
        self.intent_loss_fct = LOSS_MAP[self.intent_loss_type]
        if self.do_rdrop:
            self.intent_kl_fct = MultiLabelCrossEntropyKLD() if self.intent_loss_type == "multi_label_circle_loss" else KLDivergenceLoss()
        
        self.post_init()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                intent_labels_ids=None, is_train=True):
  
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = bert_outputs[1]
        intent_logits = self.intent_classifier(pooled_output)
        
        if self.do_export:
            return intent_logits

        bert_outputs2 = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) if self.do_rdrop else None
        total_loss, intent_loss, kl_loss_intent = 0.0, 0.0, 0.0

        if intent_labels_ids is not None:
            intent_labels_ids = intent_labels_ids.type(torch.LongTensor).cuda()
            intent_loss = self.intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_labels_ids)
            if self.do_rdrop and is_train:
                intent_logits2 = self.intent_classifier(bert_outputs2[1])                
                intent_loss2 = self.intent_loss_fct(intent_logits2.view(-1, self.num_intent_labels), intent_labels_ids)
                intent_loss = 0.5 * (intent_loss + intent_loss2)
                kl_loss_intent = self.intent_kl_fct(intent_logits2, intent_logits, reduction="sum")
        

            total_loss = intent_loss * self.intent_loss_coef
            if self.do_rdrop:
                kl_loss_intent *= self.kl_loss_intent_coef
                total_loss += kl_loss_intent 
        nlu_loss = NluLoss(total_loss=total_loss, intent_loss=intent_loss, slot_loss=0.0, kl_loss_intent=kl_loss_intent, kl_loss_slot=0.0)
        nlu_logits = NluLogits(intent_logits=intent_logits, slot_logits=None)
        # print(f"nlu_loss: {nlu_loss}")
        # print(f"nlu_logits: {nlu_logits}")
        return (nlu_loss, nlu_logits)


    def resize_token_embeddings(self, re_token_size):
        self.bert.resize_token_embeddings(re_token_size)


class ExportModel(BertPreTrainedModel):
    def __init__(self, config, args, num_intent_labels):
        super(ExportModel, self).__init__(config)
        
        self.bert = BertModel(config)
        self.intent_classifier = IntentClassifier(config.hidden_size, num_intent_labels, args.dropout_rate)
        
        self.post_init()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
  
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = bert_outputs[1]
        intent_logits = self.intent_classifier(pooled_output)        
        return intent_logits