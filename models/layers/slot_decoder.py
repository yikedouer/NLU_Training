import torch.nn as nn
from torchcrf import CRF
from models.layers.global_pointer import EfficientGlobalPointer
from models.layers.linears import SlotClassifier
from losses.circle_loss import MultiLabelCircleLoss
from torch.nn import CrossEntropyLoss
import torch


class SlotDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        raise NotImplementedError

    def compute_loss(self, logits, labels, mask):
        return self.loss_fct(logits, labels)


class CrfDecoder(SlotDecoder):
    def __init__(self, hidden_size, num_slot_labels, dropout_rate) -> None:
        super().__init__()
        self.classifier = SlotClassifier(hidden_size, num_slot_labels, dropout_rate)
        self.crf = CRF(num_tags=num_slot_labels, batch_first=True)
        self.name = "crf"

    def forward(self, logits, mask):
        logits = self.classifier(logits)
        return logits
    
    def decode(self, slot_logits, mask):
        return self.crf.decode(slot_logits, mask.byte())

    def compute_loss(self, logits, labels, mask):
        labels = labels.type(torch.long)
        return -1 * self.crf(logits, labels, mask=mask.byte(), reduction="mean")

class GlobalPointerDecoder(SlotDecoder):
    def __init__(self, hidden_size, num_slot_labels, dropout_rate) -> None:
        super().__init__()
        self.global_pointer = EfficientGlobalPointer(num_slot_labels, head_size=64, hidden_size=hidden_size, RoPE=True)
        self.loss_fct = MultiLabelCircleLoss()
        self.name = "global_pointer"

    def forward(self, logits, mask):
        logits = self.global_pointer(logits, mask=mask)
        return logits
    
    def decode(self, logits, mask):
        return logits.cpu().numpy()


class SoftmaxDecoder(SlotDecoder):
    def __init__(self, hidden_size, num_slot_labels, dropout_rate) -> None:
        super().__init__()
        self.num_slot_labels = num_slot_labels
        self.classifier = SlotClassifier(hidden_size, self.num_slot_labels, dropout_rate)
        self.loss_fct = CrossEntropyLoss(ignore_index=-1)
        self.name = "softmax"
    
    def forward(self, logits, mask):
        logits = self.classifier(logits)
        return logits.contiguous().view(-1, self.num_slot_labels)
    
    def decode(self, logits, mask):
        recover_shape = tuple(mask.shape) + (-1,)
        logits = logits.reshape(recover_shape)
        tags = torch.argmax(logits, axis=2)
        tags = tags * mask
        return tags.cpu().numpy().tolist()

    def compute_loss(self, logits, labels, mask):
        labels = labels.contiguous().view(-1).type(torch.long)
        return self.loss_fct(logits, labels)
    

def get_slot_decoder(name, hidden_size, num_slot_labels, dropout_rate):
    if name == "crf":
        return CrfDecoder(hidden_size, num_slot_labels, dropout_rate)
    if name == "global_pointer":
        return GlobalPointerDecoder(hidden_size, num_slot_labels, dropout_rate)
    if name == "softmax":
        return SoftmaxDecoder(hidden_size, num_slot_labels, dropout_rate)
    raise NotImplementedError