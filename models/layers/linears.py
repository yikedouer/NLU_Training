import torch.nn as nn
import torch


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class MARCLinear(nn.Module):
    """
    A wrapper for nn.Linear with support of MARC method.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.a = torch.nn.Parameter(torch.ones(1, out_features))
        self.b = torch.nn.Parameter(torch.zeros(1, out_features))
    
    def forward(self, input, *args):
        with torch.no_grad():
            logit_before = self.fc(input)
            w_norm = torch.norm(self.fc.weight, dim=1)
        logit_after = self.a * logit_before + self.b * w_norm
        return logit_after