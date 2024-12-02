
from models.bert_joint_nlu import BertJointNlu
from models.bert_intent_classifier import BertIntentClassifier
from models.bert_slot_tagger import BertSlottagger
from transformers import BertConfig, AutoTokenizer


MODEL_CLASSES = {
    ## bert
    'bert_nlu': (BertConfig, BertJointNlu, AutoTokenizer),
    'bert_intent': (BertConfig, BertIntentClassifier, AutoTokenizer),
    'bert_slot': (BertConfig, BertSlottagger, AutoTokenizer),
    # 'albert': (AlbertConfig, AlbertCrfForNer, CNerTokenizer)
}