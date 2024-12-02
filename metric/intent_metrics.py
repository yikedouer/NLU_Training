from sklearn.metrics import classification_report as report
from collections import OrderedDict


class IntentMetrics(object):
    def __init__(self, label_list):
        self.label_list = label_list
    
    def compute(self, true_labels, pred_labels, digits=3):
        result = report(true_labels, pred_labels, target_names=self.label_list, output_dict=True)
        if "accuracy" in result:
            result.pop("accuracy")
        result = OrderedDict(sorted(result.items(), key=lambda x: x[1]["support"], reverse=True))
        for clz, info in result.items():
            if not isinstance(info, dict):
                continue
            info = {k: round(v, digits) for k, v in info.items()}
            result[clz] = info

        class_info = {k: v for k, v in result.items() if k not in ["micro avg", "macro avg", "weighted avg", "samples avg"]}
        micro_info = result.get("micro avg")
        macro_info = result.get("macro avg")
        
        return micro_info, macro_info, class_info
