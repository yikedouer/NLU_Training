from collections import Counter, OrderedDict


class SlotMetrics(object):
    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right, digit=3):
        recall = 0.0 if origin == 0 else (right / origin)
        precision = 0.0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0.0 else (2 * precision * recall) / (precision + recall)
        recall = round(recall, digit)
        precision = round(precision, digit)
        f1 = round(f1, digit)
        
        return recall, precision, f1

    def result(self, digit=3):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])

        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right, digit)
            class_info[type_] = {"precision": precision, 'recall': recall, 'f1': f1, 'support': count, "found": found, "right": right}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right, digit)
        class_info = OrderedDict(sorted(class_info.items(), key=lambda x: x[1]["support"], reverse=True))
        micro_info = {'precision': precision, 'recall': precision, 'f1-score': precision, 'support': len(self.origins)}
        # FIXME: 算下macro
        # macro_info = 0
        return micro_info, class_info

    def update(self, true_entities, pre_entities):
        self.origins.extend(true_entities)
        self.founds.extend(pre_entities)
        self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in true_entities])
