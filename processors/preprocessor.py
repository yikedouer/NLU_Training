# -*- coding: utf-8 -*-
# Author : zhaoguangpu@bytedance.com
# Created: 2022/12/06
import json
import os
import re
import sys
import random
from collections import Counter
from types import TracebackType

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("..")

from tools.common import load_pickle, save_pickle, save_json
from configs.label_config import get_label_config
from configs.constant_parmas import NONE_LABEL
from tools.common import init_logger, logger

init_logger(log_file=None)

chaos = []
n_seat = []
del_list = []
chexing = []
pay_way = []
budget = []
garbage = []
kucun = []
non_label = []
class IntentAdjuster(object):

    def __init__(self, intent_labels, sample_ratios):
        self.intent_labels = intent_labels
        self.sample_ratios = sample_ratios
        self.full_num_rule = re.compile("^\d+$")

    def adjust(self, intent_set, query):
        # 纯数字的query，清空intent
        if self.full_num_rule.match(query) is not None:
            return set()
        return intent_set

    def sampling(self, data):
        if self.sample_ratios is None:
            return data
        data_sampled = []
        for idx, sample in enumerate(data):
            ratio = random.random()
            if len(sample["label"]["intent"]) == 0:
                non_label.append((sample["context"]["text"], sample["query"]))
            if len(sample["label"]["intent"]) == 0 and len( sample["label"]["slot"]) == 0:
                if ratio <= self.sample_ratios[NONE_LABEL]:
                    data_sampled.append(sample)
                continue
            intents = []
            drop = False
            for intent in sample["label"]["intent"]: 
                if ratio > self.sample_ratios[intent] and len( sample["label"]["slot"]) == 0:
                    drop = True
                    break

            if not drop:
                data_sampled.append(sample)

        return data_sampled


class SlotAdjuster(object):

    def __init__(self, slot_labels, sample_ratios=None):
        self.slot_labels = slot_labels
        self.sample_ratios = sample_ratios
        self.full_num_rule = re.compile("^\d+$")

    def adjust(self, slot_info, query):
        return slot_info

    def sampling(self, data):
        if self.sample_ratios is None:
            return data
        data_sampled = []
        ratio = random.random()
        for idx, sample in enumerate(data):
            if len(sample["label"]["slot"]) == 0 and len(sample["label"]["intent"]) == 0:
                if ratio <= self.sample_ratios["无"]:
                    data_sampled.append(sample)
                continue
            slots = []
            drop = False
            for slot_info in sample["label"]["slot"]:
                slot_name = list(slot_info.keys())[0]
                if ratio > self.sample_ratios[slot_name] and len(sample["label"]["intent"]) == 0:
                    drop = True
                    break
            if not drop:
                data_sampled.append(sample)

        return data_sampled


class CarIntentAdjuster(IntentAdjuster):

    def __init__(self, intent_labels, sample_ratios=None):
        super().__init__(intent_labels, sample_ratios)
        self.intent_labels = intent_labels
        self.sample_ratios = sample_ratios

        self.phone_rule = re.compile('电话|打|手机|短信|信息')
        self.wechat_rule = re.compile('微信|加|v|V|vx')
        self.meaningless_rule = re.compile("(信息|详情|型号|配置|看|那个|款|发|多久|多少|哪|什么|吗|有|咨询|介绍|链接|描述|\?|？|咋|呢|吗|嘛|么|图片|了解|推荐|什么|资料|库存)")

    def adjust(self, intent_set, query):
        # 通用性的调整放在基类中
        intents = super().adjust(intent_set, query)
        adjusted_intents = set()
        for intent in intents:
            if re.match("^[a-zA-Z0-9$？]+$", query):
                continue
            if intent == "咨询购车意见":
                intent = "求介绍"
            elif intent == '问门店/销售联系方式':
                if re.search(self.wechat_rule, query):
                    intent = "问微信"
                elif re.search(self.phone_rule, query):
                    intent = "问电话"
            elif intent == "问二手车市价":
                intent = "问车价_落地价/裸车价"
            if intent not in self.intent_labels:
                continue
                # intent = "无"
            adjusted_intents.add(intent)
        if "问有哪些车系" in adjusted_intents and "求介绍" in adjusted_intents:
            adjusted_intents.remove("求介绍")
        if "求介绍" in adjusted_intents and len(query) < 6 and not self.meaningless_rule.search(query):
            adjusted_intents.remove("求介绍")
        if "问具体车系库存" in adjusted_intents:
            kucun.append(query)
        return list(adjusted_intents)


class CarSlotAdjuster(SlotAdjuster):

    def __init__(self, slot_labels, sample_ratios=None):
        super().__init__(slot_labels, sample_ratios)
        self.slot_labels = slot_labels
        self.sample_ratios = sample_ratios

        self.phone_kws = ["手机", "电话"]
        self.wechat_kws = ["微信", "vx", "+v", "威信", "v号"]
        self.gender_suffixs = ['士', '生', '孩儿', '孩']
        self.price_kws = ['左右', '块钱', '块', '元', '多', '以上', '以下', "上下", "以内", "以下", "之内", "之下", "之上", "内"]
        self.price_rm_prefixs = ["首付", "的预算", "预算", "大概", "落地", "的", "价格"]
        self.price_rm_suffixs = ["首付", "的预算", "块钱以下", "预算", "近年份的", "大点", "的", "落地", "落", "大", "价格", "，左右", "块钱", "一点点"]

        self.time_add_prefixs = ['等', '过']
        self.time_add_suffixs = ['以后', '之前', '前', '后']
        self.time_rm_prefixs = []
        self.time_rm_suffixs = ['左右', '再说', '联系', '吧', '儿', '打']
        
        self.seat_rule = re.compile("[一二三四五六七八九1-9][座坐做]")
        self.gearbox_rule = re.compile("(自动|手动)(挡)?")
        self.energy_rule = re.compile("(新能源|油电|混动|电动|纯电|电车|油车|柴油|燃油|汽油|增程|油电混|混合动力|电混|插电混动|插混|全油|燃油电混车|混合)(车)?")
        self.pay_way_kws = ["首付", "首富", "手付", "分期", "贷款", "货款", "代款" "月供", "按揭", "免息", "月共", "利息", "贷", "全款"]
        self.filter_list = ['平行进口车', '抵押车', '汽车', '三轮', '营运车', '宽体车', 'v行车', '国产车', '平行进口车', '抵押车', '汽车', '箱车', 'bba', '合资紧凑车', '库存车', '挡suv', '私家车', '摩托车', '运损车', '旅行车', '国产', '新车库存', '二手车', '大一点配套设施齐全', '下线车', '残疾人开', '带车棚', '营转非', '冷藏车', '紧凑型自动挡', '越野库存车', '中集大箱', '瓦罐', '法规车', '有点', '朗逸', '微朗', '板车', '美发车', '网约车', '骄车', '七对拉人', '娇车', '平板车', '挡轿车', '北汽', 'c', '老破小', '三轮互换', '五菱', '黑武士', '大头车', '床车', '非直营车型', '新车', '电四轮', '版本', '硬塞越野', '危化品版本', '不带充电庄']

    def adjust(self, contexts, query, slot_infos, nested=False):
        slot_all_map = {}
        for slot_info in slot_infos:
            slot_info = json.loads(slot_info)
            slot_all_map[slot_info["slot"]] = [slot_info["start"], slot_info["end"]]
        # auto slot
        groups = self.seat_rule.finditer(query)
        for group in groups:
            start, end = group.start(), group.end()
            # end = end - 1 if start != end else end
            seat_slot = {
                "start": start, "end": end, "slot": "座位数", "slot_value": query[start: end]
            }
            slot_infos.add(json.dumps(seat_slot, ensure_ascii=False))
            slot_all_map["座位数"] = [start, end]
            """{"start": 7, "end": 12, "slot": "意向车型", "slot_value": "朗逸星空版"}"""

        groups = self.gearbox_rule.finditer(query)
        for group in groups:
            start, end = group.start(), group.end()
            # end = end - 1 if start != end else end
            seat_slot = {
                "start": start, "end": end, "slot": "变速箱类型", "slot_value": query[start: end]
            }
            slot_infos.add(json.dumps(seat_slot, ensure_ascii=False))
            slot_all_map["变速箱类型"] = [start, end]
            """{"start": 7, "end": 12, "slot": "意向车型", "slot_value": "朗逸星空版"}"""
        groups = self.energy_rule.finditer(query)
        for group in groups:
            start, end = group.start(), group.end()
            # end = end - 1 if start != end else end
            seat_slot = {
                "start": start, "end": end, "slot": "能源类型", "slot_value": query[start: end]
            }
            slot_infos.add(json.dumps(seat_slot, ensure_ascii=False))
            slot_all_map["能源类型"] = [start, end]
            """{"start": 7, "end": 12, "slot": "意向车型", "slot_value": "朗逸星空版"}"""

        adjusted_slots = []

        has_wechat = True if "微信" in slot_all_map else False
        has_phone = True if "电话" in slot_all_map else False
        # if "自动" in query or "手动" in query:
        #     print(f"contains 自动或手动 in query: {query}")
        # if re.findall("[一二三四五六七八九1-9][座坐]", query):
        #     print(f"contains 座位 in query: {query}")
        # print(f"slot_infos: {slot_infos}")
        # if "奔驰glb" in query:
        #     print(f"query: {query}")
        #     print(f"raw slot_infos: {slot_infos}")
        for slot_info in slot_infos:
            slot_info = json.loads(slot_info)
            start, end, slot_name, slot_value = slot_info.values()
            if slot_name in ["意向车系", "意向车型", "意向品牌", "意向大类"]:
                groups = list(self.seat_rule.finditer(slot_value))
                text = ""
                for group in groups:
                    text += group.group()
                if text != "":
                    # print(f"query: {query}, slot_info: {slot_info}, text: {text}, slot_value: {slot_value}")
                    if slot_value.startswith(text):
                        slot_value = slot_value.replace(text, "")
                        end -= len(text)
                    elif slot_value.endswith(text):
                        slot_value = slot_value.replace(text, "")
                        start += len(text)
                    if len(slot_value) < 2:
                        continue
                groups = list(self.gearbox_rule.finditer(slot_value))
                text = ""
                for group in groups:
                    text += group.group()
                if text != "":
                    # print(f"query: {query}, slot_info: {slot_info}, text: {text}, slot_value: {slot_value}")
                    if slot_value.startswith(text):
                        slot_value = slot_value.replace(text, "")
                        end -= len(text)
                    elif slot_value.endswith(text):
                        slot_value = slot_value.replace(text, "")
                        start += len(text)
                    if len(slot_value) < 2:
                        continue

                groups = list(self.energy_rule.finditer(slot_value))
                text = ""
                for group in groups:
                    text += group.group()
                if text != "":
                    # print(f"query: {query}, slot_info: {slot_info}, text: {text}, slot_value: {slot_value}")
                    if slot_value.startswith(text):
                        slot_value = slot_value.replace(text, "")
                        end -= len(text)
                    elif slot_value.endswith(text):
                        slot_value = slot_value.replace(text, "")
                        start += len(text)
                    if len(slot_value) < 2:
                        continue
                    # print(f"query2: {query}, slot_info: {slot_info}, text: {text}, slot_value: {slot_value}")      
            # 同时标注微信和电话，优先电话
            if not nested and slot_name == "微信" and has_phone:
                continue
            if not nested and slot_name == "电话":
                should_phone = False
                for kw in self.phone_kws:
                    if kw in query:
                        should_phone = True
                        break
                if not should_phone:
                    contexts_reverse = contexts[::-1]
                    has_phone_kw = False
                    has_wechat_kw = False
                    for context in contexts_reverse:
                        for phone_kw in self.phone_kws:
                            if phone_kw in context:
                                has_phone_kw = True
                                break
                        for wechat_kw in self.wechat_kws:
                            if wechat_kw in context:
                                has_wechat_kw = True
                                break
                        if has_phone_kw and has_wechat_kw:
                            break
                        if has_wechat_kw and not has_phone_kw:
                            slot_name = "微信"
                            break

            match_obj = self.full_num_rule.match(query)
            if match_obj is not None and slot_name not in ["微信", "电话", "QQ", "年龄", "购车预算", "意向车系",
                                                           "意向车型", "二手车车况-里程", "意向出手价"]:
                continue
            if slot_name == "二手车车况_车型":
                slot_name = "意向车型"
            if slot_name not in slot_all_map:
                continue
            if slot_name == "购车时间":
                for suffix in self.time_rm_suffixs:
                    if query[start: end].endswith(suffix):
                        end -= len(suffix)
                        slot_value = slot_value.rstrip(suffix)
                        break
                for prefix in self.time_add_prefixs:
                    if query[:start].endswith(prefix):
                        start -= len(prefix)
                        slot_value = prefix + slot_value
                        break
                for suffix in self.time_add_suffixs:
                    if query[end:].startswith(suffix):
                        end += len(suffix)
                        slot_value += suffix
                        break
            elif slot_name == "性别":
                for suffix in self.gender_suffixs:
                    if query[end:].startswith(suffix):
                        end += len(suffix)
                        slot_value += suffix
                        break

            elif slot_name in ["购车预算", "意向出手价"]:
                raw_slot_value = slot_value
                for kw in self.price_kws:
                    if query[end:].startswith(kw):
                        end += len(kw)
                        slot_value += kw
                if slot_name == "购车预算":
                    if len(re.findall("[伍两一二三四五六七八九十0-9万]", slot_value)) == 0:
                        continue
                    for kw in self.price_rm_suffixs:
                        if slot_value.endswith(kw):
                            end -= len(kw)
                    for kw in self.price_rm_prefixs:
                        if slot_value.startswith(kw):
                            start += len(kw)
                    slot_value = query[start: end]
                    if len(slot_value) == 0 or slot_value in ["日产轩逸14代悦享版", "0首付", "0", "3成", "万元"] :
                        continue
                    # print(f"slot_value=={slot_value}, query=={query}, raw_slot_value=={raw_slot_value}")
                    budget.append(slot_value)
            elif not nested and slot_name == "意向车型" and slot_name in slot_all_map:
                # slot = "意向车型"
                if "意向车系" in slot_all_map:
                    # <意向车系>奥迪a6</意向车系><意向车型>2023豪华版</意向车型>
                    if slot_all_map["意向车系"][1] == start:
                        start = slot_all_map["意向车系"][0]
                        slot_value = query[start: end]

                    # <意向车型>2023豪华版</意向车型><意向车系>奥迪a6</意向车系>
                    elif slot_all_map["意向车系"][0] == end:
                        end = slot_all_map["意向车系"][1]
                        slot_value = query[start: end]
                    # 嵌套
                    elif slot_all_map["意向车系"][0] >= start and slot_all_map["意向车系"][1] <= end:
                        del_list.append(slot_all_map["意向车系"])
                    del slot_all_map["意向车系"]
                    
                if "意向品牌" in slot_all_map:
                    # <意向品牌>奥迪</意向品牌><意向车型>2023豪华版</意向车型>
                    if slot_all_map["意向品牌"][1] == start:
                        start = slot_all_map["意向品牌"][0]
                        slot_value = query[start: end]
                    # <意向车型>2023豪华版</意向车型><意向品牌>奥迪</意向品牌>
                    elif slot_all_map["意向品牌"][0] == end:
                        end = slot_all_map["意向品牌"][1]
                        slot_value = query[start: end]
                    elif slot_all_map["意向品牌"][0] >= start and slot_all_map["意向品牌"][1] <= end:
                        del_list.append(slot_all_map["意向品牌"])
                    del slot_all_map["意向品牌"]
                        
                del slot_all_map["意向车型"]
            elif not nested and slot_name == "意向车系" and slot_name in slot_all_map:
                if "意向品牌" in slot_all_map:
                    # <意向品牌>奥迪</意向品牌><意向车系>2023豪华版</意向车系>
                    if slot_all_map["意向品牌"][1] == start:
                        start = slot_all_map["意向品牌"][0]
                        slot_value = query[start: end]
                    # <意向车系>2023豪华版</意向车系><意向品牌>奥迪</意向品牌>
                    elif slot_all_map["意向品牌"][0] == end:
                        end = slot_all_map["意向品牌"][1]
                        slot_value = query[start: end]
                    elif slot_all_map["意向品牌"][0] >= start and slot_all_map["意向品牌"][1] <= end:
                        del_list.append(slot_all_map["意向品牌"])
                    del slot_all_map["意向品牌"]

                if "意向车型" in slot_all_map:
                    # <意向车系>奥迪a6</意向车系><意向车型>2023豪华版</意向车型>
                    if slot_all_map["意向车型"][0] == end:
                        end = slot_all_map["意向车型"][1]
                        slot_value = query[start: end]

                    # <意向车型>2023豪华版</意向车型><意向车系>奥迪a6</意向车系>
                    elif slot_all_map["意向车型"][1] == start:
                        start = slot_all_map["意向车型"][0]
                        slot_value = query[start: end]
                    # 嵌套
                    elif slot_all_map["意向车型"][0] <= start and slot_all_map["意向车型"][1] >= end:
                        slot_name = "意向车型"
                        start = slot_all_map["意向车型"][0]
                        end = slot_all_map["意向车型"][1]
                        slot_value = query[start: end]
                        del_list.append(slot_all_map["意向车系"])
                    del slot_all_map["意向车型"]

                del slot_all_map["意向车系"]
                    
            elif not nested and slot_name == "意向品牌" and slot_name in slot_all_map:
                if "意向车系" in slot_all_map:
                    # <意向品牌>奥迪</意向品牌><意向车系>2023豪华版</意向车系>
                    if slot_all_map["意向车系"][0] == end:
                        slot_name = "意向车系"
                        end = slot_all_map["意向车系"][1]
                        slot_value = query[start: end]
                    # <意向车系>2023豪华版</意向车系><意向品牌>奥迪</意向品牌>
                    elif slot_all_map["意向车系"][1] == start:
                        slot_name = "意向车系"
                        start = slot_all_map["意向车系"][0]
                        slot_value = query[start: end]
                    elif start >= slot_all_map["意向车系"][0] and end <= slot_all_map["意向车系"][1]:
                        slot_name = "意向车系"
                        start = slot_all_map["意向车系"][0]
                        end = slot_all_map["意向车系"][1]
                        slot_value = query[start: end]
                    del slot_all_map["意向车系"]

                if "意向车型" in slot_all_map:
                    # <意向品牌>奥迪</意向品牌><意向车型>2023豪华版</意向车型>
                    if slot_all_map["意向车型"][0] == end:
                        slot_name = "意向车型"
                        end = slot_all_map["意向车型"][1]
                        slot_value = query[start: end]
                    # <意向车型>2023豪华版</意向车型><意向品牌>奥迪</意向品牌>
                    elif slot_all_map["意向车型"][1] == start:
                        slot_name = "意向车型"
                        start = slot_all_map["意向车型"][0]
                        slot_value = query[start: end]
                    elif start >= slot_all_map["意向车型"][0] and end <= slot_all_map["意向车型"][1]:
                        slot_name = "意向车型"
                        start = slot_all_map["意向车型"][0]
                        end = slot_all_map["意向车型"][1]
                        slot_value = query[start: end]
                    del slot_all_map["意向车型"]
                del slot_all_map["意向品牌"]
            elif slot_name == "意向大类":
                if slot_value in self.filter_list:
                    continue
                chaos.append(slot_value)
            elif slot_name == "意向车型":
                if len(slot_value) < 3 and slot_value == query:
                    # print(f"filter 车型: {slot_value}")
                    garbage.append(slot_value)
                    continue
                # if len(re.findall("(自动|手动|新能源|油电|混动|电动|电车|油车|摩托车|柴油|汽油|增程|日系|德系|韩系|法系|运损车|试驾|合资|油电混|三轮车|营运车|宽体车|混合动力|抵押车|合资|国产|新车库存|大一点的配套设施齐全的|下线车|残疾人开的|带车棚|营转非|冷藏车|大型的车|瓦罐|清障)", slot_value)) != 0 and "suv" not in slot_value.lower():
                #     print(f"自动or手动: {slot_info} in query {query}")
                    # continue
                # if len(re.findall("[一二三四五六七八九1-9][座坐做]", slot_value)) != 0:
                #     # print(f"n座: {slot_info} in query {query}")
                #     slot_name = "座位数"
                #     if slot_value.lower().startswith("suv"):
                #         start += 3
                #         slot_value = query[start: end]
                #         # print(f"slot_value start: {slot_value}")
                #     if slot_value.lower().endswith('s∪v'):
                #         end -= 3
                #         slot_value = query[start: end]
                #         # print(f"slot_value end: {slot_value}")
                #     n_seat.append(slot_value)
                    # continue
            # 我买上海的车，能上北京的牌子吗
            elif slot_name == "购车地点" and "牌" in query:
                if re.search("(%s.{0,3}牌|牌.{0,3}%s)" % (slot_value, slot_value), query) is not None:
                    continue
            elif slot_name == "付款方式":
                for kw in self.pay_way_kws:
                    if kw in slot_value and slot_value != kw and "不" not in slot_value:
                        start += slot_value.index(kw)
                        end = start + len(kw)
                        slot_value = query[start: end]
                        break
                if slot_value not in self.pay_way_kws and len(slot_value) > 4:
                    continue
                pay_way.append(slot_value)

            if slot_name not in self.slot_labels:
                continue
            adjusted_slots.append({
                slot_name: {
                    slot_value: [start, end]
                }
            })

        return adjusted_slots


class HomeIntentAdjuster(IntentAdjuster):

    def __init__(self, ind_intent_categories, sample_ratios=None):
        super().__init__(ind_intent_categories, sample_ratios)

        self.phone_rule = re.compile('电话|打|手机|短信|信息')
        self.wechat_rule = re.compile('微信|加|v|V|vx')
        self.invalid_rule = re.compile('抖|邮箱|q|Q')
        self.role_sets = ["1", "2"]
        self.sample_ratios = sample_ratios
        self.ind_intent_categories = ind_intent_categories

    def adjust(self, intent_set, query, **kwargs):
        """
        1. 行业意图标签修正
        """
        intent_list = []
        if len(intent_set) > 0:
            if self.full_num_rule.match(query) is not None:
                return intent_list
            for _intent in intent_set:
                if _intent == '要求客服联系':
                    if re.search(self.wechat_rule, query) and not re.search(self.invalid_rule, query):
                        intent_list.append('要求客服加微信')
                        continue
                if _intent == '问联系方法':
                    if re.search(self.wechat_rule, query):
                        intent_list.append('问微信')
                        continue
                    if re.search(self.phone_rule, query):
                        intent_list.append('问电话')
                        continue
                if _intent == '问机构联系方式':
                    if re.search(self.wechat_rule, query):
                        intent_list.append('问微信')
                        continue
                    if re.search(self.phone_rule, query):
                        intent_list.append('问电话')
                        continue

                if _intent != "其他":
                    intent_list.append(_intent)

            intent_list.remove('商务合作')
            # 合并业务范围和参与条件
            if '问参与条件' in intent_list:
                intent_list.remove('问参与条件')
                if '问业务范围' not in intent_list:
                    intent_list.append('问业务范围')

        return intent_list


class HomeSlotAdjuster(SlotAdjuster):

    def __init__(self, slot_categories, sample_ratios=None):
        super().__init__(slot_categories, sample_ratios)
        self.sample_ratios = sample_ratios
        self.slot_categories = slot_categories
        # self.phone_kws = ["手机", "电话"]
        # self.wechat_kws = ["微信", "vx", "+v", "威信", "v号"]
        # self.gender_suffixs = ['士', '生', '孩儿', '孩']
        # self.price_kws = ['左右', '块钱', '块', '元', '多', '以上', '以下']

        # self.time_add_prefixs = ['等', '过']
        # self.time_add_suffixs = ['以后', '之前', '前', '后']
        # self.time_rm_prefixs = []
        # self.time_rm_suffixs = ['左右', '再说', '联系', '吧', '儿', '打']

    def adjust(self, contexts, query, slot_infos, **kwargs):
        slot_list = []
        slot_categorie_dict = {}
        for slot_categorie in self.slot_categories:
            slot_categorie_dict[slot_categorie] = 1.0
            slot_sample_rate = slot_categorie_dict
            slot_sample_rate['无'] = 1.0
        is_phone = False
        is_wechat = False
        slot_all_list = []
        for _ in slot_infos:
            slot_result = json.loads(_)
            if slot_result["slot"] == "电话号":
                is_phone = True
            if slot_result["slot"] == "微信号":
                is_wechat = True
            slot_all_list.append(slot_result["slot"])

        is_wechat_in_cur_content = False  # 最终是不是微信号
        if is_phone and is_wechat:
            mem_row_content_daoxu = contexts[::-1]  # 倒叙
            is_phone_in_mem_content = False
            is_wechat_in_mem_content = False
            for mem_content in mem_row_content_daoxu:
                if "手机" in mem_content and is_wechat_in_mem_content == False:
                    is_phone_in_mem_content = True
                    break
                if ("微信" in mem_content or "好友" in mem_content) and is_phone_in_mem_content == False:
                    is_wechat_in_mem_content = True
                    break
            if is_wechat_in_mem_content == True:  # 只有满足微信号逻辑才添加微信，其余都是电话号
                is_wechat_in_cur_content = True
        # print(sess_tag[row_id]["slots"])
        for slot_info in slot_infos:
            slot_result = json.loads(slot_info)
            start, end, slot, slot_value = slot_result.values()
            # 槽位的修正规则
            if is_phone and is_wechat and slot in ["微信号", "电话号"]:  # 微信号、电话号重合后的修正规则
                if is_wechat_in_cur_content and slot == "微信号":
                    slot_list.append((start, end, slot))
                    # print(mem_row_content)
                    # print("当前用户话术【{}】标注槽位【微信号、手机号】槽值【{}】修正为【{}】".format(row_content, slot_value, "微信号"))
                elif is_wechat_in_cur_content == False and slot == "电话号":
                    slot_list.append((start, end, slot))
                    # print(mem_row_content)
                    # print("当前用户话术【{}】标注槽位【微信号、手机号】槽值【{}】修正为【{}】".format(row_content, slot_value, "电话号"))
            elif slot == "具体需求":
                if re.search(r"装|修|翻|买|定", slot_value):
                    slot_list.append((start, end, slot))
            else:
                slot_list.append((start, end, slot))

        adjusted_slots = []
        for i in slot_list:
            adjusted_slots.append({
                i[2]: {
                    query[i[0]:i[1]]: [i[0], i[1]]
                }
            })
        # '1736350744741902': '回访时间': {'明天白天': [0, 4]}
        return adjusted_slots


adjuster = {
    "ind_car_slot": [CarIntentAdjuster, CarSlotAdjuster],
    "ind_car_nlu": [CarIntentAdjuster, CarSlotAdjuster],
    "ind_home_nlu": [HomeIntentAdjuster, HomeSlotAdjuster],
    "default": [IntentAdjuster, SlotAdjuster]
}


def get_distribution(data, data_type=None):
    intent_list = []
    slot_list = []
    """
    sample = {
        "context": {"text": _contexts, "roles": _roles},
        "query": query,
        "label": {"intent": intent_list, "slot": slot_list}
    }
    """
    for line in data:
        intents = line["label"]["intent"]
        slots = line["label"]["slot"]
        if len(intents) == 0:
            intents = ["无"]
        intent_list.extend(intents)
        if len(slots) == 0:
            slot_list.append("无")
            continue
        for slot_info in slots:
            slot_list.extend(list(slot_info.keys()))
    intent_counter = Counter(intent_list)
    slot_counter = Counter(slot_list)
    info = f"Intent and Slot Distribution of {data_type} set:"
    info += "\nIntent"
    for k, v in intent_counter.most_common():
        info += f"\n{k}\t{v}"
    info += "\n\nSlot"
    for k, v in slot_counter.most_common():
        info += f"\n{k}\t{v}"
    logger.info(info)
    

def text_clean(query):
    query = re.sub("\s", "$", query).replace('&nbsp;', '$').lower()
    query = [ch if str.isprintable(ch) else "$" for ch in query]

    return "".join(query)

def merge_continue(context_list, role_list):
    context = context_list
    role = role_list
    if len(role) == 0:
        return [], []
    pre_src, pre_content = role[0], context[0]
    merged_src_list, merged_content_list = [pre_src], [pre_content]
    for src, content in zip(role[1:], context[1:]):
        if pre_src == src:
            merge_content = f"{merged_content_list[-1]}, {content}"
            merged_content_list[-1] = merge_content
        else:
            merged_content_list.append(content)
            merged_src_list.append(src)
        pre_src = src

    return merged_content_list, merged_src_list



def process_data(task_name, data_path, do_sampling=False, target_src="1", debug=False, seed=42, test_data=False, relabeled_data=None, nested_slot=False):
    # 加载源数据
    if not os.path.exists(os.path.join(data_path, task_name)):
        os.makedirs(os.path.join(data_path, task_name))
    load_path = os.path.join(data_path, task_name, f"{task_name}.pkl")
    dump_path = os.path.join(data_path, task_name)

    data_list = load_pickle(load_path)

    task_adjusters = adjuster.get(task_name, "default")

    if debug:
        data_list = data_list[:100]

    label_config = get_label_config(task_name)
    ind_intent_categories = label_config["ind_intent_categories"]
    slot_categories = label_config["slot_categories"]
    slot_sample_ratios = label_config["slot_sample_ratio"]
    intent_sample_ratios = label_config["intent_sample_ratio"]

    result = []
    if len(task_adjusters) < 2:
        logger.info(f"Task adjusters shouldn't none, please implement.")
        return []
    intent_adjuster = task_adjusters[0](ind_intent_categories, intent_sample_ratios)
    slot_adjuster = task_adjusters[1](slot_categories, slot_sample_ratios)

    # 意图槽位标签校准，生成格式化样本
    for data in tqdm(data_list):
        sess_info, sess_tag = data[0], data[1]
        contexts = []
        roles = []
        for row_id in range(1, sess_info["rounds"] + 1):
            query = sess_info[row_id]["row_content"]
            query = text_clean(query)
            src = sess_info[row_id]["src_code"]

            if src == target_src:
                if row_id not in sess_tag:
                    continue
                intent_list = intent_adjuster.adjust(sess_tag[row_id]["industry_intent"], query)
                slot_list = slot_adjuster.adjust(contexts, query, sess_tag[row_id]["slots"], nested=nested_slot)
                _contexts = contexts[:]
                _roles = roles[:]
                _intent_list = intent_list[:]
                _slot_list = slot_list[:]

                sample = {
                    "context": {"text": _contexts, "roles": _roles},
                    "query": query,
                    "label": {"intent": _intent_list, "slot": _slot_list}
                }
                result.append(sample)
                intent_list = []
                slot_list = []

            contexts.append(query)
            role = 0 if src == target_src else 1
            roles.append(role)
    if relabeled_data is not None:
        # pd.read_csv(os.join(data_path, relabeled_data), engine="python")
        # hdfs dfs -get hdfs:///home/byte_ad_va/user/zhaoguangpu/nlu/data/wild/slot_relabeled_v2.csv /
        # hdfs dfs -get hdfs:///home/byte_ad_va/user/zhaoguangpu/nlu/data/wild/intent_relabeled_v2.csv /
        # hdfs dfs -get hdfs:///home/byte_ad_va/user/zhaoguangpu/nlu/data/wild/intent_result_v3.csv /
        intent_data1 = pd.read_csv(os.path.join(data_path, task_name, "intent_relabeled_v2.csv"), engine="python")
        intent_data2 = pd.read_csv(os.path.join(data_path, task_name, "intent_result_v3.csv"), engine="python")
        slot_data = pd.read_csv(os.path.join(data_path, task_name, "slot_relabeled_v2.csv"), engine="python")
        slot_data_0530 = pd.read_csv(os.path.join(data_path, task_name, "relabeled_slot_0606.csv"), engine="python")

        intent_relabeled = {}
        slot_relabeled = {}
        for _, row in tqdm(intent_data1.iterrows()):
            if type(row["重标注"]) == float:
                continue
            context = row["历史话术"]
            query = row["当前话术"]
            key = f"{context}-{query}"
            relabeled_intent = eval(row["重标注"])
            relabeled_intent = [intent.replace("-", "_") for intent in relabeled_intent if intent != "咨询购车意见"]
            intent_relabeled[key] = relabeled_intent
        for _, row in tqdm(intent_data2.iterrows()):
            if type(row["重标注"]) == float:
                continue
            context = row["历史话术"]
            query = row["当前话术"]
            key = f"{context}-{query}"
            relabeled_intent = eval(row["重标注"])
            relabeled_intent = [intent.replace("-", "_") for intent in relabeled_intent if intent != "咨询购车意见"]
            intent_relabeled[key] = relabeled_intent

        for _, row in tqdm(slot_data.iterrows()):
            if type(row["重标注"]) == float:
                continue
            context = row["历史话术"]
            query = row["当前话术"]
            key = f"{context}-{query}"
            
            slot_set = eval(row["重标注"])
            new_slot_set = set()
            if len(slot_set) > 0:
                for slot in slot_set:
                    slot = list(slot)
                    slot[2] += 1            
                    new_slot_set.add(tuple(slot))

            slot_relabeled[key] = new_slot_set
        for _, row in tqdm(slot_data_0530.iterrows()):
            if type(row["relabeled_entities"]) == float or str(row["relabeled_entities"]) == "1":
                continue
            context_ = row["context"].split("[USR]") if type(row["context"]) != float else []

            context = []
            for c in context_:
                if len(c) == 0:
                    continue
                c = c.split("[SYS]")
                c = [i for i in c if len(c) > 0]
                context.extend(c)
            query = row["query"]
            key = f"{context}-{query}"
            slot_set = eval(row["relabeled_entities"])
            new_slot_set = set()
            if len(slot_set) > 0:
                for slot in slot_set:
                    # slot = list(slot)
                    name, start, end = slot
                    value = query[start: end]
                    _slot = [value, start, end, name]
                    new_slot_set.add(tuple(_slot))
            slot_relabeled[key] = new_slot_set
        for _, row in tqdm(slot_data_0530.iterrows()):
            if type(row["relabeled_intent"]) == float:
                continue
            context_ = row["context"].split("[USR]") if type(row["context"]) != float else []

            context = []
            for c in context_:
                if len(c) == 0:
                    continue
                c = c.split("[SYS]")
                c = [i for i in c if len(c) > 0]
                context.extend(c)
            query = row["query"]
            key = f"{context}-{query}"
            intent_set = eval(row["relabeled_intent"])
            new_slot_set = set()
            if len(intent_set) > 0:
                intent_relabeled[key] = relabeled_intent

        print(f"len(intent_relabeled): {len(intent_relabeled)}")
        print(f"len(slot_relabeled): {len(slot_relabeled)}")
        data = {
            "intent": intent_relabeled,
            "slot": slot_relabeled,
        }

        find_intent = 0
        find_slot = 0
        for idx, sample in enumerate(result):
            context = sample["context"]["text"]
            query = sample["query"]
            key =f"{context}-{query}".replace("$", "")
            intents = {k.replace("$", ""): v for k, v in data["intent"].items()}

            idx = 0

            if key in intents:
                intent_list = [intent for intent in intents[key] if intent in ind_intent_categories]
                sample["label"]["intent"] = intent_list
                result[idx] = sample
                find_intent += 1
            if key in data["slot"]:
                slot_set = data["slot"][key]
                final_slot_list = []
                for slot in slot_set:
                    if slot[-1] not in slot_categories:
                        continue
                    final_slot_list.append({
                        slot[-1]: {
                            slot[0]: [slot[1], slot[2]]
                        }
                    })
                sample["label"]["slot"] = final_slot_list
                # print(f"final_slot_list: {final_slot_list}")
                result[idx] = sample
                find_slot += 1
        print(f"find_intent: {find_intent}, find_slot: {find_slot}")        

    # 采样
    if do_sampling:
        result = intent_adjuster.sampling(result)
        result = slot_adjuster.sampling(result)

    if debug:
        for idx in range(3):
            logger.info(f"debug data: {json.dumps(result[idx], ensure_ascii=False, indent=2)}")
        save_json(result, os.path.join(dump_path, "all.json"), pretty=True)

    # hack
    all_queries = []
    for sample in result:
        contexts = sample["context"]["text"]
        all_queries.extend(contexts)
    counter = Counter(all_queries)
    filter_list = [k for k, v in counter.most_common(1000)]
    
    _result = []
    for sample in result:
        contexts = sample["context"]["text"]
        roles = sample["context"]["roles"]
        # contexts, roles = merge_continue(contexts, roles)
        _contexts = []
        _roles = []
        for ctx, role in zip(contexts, roles):
            if ctx in filter_list and random.random() < 0.1:
                _contexts.append(ctx)
                _roles.append(role)
            else:
                _contexts.append(ctx)
                _roles.append(role)
        sample["context"]["text"] = _contexts
        sample["context"]["roles"] = _roles
        _result.append(sample)
    result = _result
    
    
    get_distribution(result, data_type="All")
    # for i in range(5):
    #     print(f"result[{i}]: {json.dumps(result[i], ensure_ascii=False, indent=2)}")
    if not test_data:
        # 划分数据集，存储结果
        train_set, dev_set = train_test_split(result, test_size=0.3, random_state=seed)
        # debug mode，用训练集做验证集，快速验证
        adapte_data = []
        for sample in train_set:
            if "问微信" in sample["label"]["intent"] and random.random() <= 0.5:
                adapte_data.append(sample)
        # train_set += adapte_data
        logger.info(f"train data size: {len(train_set)}")
        get_distribution(train_set, data_type="Train")
        logger.info(f"dev data size: {len(dev_set)}")
        get_distribution(dev_set, data_type="Dev")

        save_json(train_set, os.path.join(dump_path, "train.json"), pretty=False)
        save_json(dev_set, os.path.join(dump_path, "dev.json"), pretty=True)
        save_json(result, os.path.join(dump_path, "all.json"), pretty=False)
    else:
        save_json(result, os.path.join(dump_path, "test.json"), pretty=True)
    logger.info("DONE!")
    # logger.info(f"budget: {budget}")
    # logger.info(f"pay_way: {pay_way}") 
    # print(f'n_seat size: {len(n_seat)}')
    # print(f'del_list size: {len(del_list)}')
    # print(f"n_seat: {n_seat}")
    # print(f"chaos size: : {len(chaos)}")
    # print(f"chaos: {chaos}")
    # print(f"garbage: {garbage}")
    # print(f"问库存： {kucun}")
    print(f"non_label: {len(non_label)}")
    # print(f"chexing size: {len(chexing)}")
    # print(f"chexing: {chexing}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="ind_car_nlu", type=str, required=False,
                        help="The name of the task")
    parser.add_argument("--data_dir", default="./datasets/", type=str,
                        help="The input data dir.")
    parser.add_argument("--target_src", default="1", type=str,
                        help="target src, usually 1 is custormer")
    parser.add_argument("--seed", default=3407, type=int,
                        help="target src, usually 1 is custormer")
    parser.add_argument("--do_sampling", action="store_true",
                        help="Whether to upsampling and downsampling data.")
    parser.add_argument("--debug", default="False", type=str,
                        help="Whether run in debug mode.")
    parser.add_argument("--nested", default="True", type=str,
                        help="Whether slot is nested")
    parser.add_argument("--test", default="False", type=str,
                        help="Whether data is test set")
    args = parser.parse_args()
    
    args.debug = args.debug.lower() == "true"
    args.nested = args.nested.lower() == "true"
    args.test = args.test.lower() == "true"

    
    random.seed(args.seed)

    # label_config_ = get_label_config(task_name_)

    process_data(args.task_name, args.data_dir, do_sampling=args.do_sampling, target_src=args.target_src, debug=args.debug, seed=args.seed, test_data=args.test, relabeled_data="yes", nested_slot=args.nested)
    # process_data(args.task_name, args.data_dir, do_sampling=args.do_sampling, target_src=args.target_src, debug=args.debug, seed=args.seed)
