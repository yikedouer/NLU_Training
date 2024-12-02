# -*- coding: utf-8 -*-

import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
import pickle
import json
import numpy as np
import re
import time
import csv
import math
import os
import re
import json
import traceback


csv.field_size_limit(500 * 1024 * 1024)
service_row_list = []
customer_row_list = []
slot_set = set()
all_slots = []
intent_set = set()
other_slots = []
other_intents = []

slot_norm_map = {
    "电话": "电话",
    "邮箱": "邮箱",
    "微信": "微信",
    "QQ": "QQ",
    "联系途径": "联系途径",
    "称呼（姓/姓名）": "用户称呼",
    "称谓（性别）": "性别",
    "年龄": "年龄",
    "所在地点": "购车地点",
    "个人征信": "个人征信",
    "购车预算": "购车预算",
    "计划购车时间": "购车时间",
    "意向大类": "意向大类",
    "意向品牌": "意向品牌",
    "意向车系": "意向车系",
    "意向车型": "意向车型",
    "意向颜色": "意向颜色",
    "购车目的": "购车目的",
    "付款方式（全款/分期）": "付款方式",
    "车辆持有情况": "车辆持有情况",
    "试驾情况": "试驾情况",
    "是否接受跨地域提车": "是否接受跨地域提车",
    "二手车车况（车龄）": "二手车车况_车龄",
    "二手车车况（里程）": "二手车车况_里程",
    "二手车车况（事故记录）": "二手车车况_事故记录",
    "二手车车况（车型）": "二手车车况_车型",
    "二手车车况（其他）": "二手车车况_其他",
    "意向出手价": "意向出手价",
    "配件/改装意向": "配件/改装意向"
}
ind_intent_norm_map = {
    "问电话": "问电话",
    "问微信": "问微信",
    "问门店/销售联系方式":  "问门店/销售联系方式",
    "问门店地址":  "问门店地址",
    "微信电话同号":  "微信电话同号",
    "微信电话不同号":  "微信电话不同号",
    "要求客服加微信":  "要求客服加微信",
    "要求客服打电话":  "要求客服打电话",
    "拒绝某种联系方式":  "拒绝某种联系方式",
    "索求产品介绍": "求介绍",
    "咨询购车意见":  "咨询购车意见",
    "问所在城市经销商":  "问所在城市经销商",
    "问有哪些车系":  "问有哪些车系",
    "问是否有现车":  "问是否有现车",
    "咨询配置（车型）":  "咨询配置_车型",
    "咨询配置（动力）":  "咨询配置_动力",
    "咨询配置（颜色）":  "咨询配置_颜色",
    "咨询配置（其他）":  "咨询配置_其他",
    "咨询口碑（动力）":  "咨询口碑_动力",
    "咨询口碑（性价比）":  "咨询口碑_性价比",
    "咨询口碑（舒适性）":  "咨询口碑_舒适性",
    "咨询口碑（操控）":  "咨询口碑_操控",
    "咨询口碑（空间）":  "咨询口碑_空间",
    "咨询口碑（内饰）":  "咨询口碑_内饰",
    "咨询口碑（外观）":  "咨询口碑_外观",
    "咨询口碑（口碑）":  "咨询口碑_口碑",
    "咨询口碑（其他）":  "咨询口碑_其他",
    "咨询用车成本（保险）":  "咨询用车成本_保险",
    "咨询用车成本（保养）":  "咨询用车成本_保养",
    "咨询用车成本（油耗）":  "咨询用车成本_油耗",
    "咨询用车成本（其他）":  "咨询用车成本_其他",
    "问车价（落地价/裸车价）":  "问车价_落地价/裸车价",
    "问汽车落户":  "问汽车落户",
    "问购车优惠":  "问购车优惠",
    "问分期贷款":  "问分期贷款",
    "问置换补贴":  "问置换补贴",
    "咨询预约试驾":  "咨询预约试驾",
    "请求二手车估价": "请求二手车估价",
    "询问二手车市价":  "问二手车市价",
    "咨询二手车选购经验":  "咨询二手车选购经验",
    "询问具体车系库存":  "问具体车系库存",
    "询问提车方式":  "问提车方式",
    "问车况（车龄）":  "问车况_车龄",
    "问车况（里程）":  "问车况_里程",
    "问车况（事故记录）":  "问车况_事故记录",
    "问车况（其他）":  "问车况_其他",
    "问改装": "问改装",
    "问维修":  "问维修",
    "问加装":  "问加装",
    "问汽车保养知识":  "问汽车保养知识",
    "问售后":  "问售后",
    "问征信":  "问征信",
    "商务合作/营销推广": "商务合作/营销推广"
}

ind_intent_categories = [
    "问电话","问微信","问门店/销售联系方式","问门店地址","微信电话同号","微信电话不同号","要求客服加微信","要求客服打电话","拒绝某种联系方式",
    "索求产品介绍","咨询购车意见","问所在城市经销商","问有哪些车系","问是否有现车","咨询配置（车型）","咨询配置（动力）","咨询配置（颜色）","咨询配置（其他）","咨询口碑（动力）","咨询口碑（性价比）","咨询口碑（舒适性）","咨询口碑（操控）","咨询口碑（空间）","咨询口碑（内饰）","咨询口碑（外观）","咨询口碑（口碑）","咨询口碑（其他）","咨询用车成本（保险）","咨询用车成本（保养）","咨询用车成本（油耗）","咨询用车成本（其他）","问车价（落地价/裸车价）","问汽车落户","问购车优惠","问分期贷款","问置换补贴","咨询预约试驾",
    "请求二手车估价","询问二手车市价","咨询二手车选购经验","询问具体车系库存","询问提车方式","问车况（车龄）","问车况（里程）","问车况（事故记录）","问车况（其他）",
    "问改装","问维修","问加装","问汽车保养知识","问售后","问征信","商务合作/营销推广"
]


slot_categories = [
    "电话","邮箱","微信","QQ","联系途径",
    "称呼（姓/姓名）","称谓（性别）","年龄","所在地点","个人征信",
    "购车预算","计划购车时间","意向大类","意向品牌","意向车系","意向车型","意向颜色","购车目的","付款方式（全款/分期）","车辆持有情况","试驾情况","是否接受跨地域提车","二手车车况（车龄）","二手车车况（里程）","二手车车况（事故记录）","二手车车况（车型）","二手车车况（其他）","意向出手价",
    "配件/改装意向"
]

other_ind_intent_categories = []

other_slot_categories = []


def clean_func():
    """用户回答序号且被识别为 slot 时，替换为具体选项值"""
    return


def text_process(text):
    # TODO 暂时简单将空字符替换为 - ，保证槽位前后标能对上
    text = re.sub(r"‪| |\t|\n", "，", text)
    text = text.replace('️', "，")  # 奇怪的符号
    text = text.strip()

    return text


def object_parser(object_str):
    sess_info = {}

    obj = json.loads(object_str)
    sess_info["sess_id"] = obj["session_talk_id"]
    sess_info["industry_id"] = obj["industry_id"]
    sess_info["rounds"] = len(obj["session_talk"])

    for row in obj["session_talk"]:
        row_id = row["raw_id"]

        # 商服会话有混入教育会话，通过关键词匹配去除这些会话
        for kw in ['小学', '本科', '大专', '高中', '职称', '考试科目', '院校专业']:
            if kw in row["raw"]:
                return {}

        sess_info[row_id] = {
            "row_content": text_process(row["raw"]),
            "src_code": row["src_code"]
        }

        if row["src_code"] == "1":
            # TODO 存在一些空字符暂未处理
            if re.findall(r"‪| |\t", row["raw"]) or '️' in row["raw"]:
                pass
#                 print("话术中存在空、不可见字符， 对应话术为【%s】，已替换为【，】" % row["raw"])

        if row["src_code"] == "1":
            customer_row_list.append(text_process(row["raw"]))
        else:
            service_row_list.append(text_process(row["raw"]))

    return sess_info


def verify_parser(verify_str, sess_id, sess_info):
    """
    """

    def find_all(sub, s):
        index_list = []
        index = s.find(sub)
        while index != -1:
            index_list.append(index)
            index = s.find(sub, index + 1)

        return index_list

    sess_tags = {}
    verify = json.loads(verify_str)

    for _main_form in verify["mainForm"]:
        # slot 标注梳理
        if _main_form["name"] == "markeds":
            row_id = int(_main_form["value"]['rawId'])
            if row_id not in sess_tags:
                sess_tags[row_id] = {"slots": [], "industry_intent": set()}
            _slot = _main_form["value"]["label"]
            _slot = "" if len(_slot) == 0 else _slot[0]
            _slot = str(_slot.strip())

            slot_value = _main_form["value"]["text"]
            slot_value = text_process(slot_value)
            row_content = sess_info[row_id]["row_content"]
            start = _main_form["value"]["start"]
            end = _main_form["value"]["end"]

            # 有些 start end 跟 slot value 对不上，采用子串查询作为兜底
            assert slot_value in row_content


            if len(find_all(slot_value, row_content)) == 1:
                # 只有一个子串时使用查找索引的结果，用正则 find_iter 查找有坑，+ 会识别为转义符
                start = row_content.index(slot_value)
                end = start + len(slot_value)
            else:
                # 有多个子串时使用标注结果，并打印错误的标注
                if slot_value != row_content[start: end]:
                    print("存在暂无法自动修正的 slot 标注", slot_value, row_content[start: end], row_content, end)
#                     end -= 1
                    continue
            if slot_value != row_content[start: end]:
                print(f"slot_value: {slot_value};, row_content[start: end]: {row_content[start: end]};")
                print(json.dumps(_main_form["value"], ensure_ascii=False, indent=2))
            if "探岳两驱三周年有没有购置税" in row_content:
                print(f"slot_value: {slot_value};, row_content[start: end]: {row_content[start: end]};")
                print(json.dumps(_main_form["value"], ensure_ascii=False, indent=2))

            if _slot not in slot_norm_map:
                other_slots.append(_slot)
            else:
                slot_set.add(_slot)

                slot_dict = {
                    "start": start,
                    "end": end,
                    "slot": slot_norm_map[_slot],
                    "slot_value": slot_value
                }
                sess_tags[row_id]["slots"].append(json.dumps(slot_dict, ensure_ascii=False))

        # 意图 标注梳理
        elif _main_form["name"] == "intention":
            try:
                value = _main_form["value"]
                row_id = int(value["raw_id"])
                if row_id not in sess_tags:
                    sess_tags[row_id] = {"slots": [], "industry_intent": set()}
                if "car_intention" not in value["data"] and "other_car_intention" not in value["data"]:
                    continue
                other_car_intention = ""
                car_intention = []
                if "other_car_intention" in value["data"]:
                    other_car_intention = value["data"]["other_car_intention"]
                    other_intents.append(other_car_intention.strip())
                if "car_intention" in value["data"]:
                    _intent = [i.strip() for i in value["data"]["car_intention"]]
                    _intent = set(_intent)
                    if other_car_intention in _intent:
                        _intent.remove(other_car_intention)
                    for i in _intent:
                        if i not in ind_intent_categories:
                            other_intents.append(i)
                            print(f"weird intent: {i}, add into other_intents")
                        else:
                            for i in _intent:
                                sess_tags[row_id]["industry_intent"].add(ind_intent_norm_map[i])

                    intent_set.update(_intent)

                    
            except Exception as e:
                print(traceback.format_exc())
                print(f"Exception: \n{json.dumps(_main_form, ensure_ascii=False, indent=2)}")

        else:
            pass
    # idx = 0
    # for tag in sess_tags:
    #     print(tag)
    #     idx += 1
    #     if idx == 5:
    #         break
    # print(f"sess_tags: =======\n{sess_tags}")

    return sess_tags
def load_pickle_data(data_path, file_name):
    import joblib
    file_path = os.path.join(data_path, file_name)
    data =[]
    with open(file_path, "rb") as reader:
        data = pickle.load(reader)
    print(f"load data success.")
    print(f"data size of {file_name} is: {len(data)}")
    return data
# data_file = open("../datasets/ind_car_nlu/car.pkl")
data = load_pickle_data("../datasets/ind_car_nlu/", "car.pkl")
print(len(data),data[0], len(data[0]))

# all_examples = 0
# valid_examples = 0
# sess = []
# cnt = 0
# for _, row in tqdm(data.iterrows()):
#     cnt += 1
#     _sess_id = row[0]
#     _sess_info = object_parser(row[2])
# #     if cnt < 10:
# #         print(f"_sess_info:\n{_sess_info}")
# #         print(f"row[3]: \n{row[3]}")
#     if _sess_info == {}:
#         print('去除教育会话')
#     else:
#         _sess_tag = verify_parser(row[3], _sess_id, _sess_info)
#         valid_examples += len(_sess_tag)
#         sess.append((_sess_info, _sess_tag))
# #         if _sess_id == 50237055679:
# #             print(f"++++{len(sess)}==={type(sess)}")
# #             for i in sess:
# # #                 print(i)
# #                 print(f"+++++++++{json.dumps(i[0], ensure_ascii=False, indent=2)}")

# print()
# print("总会话数量：", len(sess))
# print()
# print("总标注用户对话行数：", valid_examples)
# print()
# print(f"intent set: \n{intent_set}====={len(intent_set)}")
# print(f"other_intents:\n{other_intents}====={len(other_intents)}")
# print(f"slot set: \n{slot_set}===={len(slot_set)}")
# print(f"other_slots:\n{other_slots}==={len(other_slots)}")


# for i in intent_set:
#     if i not in ind_intent_categories:
#         print(f"out of intent scope: {i}")
        
# for i in slot_set:
#     if i not in slot_categories:
#         print(f"out of slot scope: {i}")

# slot_list = []
# ind_intent_list = []

# for _sess_info, _sess_tag in sess:
#     for row_id in _sess_tag:
#         for key in _sess_tag[row_id]:
#             if key in ["industry_intent"]:
#                 for _ in _sess_tag[row_id][key]:
#                     ind_intent_list.append(_)
#             else:
#                 for _ in _sess_tag[row_id][key]:
#                     slot_list.append(json.loads(_)['slot'])

# print('行业意图频次')
# for i, j in Counter(ind_intent_list).most_common():
#     print(i, j)

# print('\n槽位频次')
# for i, j in Counter(slot_list).most_common():
#     print(i, j)

# with open("./datasets/car.pkl", "wb") as writer:
#     pickle.dump(sess, writer)

