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
from configs.label_config import ind_home_nlu

csv.field_size_limit(500 * 1024 * 1024)
service_row_list = []
customer_row_list = []

ind_intent_categories = ind_home_nlu["ind_intent_categories"]
slot_categories = ind_home_nlu["slot_categories"]

def filter_emoji(desstr, restr='，'):
    # 过滤表情
    try:
        co = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    return co.sub(restr, desstr)


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

    # obj = json.loads(object_str)
    obj = json.loads(str(json.dumps(eval(object_str))))
    sess_info["sess_id"] = obj["session_talk_id"]
    sess_info["industry_id"] = obj["industry_id"]
    sess_info["rounds"] = len(obj["session_talk"])

    for row in obj["session_talk"]:
        row_id = row["raw_id"]

        # # 商服会话有混入教育会话，通过关键词匹配去除这些会话
        # for kw in ['小学', '本科', '大专', '高中', '职称', '考试科目', '院校专业']:
        #     if kw in row["raw"]:
        #         return {}
        # 将表情替换成，
        raw_row = row["raw"]
        raw_row = filter_emoji(raw_row)
        if raw_row != row["raw"]:
            print("原始 ", row["raw"])
            print("现在 ", raw_row)
            print("")
        row["raw"] = raw_row
        sess_info[row_id] = {
            "row_content": text_process(row["raw"]),
            "src_code": row["src_code"]
        }

        if row["src_code"] == "1":
            # TODO 存在一些空字符暂未处理
            if re.findall(r"‪| |\t", row["raw"]) or '️' in row["raw"]:
                print("话术中存在空、不可见字符， 对应话术为【%s】，已替换为【，】" % row["raw"])

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
    # verify = json.dumps(eval(verify_str))
    verify = json.loads(str(json.dumps(eval(verify_str))))
    for _main_form in verify["mainForm"]:
        # slot 标注梳理
        if _main_form["name"] == "markeds":
            row_id = int(_main_form["value"]['rawId'])
            if row_id not in sess_tags:
                sess_tags[row_id] = defaultdict(set)

            _slot = ""
            if type(_main_form["value"]["label"]) == list:  # 'label': ['性别']
                if len(_main_form["value"]["label"]) >= 1:
                    _slot = str(_main_form["value"]["label"][0].strip())
            else:
                _slot = str(_main_form["value"]["label"].strip())
            if _slot == "":
                print("slot没有标签 ", _main_form)
                continue
            # print(sess_info)
            slot_value = _main_form["value"]["text"]

            slot_value = text_process(slot_value)
            row_content = sess_info[row_id]["row_content"]
            start = _main_form["value"]["start"]
            end = _main_form["value"]["end"]

            # 有些 start end 跟 slot value 对不上，采用子串查询作为兜底
            assert slot_value in row_content
            # assert slot_value == slot_value.strip(), print(slot_value)

            if len(find_all(slot_value, row_content)) == 1:
                # 只有一个子串时使用查找索引的结果，用正则 find_iter 查找有坑，+ 会识别为转义符
                start = row_content.index(slot_value)
                end = start + len(slot_value)
            else:
                # 有多个子串时使用标注结果，并打印错误的标注
                if slot_value != row_content[start: end]:
                    print("存在暂无法自动修正的 slot 标注", slot_value, row_content[start: end], row_content)
                    continue

            slot_dict = {
                "start": start,
                "end": end,
                "slot": _slot,
                "slot_value": slot_value
            }

            if _slot in slot_categories:
                sess_tags[row_id]["slots"].add(json.dumps(slot_dict, ensure_ascii=False))

        # 意图 标注梳理
        # 如果标注的 行业意图 ，则根据原始定义的 行业意图 和后期 人工整理的行业意图 和 原始意图 的映射关系 来进行映射
        elif _main_form["name"].startswith("business_intention_") or _main_form["name"].startswith(
                "other_business_intention_"):
            row_id = int(_main_form['name'].split('_')[-1])
            if row_id not in sess_tags:
                sess_tags[row_id] = defaultdict(set)

            _intent = _main_form["value"]
            _intent = _intent.strip()

            if _intent in ind_intent_categories:
                sess_tags[row_id]["industry_intent"].add(_intent)

            else:
                # 选择部分高频其他意图的加入
                if '拒绝某种联系' in _intent:
                    sess_tags[row_id]["industry_intent"].add('拒绝某种联系途径')
                if '问资料' in _intent or '求资料' in _intent:
                    sess_tags[row_id]["industry_intent"].add('问资料')
                if '求加速' in _intent:
                    sess_tags[row_id]["industry_intent"].add('求加速')
                if '问收益' in _intent:
                    sess_tags[row_id]["industry_intent"].add('问收益')

        else:
            pass

    return sess_tags


data = pd.read_csv("./datasets/ind_home_nlu/all_home_data_20220816.csv", engine="python")

all_examples = 0
valid_examples = 0
other_examples = 0
sess = []
for _, row in tqdm(data.iterrows()):
    # print(">object_data>>>>>> ",_, row["object_data"])
    # print(">verify_data>>>>>> ",_, row["verify_data"])
    _sess_id = row[0]
    _sess_info = object_parser(row[2])
    if _sess_info == {} or '非行业' in str(row["verify_data"]):
        other_examples += 1
    else:
        _sess_tag = verify_parser(row[3], _sess_id, _sess_info)
        valid_examples += len(_sess_tag)
        sess.append((_sess_info, _sess_tag))
print("非家具行业的回话量共", other_examples)
print()
print("总会话数量：", len(sess))
print()
print("总标注用户对话行数：", valid_examples)
print()

slot_list = []
ind_intent_list = []


# 修正函数
def read_data_correction(path_file):
    # 生成标准槽位和意图
    pd_reader = pd.read_csv(path_file).fillna("")
    person_slot_dict = {}
    person_intent_dict = {}
    data_list = pd_reader.values.tolist()
    for row in data_list:
        if len(row) == 10:
            # print(len(row),row)
            # print(">>>>>>>>>>>")
            correct_intent = row[5].strip()
            raw_intent = row[4].strip()
            correct_slot = row[8].strip()
            if correct_intent != "" or correct_slot != "":
                # print(">>>>>>>>>>> ", correct_intent)
                # print("<<<<<<<<<<< ", correct_slot)

                session_talk_id = row[0]
                cur_id = row[1]
                cur_text = row[2]

                if correct_intent != "":  # 存在意图更新
                    # print(">>>>>>>>>>> ", correct_intent)
                    try:
                        if type(correct_intent) == str and '[' not in correct_intent:
                            if correct_intent in ind_intent_categories:
                                person_intent_dict[str(session_talk_id) + ">>" + str(cur_id)] = [correct_intent]
                        else:
                            # print(type(correct_intent))
                            # print(eval(correct_intent))
                            raw_intent_list = eval(raw_intent)
                            correct_intent_list = []
                            for _ in eval(correct_intent):
                                if _ in ind_intent_categories:
                                    correct_intent_list.append(_)
                            # print("11111111112222222222 >>>>>>>>> ", print_correct_intent, type(print_correct_intent))
                            if "要求微信沟通" in raw_intent_list and "要求微信沟通" not in correct_intent_list:
                                continue
                            person_intent_dict[str(session_talk_id) + ">>" + str(cur_id)] = correct_intent_list
                    except:
                        pass

                if correct_slot != "":  # 存在槽位更新
                    # print("<<<<<<<<<<< ", correct_slot)
                    try:
                        correct_slot_list = []
                        for _ in eval(correct_slot):
                            if _[2] in slot_categories:
                                correct_slot_list.append(
                                    json.dumps({
                                        "start": _[0],
                                        "end": _[1],
                                        "slot": _[2],
                                        "slot_value": cur_text[_[0]:_[1] + 1],
                                    }, ensure_ascii=False)
                                )
                        if len(correct_slot_list) > 0:
                            person_slot_dict[str(session_talk_id) + ">>" + str(cur_id)] = correct_slot_list
                    except:
                        pass
        else:
            pass
    print(person_slot_dict["1719119529802759>>4"])
    return person_intent_dict, person_slot_dict


# 修正标注之后的文件
path_file = "./datasets/ind_home_nlu/all_home_data_20220816_by_txy_final.csv"
person_intent_dict, person_slot_dict = read_data_correction(path_file)

for _sess_info, _sess_tag in sess:

    sess_id = _sess_info["sess_id"]

    for row_id in _sess_tag:

        recorrection_key = str(sess_id) + ">>" + str(row_id)

        # 修正意图
        if person_intent_dict.get(recorrection_key):
            _sess_tag[row_id]["industry_intent"] = person_intent_dict[recorrection_key]
            print("recorrect key is {} , value is {}".format(recorrection_key, person_intent_dict[recorrection_key]))

        if person_slot_dict.get(recorrection_key):
            _sess_tag[row_id]["slots"] = person_slot_dict[recorrection_key]

        for key in _sess_tag[row_id]:
            if key in ["industry_intent"]:
                for _ in _sess_tag[row_id][key]:
                    ind_intent_list.append(_)
            elif key in ["slots"]:
                for _ in _sess_tag[row_id][key]:
                    slot_list.append(json.loads(_)['slot'])
    # print(">>>>",_sess_info)
    # print("2222",_sess_tag)

print('行业意图频次')
for i, j in Counter(ind_intent_list).most_common():
    print(i + "\t" + str(j))

print('\n槽位频次')
for i, j in Counter(slot_list).most_common():
    print(i + "\t" + str(j))

with open("./datasets/ind_home_nlu/ind_home_nlu.pkl", "wb") as writer:
    pickle.dump(sess, writer)

with open("./datasets/ind_home_nlu/ind_home_nlu.pkl", "rb") as writer:
    sess1 = pickle.load(writer)

    for _1, _2 in sess1[:10]:
        print(_2)


