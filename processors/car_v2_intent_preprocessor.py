# -*- coding: utf-8 -*-
# Author : zhaoguangpu@bytedance.com
# Created: 2022/12/06
import json
import os
import re
import sys
import random
from collections import Counter


from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("..")

from tools.common import load_json, save_json
from configs.label_config import get_label_config
from tools.common import init_logger, logger
import os



init_logger(log_file=None)

# categores = ['问微信', '问门店地址', '问具体车系库存', '求介绍', '问车价_落地价/裸车价', '问是否有现车', '请求二手车估价', '问分期贷款', '问提车方式', '问售后', '问电话', '问置换补贴', '要求客服打电话', '拒绝某种联系方式', '问购车优惠', '问车况_车龄', '问有哪些车系', '微信电话同号', '咨询配置_颜色', '要求客服加微信', '问汽车落户', '问所在城市经销商', '问门店/销售联系方式']
categores = ["其他","问新车价格","提供电话号","给出地址","询问有哪些车或指定车","提供微信号","问优惠","问门店位置","问首付","要价格表","问可否分期","希望微信沟通","买车","请求被加微信","问配置情况","问月供","要库存表","以旧换新","请求加对方微信","微信电话同号","请求图片","问是否有门店","询问联系方式","问微信","问电话","怎样预约到店","希望不要打电话","问分期方案","问分期利息","问指定车收车价","问分期期数","请求联系","问能否上指定地区牌照","求推荐车","问年款","是否有现车","问指定车的二手价","问二手车车况","已加过微信","问分期全款","给定时间到店","表明指定时间不方便电话","问以旧换新补贴优惠","卖车","希望电话沟通","问置换流程和政策","已留过电话","问二手车里程","问提车地址","问续航","问排量","不希望留联系方式","请求链接","问变速箱类型","是否卖新车","问最低首付","问二手车车龄","问座位数","问油耗","问颜色","问收车价格","问可否过户","问质保/保养政策","改装","问发动机","卖新车还是二手车","问提车条件","问质保期限","问能否上牌","可否低于指定价格","问是否有过事故","问指定车置换指定车","希望联系销售","问改装价格","是否卖二手车","问上哪里牌照","是否收给定条件的车","问可置换哪些车","问是否为指定厂商国别","多久有现车","问是否可以到店","问可否异地上牌","问代办贷款","问指定地区车可否过户","问最低价格是否为指定价格","希望不要加微信","问过户流程","问二手车补漆","问车零部件价格","求推荐其他车","问是否为抵账车","问分期费用明细","希望到车后联系","表示信号不好","问指定车是否有指定驱动类型","问售后联系方式","问外地户是否可以","问怎么注销车辆","问是否承担过户","问有哪些价位的车","是否有车零部件","问二手车过户","给定时间到店试驾","询问指定时间到店是否可以","问指定车是否为指定年款","问二手车新旧","问哪些车有置换补贴","问是否需要留电话","问指定车是否为指定颜色","问指定首付的分期方案","问补漆费用","问以旧换新退车是否需要退还置换补贴","是否提供上门收车","微信添加频繁"]
keep_categores = ['其他', '问新车价格', '提供电话号', '给出地址', '询问有哪些车或指定车', '提供微信号', '问优惠', '问门店位置', '问首付', '要价格表', '问可否分期', '希望微信沟通', '买车', '请求被加微信', '问配置情况', '问月供', '要库存表', '以旧换新', '请求加对方微信', '微信电话同号', '请求图片', '问是否有门店', '询问联系方式', '问微信', '问电话', '怎样预约到店', '希望不要打电话', '问分期方案', '问分期利息', '问指定车收车价', '问分期期数', '请求联系', '问能否上指定地区牌照', '求推荐车', '问年款', '是否有现车', '问指定车的二手价', '问二手车车况', '已加过微信', '问分期全款', '给定时间到店', '表明指定时间不方便电话', '问以旧换新补贴优惠', '卖车', '希望电话沟通']
sample_ratios = {k: 1.0 for k in categores}

sample_ratios["其他"] = 0.25
sample_ratios["问新车价格"] = 0.25
sample_ratios["提供电话号"] = 0.25
sample_ratios["给出地址"] = 0.25
sample_ratios["询问有哪些车或指定车"] = 0.7
sample_ratios["提供微信号"] = 0.5
sample_ratios["问优惠"] = 0.65
sample_ratios["问门店位置"] = 0.65
sample_ratios["问首付"] = 0.8
sample_ratios["要价格表"] = 0.85
sample_ratios["问可否分期"] = 0.9


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
    print(info)

def sampling(data):
    if sample_ratios is None:
        return data
    data_sampled = []
    for idx, sample in enumerate(data):
        ratio = random.random()
        intents = []
        drop = False
        label_intent = sample["label"]["intent"]
        label_intent = [intent if intent in keep_categores else "其他" for intent in label_intent]
        for intent in label_intent: 
            if ratio > sample_ratios[intent] and len(label_intent) == 1:
                drop = True
                break

        if not drop:
            sample["label"]["intent"] = label_intent
            data_sampled.append(sample)

    return data_sampled

def text_clean(query):
    query = re.sub("\s", "$", query).replace('&nbsp;', '$').lower()
    query = [ch if str.isprintable(ch) else "$" for ch in query]

    return "".join(query)


def get_intent(x):
    x = json.loads(x)
    mainForm = x["mainForm"]
    # print(mainForm)
    label = [mf["value"] for mf in mainForm if mf["name"] == "car_intention"]
    # print(label)
    return label
    
def parse_obj(x, raw_data_map):
    x = json.loads(x)
    # print(f'x["id"]: {x["id"]}===={type(x["id"])}')
    if x["id"] not in raw_data_map:
        print(f"not in map: ====={x}")
        return None
    sample = raw_data_map[x["id"]]
    # print(f"sample: {sample}")
    return sample


def process_data(task_name, data_path, do_sampling=False, debug=False, seed=42, test_set=False):
    # 加载源数据
    if not os.path.exists(os.path.join(data_path, task_name)):
        os.makedirs(os.path.join(data_path, task_name))
    all_data_path = os.path.join(data_path, task_name, "intent_all_session_raw.json")
    load_path = os.path.join(data_path, task_name, "tcs_data0724.csv")
    dump_path = os.path.join(data_path, task_name)
    
    all_raw_data = load_json(all_data_path)
    raw_data_map = {}
    for idx, sample in enumerate(all_raw_data):
        unique_id = sample["id"]
        raw_data_map[unique_id] = sample

    data = pd.read_csv(load_path)
    data["intent"] = data["verify_data"].apply(get_intent)
    data["sample"] = data["object_data"].apply(parse_obj, args=(raw_data_map, ))
    sample_list = data["sample"].tolist()
    label_list = data["intent"].tolist()
    all_samples_list = []
    for sample, label in zip(sample_list, label_list):
        # 阿拉伯语过滤
        if re.findall(r'[\u0600-\u06FF]+',sample["query"]):
            continue
        if re.findall(r'[\u0600-\u06FF]+',sample["pre_dialog_text"]):
            continue
        label = [i if i in categores else "其他" for i in label]
        sample["label"] = label
        all_samples_list.append(sample)
    print(json.dumps(all_samples_list[:1], ensure_ascii=False, indent=2))

    result = []
    for idx, sample in enumerate(all_samples_list):

        context, roles = [], []
        pre_dialog_text = sample["pre_dialog_text"]
        pre_dialog_text = re.sub(r'(\n(用户:)|\n(客服:))', "###\g<1>", pre_dialog_text)
        pre_dialog_text = pre_dialog_text.replace("\n", "")
        if len(pre_dialog_text) > 0:
            pre_dialog_list = pre_dialog_text.split("###")
            for dialog in pre_dialog_list:
                dialog_token = dialog.split(":")
                role = 1 if dialog_token[0] == "客服" else 0
                if len(dialog_token) < 2:
                    continue
                text = dialog_token[1].strip()
                context.append(text)
                roles.append(role)
        
        result.append(
            {
                "context": {
                    "text": context,
                    "roles": roles
                },
                "query": sample["query"],
                "label": {
                    "intent": sample["label"],
                    "slot": []
                }
            }
        )

    # save_json(format_sample_list, dump_path, pretty=True)
    print(f"format_sample_list size: {len(result)}")
    get_distribution(result)
    if do_sampling:
        result = sampling(result)
        print(f"sampled_data size: {len(result)}")
    if not test_set:
        train_set, dev_set = train_test_split(result, test_size=0.3, random_state=1024)
        print(f"train_set size: {len(train_set)}")
        print(f"dev_set size: {len(dev_set)}")
        get_distribution(result)
        save_json(train_set, os.path.join(dump_path, "all.json"), pretty=False)
        save_json(train_set, os.path.join(dump_path, "train.json"), pretty=False)
        save_json(dev_set, os.path.join(dump_path, "dev.json"), pretty=True)
    else:
        save_json(result, os.path.join(dump_path, "test.json"), pretty=True)

    logger.info("DONE!")






if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="ind_car_v2_nlu", type=str, required=False,
                        help="The name of the task")
    parser.add_argument("--data_dir", default="./datasets/", type=str,
                        help="The input data dir.")
    parser.add_argument("--seed", default=3407, type=int,
                        help="target src, usually 1 is custormer")
    parser.add_argument("--do_sampling", action="store_true",
                        help="Whether to upsampling and downsampling data.")
    parser.add_argument("--debug", default="False", type=str,
                        help="Whether run in debug mode.")
    args = parser.parse_args()
    
    args.debug = args.debug.lower() == "true"

    
    random.seed(args.seed)


    process_data(args.task_name, args.data_dir, do_sampling=True, debug=args.debug, seed=args.seed, test_set=False)

