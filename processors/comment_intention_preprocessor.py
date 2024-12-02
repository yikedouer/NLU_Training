# -*- coding: utf-8 -*-
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

from tools.common import load_pickle, save_pickle, save_json, load_json
from configs.label_config import get_label_config
from tools.common import init_logger, logger
import os



init_logger(log_file=None)



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
        if slots is None or len(slots) == 0:
            continue
        for slot_info in slots:
            slot_list.extend(list(slot_info.keys()))
    intent_counter = Counter(intent_list)
    info = f"Intent and Slot Distribution of {data_type} set:"
    info += "\nIntent"
    for k, v in intent_counter.most_common():
        info += f"\n{k}\t{v}"

    logger.info(info)
    

def text_clean(query):
    query = re.sub("\s", "$", query).replace('&nbsp;', '$').lower()
    query = [ch if str.isprintable(ch) else "$" for ch in query]

    return "".join(query)



def process_data(task_name, data_path, do_sampling=False, debug=False, seed=42, test_set=False, use_ind_token=False, use_supply_data=False):
    # 加载源数据
    if not os.path.exists(os.path.join(data_path, task_name)):
        os.makedirs(os.path.join(data_path, task_name))
    load_path = os.path.join(data_path, task_name, "comment_intention_5600_relabeled.csv")
    supply_load_path = os.path.join(data_path, task_name, "comment_intention_supply.csv")
    # load_path = os.path.join(data_path, task_name, "comment_intention_0608_clean.csv")
    # load_path = os.path.join(data_path, task_name, "xingtu.csv")
    
    dump_path = os.path.join(data_path, task_name)
    data = pd.read_csv(load_path)


    supply_data = pd.read_csv(supply_load_path)
    supply_content_list = supply_data["content"].tolist()
    supply_content_list = [i.replace("\n", " ") for i in supply_content_list if len(i) > 0]
    supply_comment_id_list = supply_data["uid"].tolist()
    supply_ind_list = supply_data["first_industry_name"].tolist()
    supply_label_list = supply_data["标签"].tolist()
    print(f"supply_data size: {len(supply_data)}")
    print(f"supply_content_list size: {len(supply_content_list)}")
    print(f"supply_comment_id_list size: {len(supply_comment_id_list)}")
    print(f"supply_ind_list size: {len(supply_ind_list)}")
    print(f"supply_label_list size: {len(supply_label_list)}")


    content_list = data["content"].tolist()
    comment_id_list = data["comment_id"].tolist()
    ind_list = data["first_industry_name"].tolist()
    content_list = [i.replace("\n", " ") for i in content_list if len(i) > 0]

    if not test_set:
        label_list = data["人工标注结果"].tolist()
        label_list = ["非高意向" if int(label) == 2 else "高意向" for label in label_list]
    else:
        label_list = data["意向"].tolist()
    print(f"len(label_list) raw: {len(label_list)}")
    if use_supply_data:
        content_list += supply_content_list
        comment_id_list += supply_comment_id_list
        ind_list += supply_ind_list
        label_list += supply_label_list
    print(f"len(label_list) now: {len(label_list)}")

        # label_list = ["非高意向"] * len(content_list)
    """
    sample = {
        "context": {"text": _contexts, "roles": _roles},
        "query": query,
        "label": {"intent": _intent_list, "slot": _slot_list}
    }
    """
    result = []
    print(f"len(content_list): {len(content_list)}")
    print(f"len(label_list): {len(label_list)}")
    for content, label, ind in zip(content_list, label_list, ind_list):

        if use_ind_token:
            content = f"[{ind}]{content}"
        sample = {
            "context": None,
            "query": content,
            "label": {"intent": [label], "slot": []}
        }
        result.append(sample)
    print(f"result: {len(result)}")
    print(f"result[0]: {result[0]}")

    if debug:
        result = result[:100]
    

    # 采样
    if do_sampling:
        pass

    if debug:
        for idx in range(3):
            logger.info(f"debug data: {json.dumps(result[idx], ensure_ascii=False, indent=2)}")
        save_json(result, os.path.join(dump_path, "all.json"), pretty=True)
    
    get_distribution(result, data_type="All")
    if not test_set:
        train_set, dev_set = train_test_split(result, test_size=0.3, random_state=seed)

        logger.info(f"train data size: {len(train_set)}")
        get_distribution(train_set, data_type="Train")
        logger.info(f"dev data size: {len(dev_set)}")
        get_distribution(dev_set, data_type="Dev")

        save_json(train_set, os.path.join(dump_path, "train.json"), pretty=False)
        save_json(dev_set, os.path.join(dump_path, "dev.json"), pretty=True)
        save_json(result, os.path.join(dump_path, "all.json"), pretty=False)
    else:
        save_json(result, os.path.join(dump_path, "test.json"), pretty=False)
    logger.info("DONE!")

def process_data_live(task_name, data_path, do_sampling=False, debug=False, seed=42, test_set=False, use_ind_token=False, use_supply_data=False):
    # 加载源数据
    if not os.path.exists(os.path.join(data_path, task_name)):
        os.makedirs(os.path.join(data_path, task_name))
    load_path = os.path.join(data_path, task_name, "live_comment.csv")

    
    dump_path = os.path.join(data_path, task_name)
    data = pd.read_csv(load_path)



    content_list = data["content"].tolist()
    content_list = [i.replace("\n", " ") for i in content_list if len(i) > 0]

    if not test_set:
        label_list = data["人工标注结果"].tolist()
        label_list = ["非高意向" if int(label) == 2 else "高意向" for label in label_list]
    else:
        label_list = data["意向"].tolist()
    print(f"len(label_list) raw: {len(label_list)}")

    print(f"len(label_list) now: {len(label_list)}")

        # label_list = ["非高意向"] * len(content_list)
    """
    sample = {
        "context": {"text": _contexts, "roles": _roles},
        "query": query,
        "label": {"intent": _intent_list, "slot": _slot_list}
    }
    """
    result = []
    print(f"len(content_list): {len(content_list)}")
    print(f"len(label_list): {len(label_list)}")
    ind_list = ["none"]*len(label_list)
    for content, label, ind in zip(content_list, label_list, ind_list):

        if use_ind_token:
            content = f"[{ind}]{content}"
        sample = {
            "context": None,
            "query": content,
            "label": {"intent": [label], "slot": []}
        }
        result.append(sample)
    print(f"result: {len(result)}")
    print(f"result[0]: {result[0]}")

    if debug:
        result = result[:100]
    

    # 采样
    if do_sampling:
        pass

    if debug:
        for idx in range(3):
            logger.info(f"debug data: {json.dumps(result[idx], ensure_ascii=False, indent=2)}")
        save_json(result, os.path.join(dump_path, "all.json"), pretty=True)
    
    get_distribution(result, data_type="All")
    if not test_set:
        train_set, dev_set = train_test_split(result, test_size=0.3, random_state=seed)

        logger.info(f"train data size: {len(train_set)}")
        get_distribution(train_set, data_type="Train")
        logger.info(f"dev data size: {len(dev_set)}")
        get_distribution(dev_set, data_type="Dev")

        save_json(train_set, os.path.join(dump_path, "train.json"), pretty=False)
        save_json(dev_set, os.path.join(dump_path, "dev.json"), pretty=True)
        save_json(result, os.path.join(dump_path, "all.json"), pretty=False)
    else:
        save_json(result, os.path.join(dump_path, "test.json"), pretty=False)
    logger.info("DONE!")


def process_data_all(task_name, data_path, do_sampling=False, debug=False, seed=42, test_set=False, use_ind_token=False, use_supply_data=False):
    # 加载源数据
    if not os.path.exists(os.path.join(data_path, task_name)):
        os.makedirs(os.path.join(data_path, task_name))
    short_video_path = os.path.join(data_path, task_name, "short_video.csv")
    short_video_supply_path = os.path.join(data_path, task_name, "short_video_supply.csv")
    live_path = os.path.join(data_path, task_name, "live_comment.csv")
    
    dump_path = os.path.join(data_path, task_name)
    short_video_data = pd.read_csv(short_video_path)
    live_data = pd.read_csv(live_path)

    if use_supply_data:
        supply_data = pd.read_csv(short_video_supply_path)
        supply_content_list = supply_data["content"].tolist()
        supply_content_list = [i.replace("\n", " ") for i in supply_content_list if len(i) > 0]
        supply_comment_id_list = supply_data["uid"].tolist()
        supply_ind_list = supply_data["first_industry_name"].tolist()
        supply_label_list = supply_data["标签"].tolist()
        print(f"supply_data size: {len(supply_data)}")
        print(f"supply_content_list size: {len(supply_content_list)}")
        print(f"supply_comment_id_list size: {len(supply_comment_id_list)}")
        print(f"supply_ind_list size: {len(supply_ind_list)}")
        print(f"supply_label_list size: {len(supply_label_list)}")


    content_list = data["content"].tolist()
    ind_list = [0] * len(content_list)
    content_list = [i.replace("\n", " ") for i in content_list if len(i) > 0]

    if not test_set:
        label_list = data["人工标注结果"].tolist()
        label_list = ["非高意向" if int(label) == 2 else "高意向" for label in label_list]
    else:
        label_list = data["意向"].tolist()
    print(f"len(label_list) raw: {len(label_list)}")
    if use_supply_data:
        content_list += supply_content_list
        # comment_id_list += supply_comment_id_list
        ind_list += supply_ind_list
        label_list += supply_label_list
    print(f"len(label_list) now: {len(label_list)}")

        # label_list = ["非高意向"] * len(content_list)
    """
    sample = {
        "context": {"text": _contexts, "roles": _roles},
        "query": query,
        "label": {"intent": _intent_list, "slot": _slot_list}
    }
    """
    result = []
    print(f"len(content_list): {len(content_list)}")
    print(f"len(label_list): {len(label_list)}")
    for content, label, ind in zip(content_list, label_list, ind_list):

        if use_ind_token:
            content = f"[{ind}]{content}"
        sample = {
            "context": None,
            "query": content,
            "label": {"intent": [label], "slot": []}
        }
        result.append(sample)
    print(f"result: {len(result)}")
    print(f"result[0]: {result[0]}")

    if debug:
        result = result[:100]
    

    # 采样
    if do_sampling:
        pass

    if debug:
        for idx in range(3):
            logger.info(f"debug data: {json.dumps(result[idx], ensure_ascii=False, indent=2)}")
        save_json(result, os.path.join(dump_path, "all.json"), pretty=True)
    
    get_distribution(result, data_type="All")
    if not test_set:
        train_set, dev_set = train_test_split(result, test_size=0.3, random_state=seed)

        logger.info(f"train data size: {len(train_set)}")
        get_distribution(train_set, data_type="Train")
        logger.info(f"dev data size: {len(dev_set)}")
        get_distribution(dev_set, data_type="Dev")

        save_json(train_set, os.path.join(dump_path, "train.json"), pretty=False)
        save_json(dev_set, os.path.join(dump_path, "dev.json"), pretty=True)
        save_json(result, os.path.join(dump_path, "all.json"), pretty=False)
    else:
        save_json(result, os.path.join(dump_path, "test.json"), pretty=False)
    logger.info("DONE!")

def convert(task_name, data_path):
    # 加载源数据
    if not os.path.exists(os.path.join(data_path, task_name)):
        os.makedirs(os.path.join(data_path, task_name))

    short_video_path = os.path.join(data_path, task_name, "comment_intention_5600_relabeled.csv")
    short_video_supply_path = os.path.join(data_path, task_name, "comment_intention_supply.csv")
    short_video_data = pd.read_csv(short_video_path)
    short_video_supply_data = pd.read_csv(short_video_supply_path)
    columns = short_video_data.columns.values
    keep_columns = ['content', '人工标注结果']
    rm_columns = [c for c in columns if c not in keep_columns]
    # print(f"rm_columns: {rm_columns}")
    short_video_data = short_video_data.drop(columns=rm_columns)
    # print(f"finally: {short_video_data.columns.values}")
    short_video_data.to_csv(os.path.join(data_path, task_name, "short_video.csv"), index=False)
    
    columns = short_video_supply_data.columns.values
    keep_columns = ['content', '标签']
    rm_columns = [c for c in columns if c not in keep_columns]
    print(f"rm_columns: {rm_columns}")
    short_video_supply_data = short_video_supply_data.drop(columns=rm_columns)
    print(f"finally: {short_video_supply_data.columns.values}")
    short_video_supply_data.to_csv(os.path.join(data_path, task_name, "short_video_supply.csv"), index=False)
    live_comment_path = os.path.join(data_path, task_name, "live_comment.csv")
    live_comment_data = pd.read_csv(live_comment_path)
    print(f"live: {live_comment_data.columns.values}")
    columns = live_comment_data.columns.values
    keep_columns = ['content', '人工标注结果']
    rm_columns = [c for c in columns if c not in keep_columns]
    print(f"rm_columns: {rm_columns}")
    live_comment_data = live_comment_data.drop(columns=rm_columns)
    print(f"finally: {live_comment_data.columns.values}")
    live_comment_data.to_csv(os.path.join(data_path, task_name, "live_comment2.csv"), index=False)

def merge(task_name, data_path):
    if not os.path.exists(os.path.join(data_path, task_name)):
        os.makedirs(os.path.join(data_path, task_name))
    dump_path = os.path.join(data_path, task_name)
    live_all = load_json(os.path.join(data_path, task_name, "live_all.json"))
    live_dev = load_json(os.path.join(data_path, task_name, "live_dev.json"))
    live_train = load_json(os.path.join(data_path, task_name, "live_train.json"))
    
    short_video_all = load_json(os.path.join(data_path, task_name, "short_video_all.json"))
    short_video_dev = load_json(os.path.join(data_path, task_name, "short_video_dev.json"))
    short_video_train = load_json(os.path.join(data_path, task_name, "short_video_train.json"))
    train_set = live_train + short_video_train
    dev_set = live_dev + short_video_dev

    save_json(train_set, os.path.join(dump_path, "train.json"), pretty=False)
    save_json(dev_set, os.path.join(dump_path, "dev.json"), pretty=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="comment_intention", type=str, required=False,
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


    # process_data_live(args.task_name, args.data_dir, do_sampling=args.do_sampling, debug=args.debug, seed=args.seed, test_set=False, use_ind_token=False, use_supply_data=False)
    # process_data(args.task_name, args.data_dir, do_sampling=args.do_sampling, debug=args.debug, seed=args.seed, test_set=False, use_ind_token=False, use_supply_data=False)
    # convert(args.task_name, args.data_dir)
    merge(args.task_name, args.data_dir)