#!/bin/bash

TASK_NAME="ind_car_nlu"
DEBUG="False"
NESTED="False"
TEST="False"

ARGS=`getopt --option dnht: --long debug,nested,help,task: -n $(basename $0) -- "$@"`
eval set -- "${ARGS}"

function USAGE() {
    echo "Usage:"
    echo "   --debug, -d: 是否为debug模式，使用该参数代表为True, 默认False"
    echo "   --nested, -n: slot是否包含嵌套，使用该参数代表为True, 默认False"
    echo "   --help, -h: usage"
    echo "   --task, -t: task_name, 默认ind_car_nlu"
}

while true
do
    case "$1" in
        -d|--debug)
            DEBUG="True"; shift 1 ;;
        -n|--nested)
            NESTED="True"; shift 1 ;;
        -h|--help)
            USAGE; exit 1 ;;
        -t|--task) TASK_NAME=$2; shift 2 ;;
        --) shift 1; break ;;
        *)
            echo get undefined option: $1
            USAGE; exit 1 ;;
    esac
done

echo Check params please:
echo Debug is ${DEBUG},  Nested slot is ${NESTED}, Task name is ${TASK_NAME}
if [ ! -d "./datasets/${TASK_NAME}" ]; then
  mkdir ./datasets/${TASK_NAME}
  hdfs dfs -get hdfs:///home/byte_ad_va/user/zhaoguangpu/nlu/data/processed/${TASK_NAME}.pkl ./datasets/${TASK_NAME}/${TASK_NAME}.pkl
fi

if [ "`ls -A ./datasets/${TASK_NAME}`" = "" ]; then
  hdfs dfs -get hdfs:///home/byte_ad_va/user/zhaoguangpu/nlu/data/processed/${TASK_NAME}.pkl ./datasets/${TASK_NAME}/${TASK_NAME}.pkl
fi

if [ "`ls -A ./pretrained_model`" = "" ]; then
  hdfs dfs -get hdfs:///home/byte_ad_va/user/zhaoguangpu/nlu/pretrained_model/bert-base-chinese ./pretrained_model/
fi

python3 processors/preprocessor.py --task_name ${TASK_NAME} --debug ${DEBUG} --nested ${NESTED} --test ${TEST}
