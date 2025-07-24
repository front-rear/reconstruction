#!/bin/sh

cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

TRAIN_CODE=train.py

CONFIG_DIR="None"
WEIGHT="None"
RESUME=false
GPU=None
OID="None"
LABEL="None"


while getopts "p:c:n:w:g:r:o:l:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    c)
      CONFIG_DIR=$OPTARG
      ;;
    n)
      EXP_DIR=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=$OPTARG
      ;;
    g)
      GPU=$OPTARG
      ;;
    o)
      OID=$OPTARG
      ;;
    l)
      LABEL=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

echo "Python interpreter dir: $PYTHON"
echo "Config: $CONFIG"
echo "GPU Num: $GPU"

MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code


echo " =========> CREATE EXP DIR <========="
mkdir -p "$MODEL_DIR" "$CODE_DIR"
cp -r scripts launch pointcept "$CODE_DIR"


echo "Loading config in:" $CONFIG_DIR
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="

$PYTHON "$CODE_DIR"/launch/$TRAIN_CODE \
--config-file "$CONFIG_DIR" \
--num-gpus "$GPU" \
--options save_path="$EXP_DIR" oid="$OID" label="$LABEL"
