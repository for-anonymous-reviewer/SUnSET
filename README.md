# Who’s important? — SUnSET: Synergistic Understanding of Stakeholder, Events and Time for Timeline Generation
For ARR Blind Review

## Install
Installing necessary dependency for ROUGE score calculation.
For Ubuntu/Debian:
```
apt-get update
apt-get install libxml-parser-perl
```
For CentOS/RHEL/Fedora:
```
yum install perl-XML-Parser
```
Install Python environment via Conda:
```
conda create -n sunset_env python==3.10
conda activate sunset_env
pip install -r requirements.txt
```

## Dataset
To download the original dataset, please refer to [complementizer/news-tls](https://github.com/complementizer/news-tls).

All SET data of T17 and Crisis (after KG resolution) are [available here](https://drive.google.com/drive/folders/1b8eeetdjMNgUACJsVu0HQu9Bcj-4E19C?usp=sharing). 

## Steps
With the SET data provided, you can start from Step 1 (skip the SET extraction and KG resolution steps).
 
<details>
    <summary>If you want to start from scratch, unfold and run the following parts.</summary>
<!-- ### More -->

### -2. SET extraction
```
python src/set_extraction_vllm.py \
    --model_path $LLM_PATH \
    --dataset $DATASET \
    --input_path $INPUT_PATH \
    --output_path $INPUT_PATH/set/$LLM_NAME/$DATASET \
    --openai_api_base $VLLM_API_BASE
```

### -1. KG Construction & Resolution

</details>

### 1. Create vector database 
```
python src/create_milvus_db.py \
    --model_path $EMBEDDING_MODEL_PATH \
    --client_name $DATASET.db \
    --dataset $DATASET \
    --input_path $RAW_DATA_PATH \
    --set_path $SET_DATA_PATH
```

### 2. Run Preprocessing
```
python src/preprocessing.py \
    --dataset $DATASET \
    --input_path $RAW_DATA_PATH \
    --output_path $INPUT_DATA_PATH
```

### 3. Run SET Matching
```
python src/set_matching.py \
    --model_path $EMBEDDING_MODEL_PATH \
    --dataset $DATASET \
    --gt_path $RAW_DATA_PATH \
    --scoring_path $SCORING_PATH \
    --input_path $INPUT_DATA_PATH/timeline_outputs \
    --output_path $INPUT_DATA_PATH/timeline_outputs \
    --min_common_stake "$stake" \
    --beta "$beta"
```

### 4. Run Timeline Generation & Evaluation
```
python src/cluster_tls_eval.py \
    --text_rank \
    --dataset $DATASET \
    --model_path $EMBEDDING_MODEL_PATH \
    --scoring_path $SCORING_PATH \
    --raw_data_path $RAW_DATA_PATH \
    --info_save_path $INPUT_DATA_PATH/info_outputs \
    --timelines_path $INPUT_DATA_PATH/timeline_outputs/$DATASET \
    --events_path $INPUT_DATA_PATH/event_outputs/$DATASET \
    --output_path $INPUT_DATA_PATH/result/$DATASET \
    --beta $beta
```

