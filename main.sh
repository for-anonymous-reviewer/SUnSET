MODEL_PATH="/mnt/disk0/models"
LLM_NAME="qwen2_5-72b-instruct"
EMBEDDING_MODEL_NAME="gte-multilingual-base"
LLM_PATH="$MODEL_PATH/$LLM_NAME"
EMBEDDING_MODEL_PATH="$MODEL_PATH/$EMBEDDING_MODEL_NAME"
VLLM_API_BASE="http://localhost:6767/v1"

DATASET=t17
INPUT_PATH="/mnt/disk0/y84387018/sunset/data"
RAW_DATA_PATH=$INPUT_PATH
VECTOR_DB_PATH="/mnt/disk0/y84387018/sunset/db/$LLM_NAME"
PROCESSED_DATA_PATH="$INPUT_PATH/processed/$LLM_NAME"
SET_DATA_PATH="$INPUT_PATH/set/$LLM_NAME"
KG_PATH="$INPUT_PATH/dictionaryv3.pkl"
SCORING_PATH="$INPUT_PATH/relevancy_score"
# Parameters
stake=1
beta=0.9

# set extraction
python src/set_extraction_vllm.py \
    --model_path $LLM_PATH \
    --dataset $DATASET \
    --input_path $INPUT_PATH \
    --output_path $INPUT_PATH/set/$LLM_NAME/$DATASET \
    --openai_api_base $VLLM_API_BASE

# create milvus db
python src/create_milvus_db.py \
    --model_path $EMBEDDING_MODEL_PATH \
    --client_name $VECTOR_DB_PATH/$DATASET.db \
    --dataset $DATASET \
    --input_path $RAW_DATA_PATH \
    --set_path $SET_DATA_PATH

# Preprocessing
python src/preprocessing.py \
    --dataset $DATASET \
    --input_path $RAW_DATA_PATH \
    --set_path $SET_DATA_PATH \
    --output_path $PROCESSED_DATA_PATH

# Scoring

# SET matching
python src/set_matching.py \
    --model_path $EMBEDDING_MODEL_PATH \
    --vector_db_path $VECTOR_DB_PATH \
    --dataset $DATASET \
    --gt_path $RAW_DATA_PATH \
    --scoring_path $SCORING_PATH \
    --input_path $PROCESSED_DATA_PATH/timeline_outputs \
    --output_path $PROCESSED_DATA_PATH/timeline_outputs \
    --min_common_stake "$stake" \
    --beta "$beta"

# 4) Timeline generation & Evaluation
python src/cluster_tls_eval.py \
    --text_rank \
    --dataset $DATASET \
    --model_path $EMBEDDING_MODEL_PATH \
    --scoring_path $SCORING_PATH \
    --raw_data_path $RAW_DATA_PATH \
    --info_save_path $PROCESSED_DATA_PATH/info_outputs \
    --timelines_path $PROCESSED_DATA_PATH/timeline_outputs/$DATASET \
    --events_path $PROCESSED_DATA_PATH/event_outputs/$DATASET \
    --output_path $PROCESSED_DATA_PATH/result/$DATASET \
    --beta $beta