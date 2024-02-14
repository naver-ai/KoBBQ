# Evaluation on KoBBQ

## Prompt

## Pre-process
```
for PROMPT_ID in {1..5}
do
echo KoBBQ_test $PROMPT_ID
python3 1_preprocess.py \
    --samples-tsv-path ../data/KoBBQ_test_samples.tsv \
    --evaluation-tsv-path data/KoBBQ_test/KoBBQ_test_evaluation_$PROMPT_ID.tsv \
    --evaluation-json-path data/KoBBQ_test/KoBBQ_test_evaluation_$PROMPT_ID.json \
    --prompt-tsv-path 0_evaluation_prompts.tsv \
    --prompt-id $PROMPT_ID
done
```

## Model Inference
```
MODELS=$1
PROMPT_ID=$2

python3 2_model_inference.py \
    --data-path data/KoBBQ_test/KoBBQ_test_evaluation_$PROMPT_ID.json \
    --output-dir outputs/raw/KoBBQ_test_$PROMPT_ID \
    --model-names $MODELS
```

## Post-process
```
MODELS='gpt-3.5-turbo gpt-4 claude-instant-1.2 claude-2.0 clova-x KoAlpaca-Polyglot-12.8B'

for MODEL in $MODELS
do
    for PROMPT_ID in {1..5}
    do
    echo KoBBQ_test $PROMPT_ID $MODEL
    python3 3_postprocess_predictions.py \
        --predictions-tsv-path outputs/raw/KoBBQ_test_$PROMPT_ID/KoBBQ_test_evaluation_$PROMPT_ID\_$MODEL\_predictions.tsv \
        --preprocessed-tsv-path data/KoBBQ_test/KoBBQ_test_evaluation_$PROMPT_ID.tsv
    python3 4_predictions_to_evaluation.py \
        --predictions-tsv-path outputs/raw/KoBBQ_test_$PROMPT_ID/KoBBQ_test_evaluation_$PROMPT_ID\_$MODEL\_predictions.tsv \
        --preprocessed-tsv-path data/KoBBQ_test/KoBBQ_test_evaluation_$PROMPT_ID.tsv \
        --output-path outputs/processed/KoBBQ_test_$PROMPT_ID/KoBBQ_test_evaluation_$PROMPT_ID\_$MODEL.tsv
    done
done
```

## Evaluation
```
MODELS='gpt-3.5-turbo gpt-4 claude-instant-1.2 claude-2.0 clova-x KoAlpaca-Polyglot-12.8B'

for PROMPT_ID in {1..5}
do
python3 5_evaluation.py \
    --evaluation-result-path evaluation_result/KoBBQ_test_$PROMPT_ID.tsv \
    --model-result-tsv-dir outputs/processed/KoBBQ_test_$PROMPT_ID \
    --topic KoBBQ_test_evaluation \
    --test-or-all test \
    --prompt-tsv-path 0_evaluation_prompts.tsv \
    --prompt-id $PROMPT_ID \
    --models $MODELS
done
```
