# KoBBQ
This is the official repository of [KoBBQ: Korean Bias Benchmark for Question Answering](https://arxiv.org/abs/2307.16778) (TACL 2024).


## Webpage
- https://jinjh0123.github.io/KoBBQ


## Datasets
- Our KoBBQ datasets can be found in [KoBBQ/data](./data).

### Examples
![KoBBQ examples](examples.png)

### Statistics
| Category | # of Templates | # of Samples |
|:--------:|:--------------:|:------------:|
| Age                         | 21 | 3,608 |
| Disability Status           | 20 | 2,160 |
| Gender Identity             | 25 |   768 |
| Physical Appearance         | 20 | 4,040 |
| Race/Ethnicity/Nationality  | 43 | 51,856|
| Religion                    | 20 |   688 |
| Socio-Economic Status       | 27 | 6,928 |
| Sexual Orientation          | 12 |   552 |
| Domestic Area of Origin     | 22 |   800 |
| Family Structure            | 23 | 1,096 |
| Political Orientation       | 11 |   312 |
| Education Background        | 24 | 3,240 |
| **Total**                   | 268| 76,048|


## How to Evaluate
### With Evaluation Set
1. Put model outputs to ``prediction`` column in [KoBBQ/data/KoBBQ_test_samples.tsv](./data/KoBBQ_test_samples.tsv) and save the file as ``KoBBQ/evaluation/outputs/KoBBQ_test/KoBBQ_test_evaluation_1_{$MODEL}.tsv``.
    - The model outputs should be one of the choices (as in ``choices`` column). Otherwise, they will be regarded as <em>out-of-choice</em> answers.
2. Run [KoBBQ/evaluation/5_evaluation.py](./evaluation/5_evaluation.py) with ``test`` option.
    ```bash
    cd evaluation
    python3 5_evaluation.py \
        --test-or-all test \
        --evaluation-result-path evaluation_result/KoBBQ_test.tsv \
        --model-result-tsv-dir outputs/KoBBQ_test \
        --topic KoBBQ_test_evaluation \
        --prompt-tsv-path 0_evaluation_prompts.tsv \
        --prompt-id 1 \
        --models $MODEL
    ```

### With Entire Set
1. Put model outputs to ``prediction`` column in [KoBBQ/data/KoBBQ_all_samples.tsv](./data/KoBBQ_all_samples.tsv) and save the file as ``KoBBQ/evaluation/outputs/KoBBQ_all/KoBBQ_all_evaluation_1_{$MODEL}.tsv``.
    - The model outputs should be one of the choices (as in ``choices`` column). Otherwise, they will be regarded as <em>out-of-choice</em> answers.
2. Run [KoBBQ/evaluation/5_evaluation.py](./evaluation/5_evaluation.py) with ``all`` option.
    ```bash
    cd evaluation
    python3 5_evaluation.py \
        --test-or-all all \
        --evaluation-result-path evaluation_result/KoBBQ_all.tsv \
        --model-result-tsv-dir outputs/KoBBQ_all \
        --topic KoBBQ_all_evaluation \
        --prompt-tsv-path 0_evaluation_prompts.tsv \
        --prompt-id 1 \
        --models $MODEL
    ```

## Ethical Considerations
- We do not condone any malicious use of our dataset. It must not be used as training data to automatically generate and publish biased languages targeting specific groups. We strongly encourage researchers and practitioners to utilize this dataset in beneficial ways, such as mitigating bias
in language models.
