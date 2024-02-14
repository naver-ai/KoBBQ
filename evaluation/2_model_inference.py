import os
import re
import csv
import json
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime

from model_inference.openai_utils import GPT_MODEL, get_gpt_response
from model_inference.claude_utils import CLAUDE_MODEL, get_claude_response
from model_inference.hyperclova_utils import HYPERCLOVA_MODEL, get_hyperclova_response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--model-names', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--max-tokens', type=int, default=30)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if not args.data_path.endswith('.json'):
        raise ValueError

    data_path = Path(args.data_path)
    topic = data_path.name.replace('.json', '')
    print(topic)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_eval = args.model_names.split(',')

    for model_name in models_to_eval:
        accuracies = []

        data = json.load(open(data_path, 'r', encoding='utf-8'))
        prefix = data['prefix']

        output_path = output_dir / f'{topic}_{model_name}_predictions.tsv'
        if output_path.is_file():
            print(f'Continue on {output_path}')
            done_ids = pd.read_csv(output_dir / f'{topic}_{model_name}_predictions.tsv', sep='\t')['guid'].to_list()
        else:
            done_ids = []
            with open(output_dir / f'{topic}_{model_name}_predictions.tsv', 'w', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['time', 'topic', 'guid', 'truth', 'raw'])

        for idx, instance in enumerate(tqdm(data['data'], desc=model_name)):
            # instance: prompt, A, B, C, truth, guid
            if instance[-1] in done_ids:
                continue

            prompt = prefix + instance[0]

            if model_name in GPT_MODEL:
                result = get_gpt_response(
                    prompt,
                    model_name,
                    max_tokens=args.max_tokens,
                    greedy=True
                )
            elif model_name in HYPERCLOVA_MODEL:
                result = get_hyperclova_response(
                    prompt,
                    model_name,
                    max_tokens=args.max_tokens,
                    greedy=True
                )
            elif model_name in CLAUDE_MODEL:
                result = get_claude_response(
                    prompt,
                    model_name,
                    max_tokens=args.max_tokens
                )
            else:
                raise ValueError(model_name)
                
            if result is None:
                raise Exception("Model Inference Fail")

            open_trial = 0
            while True:
                if open_trial >= 10:
                    raise Exception("File Open Fail")

                try:
                    with open(output_dir / f"{topic}_{model_name}_predictions.tsv", "a", encoding="utf-8") as f:
                        writer = csv.writer(f, delimiter='\t')
                        writer.writerow([datetime.now(), topic, instance[-1], instance[-2], result])
                    break
                except KeyboardInterrupt:
                    raise Exception("Keyboard Interrupt")
                except:
                    print("open failed")
                    open_trial += 1
                    continue

        print(f"{topic} - {model_name} done")
    