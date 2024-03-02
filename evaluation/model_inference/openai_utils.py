'''
KoBBQ
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
'''

import os
import re
import sys
import time
import json
import openai
import requests


GPT_MODEL = ['davinci', 'gpt-3.5-turbo', 'gpt-4']
openai.organization = os.environ.get('OPENAI_ORG')
openai.api_key = os.environ.get('OPENAI')


def check_gpt_input_list(history):
    check = True
    for i, u in enumerate(history):
        if not isinstance(u, dict):
            check = False
            break
            
        if not u.get('role') or not u.get('content'):
            check = False
            break
        
    return check


def get_gpt_response(
    text,
    model_name,
    temperature=1.0,
    top_p=1.0,
    max_tokens=128,
    greedy=False,
    num_sequence=1,
    max_try=60,
    dialogue_history=None
):
    assert model_name in GPT_MODEL

    if model_name.startswith('gpt-3.5-turbo') or model_name.startswith('gpt-4'):
        if dialogue_history:
            if not check_gpt_input_list(dialogue_history):
                raise Exception('Input format is not compatible with chatgpt api! Please see https://platform.openai.com/docs/api-reference/chat')
            messages = dialogue_history
        else:
            messages = []
        
        messages.append({'role': 'user', 'content': text})

        prompt = {
            'model': model_name,
            'messages': messages,
            'temperature': 0. if greedy else temperature,
            'top_p': top_p,
            'max_tokens': max_tokens,
            'n': num_sequence
        }

    else:    
        prompt = {
            'model': model_name,
            'prompt': text,
            'temperature': 0. if greedy else temperature,
            'top_p': top_p,
            'max_tokens': max_tokens,
            'n': num_sequence
        }
    
    n_try = 0
    while True:
        if n_try == max_try:
            raise Exception('Something Wrong')
        
        try:
            if model_name.startswith('gpt-3.5-turbo') or model_name.startswith('gpt-4'):
                time.sleep(1)
                res = openai.ChatCompletion.create(**prompt)
                outputs = [o['message']['content'].strip("\n ") for o in res['choices']]
            else:
                res = openai.Completion.create(**prompt)
                outputs = [o['text'].strip('\n ') for o in res['choices']]
            break
            
        except KeyboardInterrupt:
            raise Exception('KeyboardInterrupt')
        except Exception as e:
            print(e)
            print('Exception: Sleep for 5 sec')
            time.sleep(5)
            n_try += 1
            continue
        
    if len(outputs) == 1:
        outputs = outputs[0]
        
    return outputs
