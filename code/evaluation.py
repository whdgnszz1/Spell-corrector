from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import datasets
import pandas as pd
import os
import sys
import json
from tqdm import tqdm
from datetime import datetime
import argparse
import Levenshtein
import random

def load_datasets(test_file, sample_size=100):
    with open(test_file, 'r') as f:
        json_dataset = json.load(f)

    list_dataset = {
        'id': [str(data['metadata_info']['id']) for data in json_dataset['data']],
        'err_sentence': [data['annotation']['err_sentence'] for data in json_dataset['data']],
        'cor_sentence': [data['annotation']['cor_sentence'] for data in json_dataset['data']]
    }

    indices = random.sample(range(len(list_dataset['id'])), min(sample_size, len(list_dataset['id'])))
    sampled_dataset = {
        'id': [list_dataset['id'][i] for i in indices],
        'err_sentence': [list_dataset['err_sentence'][i] for i in indices],
        'cor_sentence': [list_dataset['cor_sentence'][i] for i in indices]
    }

    dataset_dict = {
        'test': datasets.Dataset.from_dict(
            sampled_dataset,
            features=datasets.Features({
                'id': datasets.Value('string'),
                'err_sentence': datasets.Value('string'),
                'cor_sentence': datasets.Value('string')
            }),
            split='test'
        )
    }
    return datasets.DatasetDict(dataset_dict)


def load_candidates(candidate_file):
    with open(candidate_file, 'r') as f:
        json_dataset = json.load(f)
    candidates = [data['annotation']['cor_sentence'] for data in json_dataset['data']]
    return candidates


def calc_accuracy(cor_sentence, prd_sentence):
    return 1.0 if prd_sentence == cor_sentence else 0.0


def calc_edit_distance(cor_sentence, prd_sentence):
    return Levenshtein.distance(cor_sentence, prd_sentence)


def calc_char_accuracy(cor_sentence, prd_sentence):
    if not cor_sentence:
        return 0.0
    matches = sum(1 for c1, c2 in zip(cor_sentence, prd_sentence) if c1 == c2)
    return matches / len(cor_sentence)


def remove_repetition(sentence):
    words = sentence.split()
    seen = set()
    result = []
    for word in words:
        if word not in seen:
            seen.add(word)
            result.append(word)
    return ' '.join(result)


def find_closest_candidates(raw_prd_sentence, candidates, top_n=3):
    distances = [(candidate, Levenshtein.distance(raw_prd_sentence, candidate)) for candidate in candidates]
    sorted_distances = sorted(distances, key=lambda x: x[1])
    return sorted_distances[:top_n]


def my_train(gpus='cpu', model_path=None, test_file=None, save_path=None, pb=False):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset = load_datasets(test_file, sample_size=100)
    candidates = [data['cor_sentence'] for data in dataset['test']]

    device = torch.device(gpus)
    model.to(device)

    id_list, err_sentence_list, cor_sentence_list = [], [], []
    prd_sentence_list, accuracy_list, edit_distance_list, char_accuracy_list = [], [], [], []

    data_len = len(dataset['test'])

    print('=' * 100)
    for n in tqdm(range(data_len), disable=pb):
        data_id = dataset['test'][n]['id']
        err_sentence = dataset['test'][n]['err_sentence']
        cor_sentence = dataset['test'][n]['cor_sentence']

        tokenized = tokenizer(err_sentence, return_tensors='pt')
        input_ids = tokenized['input_ids'].to(device)

        cor_length = len(tokenizer.tokenize(cor_sentence))
        max_length = cor_length + 3
        min_length = max(cor_length - 2, 1)

        res = model.generate(
            inputs=input_ids,
            num_beams=10,
            num_return_sequences=1,
            temperature=1.0,
            repetition_penalty=2.0,
            length_penalty=0.8,
            no_repeat_ngram_size=2,
            max_length=max_length,
            min_length=min_length,
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        ).cpu().tolist()

        raw_prd_sentence = tokenizer.decode(res[0], skip_special_tokens=True).strip()
        prd_sentence = raw_prd_sentence

        closest_candidates = find_closest_candidates(prd_sentence, candidates, top_n=3)
        prd_sentence, edit_distance = closest_candidates[0]

        accuracy = calc_accuracy(cor_sentence, prd_sentence)
        char_accuracy = calc_char_accuracy(cor_sentence, prd_sentence)

        id_list.append(data_id)
        err_sentence_list.append(err_sentence)
        cor_sentence_list.append(cor_sentence)
        prd_sentence_list.append(prd_sentence)
        accuracy_list.append(accuracy)
        edit_distance_list.append(edit_distance)
        char_accuracy_list.append(char_accuracy)

        _cnt = n + 1
        _per_calc = round(_cnt / data_len, 4)
        _now_time = datetime.now().__str__()
        print(f'[{_now_time}] - [{_per_calc:6.1%} {_cnt:06,}/{data_len:06,}] - Evaluation Result (Data id : {data_id})')
        print(f'{" " * 30} >       TEST : {err_sentence}')
        print(f'{" " * 30} >    RAW PREDICT : {raw_prd_sentence}')
        print(f'{" " * 30} >    PREDICT : {prd_sentence}')
        print(f'{" " * 30} >      LABEL : {cor_sentence}')
        print(f'{" " * 30} > ACCURACY : {accuracy:6.3f}')
        print(f'{" " * 30} > EDIT DISTANCE : {edit_distance}')
        print(f'{" " * 30} > CHAR ACCURACY : {char_accuracy:6.3f}')
        print(f'{" " * 30} > TOP 3 CANDIDATES (w/ PREDICT):')
        for i, (candidate, distance) in enumerate(closest_candidates, 1):
            print(f'{" " * 30} >     {i}. {candidate} (편집 거리: {distance})')
        print('=' * 100)

    save_file_name = os.path.split(test_file)[-1].replace('.json', '') + '.csv'
    save_file_path = os.path.join(save_path, save_file_name)
    df = pd.DataFrame({
        'id': id_list,
        'err_sentence': err_sentence_list,
        'prd_sentence': prd_sentence_list,
        'cor_sentence': cor_sentence_list,
        'accuracy': accuracy_list,
        'edit_distance': edit_distance_list,
        'char_accuracy': char_accuracy_list
    })
    df.to_csv(save_file_path, index=True)
    print(f'[{datetime.now()}] - Save Result File(.csv) - {save_file_path}')

    print('=' * 100)
    mean_accuracy = sum(accuracy_list) / len(accuracy_list)
    mean_edit_distance = sum(edit_distance_list) / len(edit_distance_list)
    mean_char_accuracy = sum(char_accuracy_list) / len(char_accuracy_list)
    print(f'       Average Accuracy : {mean_accuracy:6.3f}')
    print(f'       Average Edit Distance : {mean_edit_distance:6.3f}')
    print(f'       Average Char Accuracy : {mean_char_accuracy:6.3f}')
    print('=' * 100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_no", dest="gpu_no", type=int, action="store")
    parser.add_argument("--model_path", dest="model_path", type=str, action="store")
    parser.add_argument("--test_file", dest="test_file", type=str, action="store")
    parser.add_argument("-pb", dest="pb", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    save_path = './data/results'
    os.makedirs(save_path, exist_ok=True)

    gpu_no = 'cpu'
    if args.gpu_no is not None:
        gpu_no = f'cuda:{args.gpu_no}'

    pb = not args.pb

    print(f'[{datetime.now()}] ========== Evaluation Start ==========')
    print(f'DEVICE : {gpu_no}, MODEL PATH : {args.model_path}, FILE PATH : {args.test_file}, SAVE PATH : {save_path}')
    my_train(gpu_no, model_path=args.model_path, test_file=args.test_file, save_path=save_path, pb=pb)
    print(f'[{datetime.now()}] ========== Evaluation Finished ==========')
