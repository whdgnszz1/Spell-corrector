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
import random  # 랜덤 샘플링을 위해 추가


def load_datasets(test_file):
    with open(test_file, 'r') as f:
        json_dataset = json.load(f)

    list_dataset = {
        'err_sentence': list(map(lambda x: str(x['annotation']['err_sentence']), json_dataset['data'])),
        'cor_sentence': list(map(lambda x: str(x['annotation']['cor_sentence']), json_dataset['data']))
    }

    dataset_dict = {
        'test': datasets.Dataset.from_dict(list_dataset, split='test')
    }
    dataset = datasets.DatasetDict(dataset_dict)
    return dataset


def get_ngram(text, n_gram):
    ngram_list = []
    text_length = len(text)
    for i in range(text_length - n_gram + 1):
        ngram_list.append(text[i:i + n_gram])
    return ngram_list


def calc_f_05(cor_sentence, prd_sentence, n_gram):
    # Generate n-grams for predicted and correct sentences
    prd_word_list = get_ngram(prd_sentence, n_gram)
    cor_word_list = get_ngram(cor_sentence, n_gram)

    # If cor_word_list is empty, return 0 for all metrics
    if not cor_word_list:
        return 0, 0, 0  # Precision, Recall, F0.5

    # Count matches between predicted and correct n-grams
    cnt = 0
    for idx in range(len(prd_word_list)):
        start_idx = max(0, idx - 2)
        end_idx = min(len(cor_word_list), idx + 3)
        if prd_word_list[idx] in cor_word_list[start_idx:end_idx]:
            cnt += 1

    # If prd_word_list is empty, return 0 for all metrics
    if not prd_word_list:
        return 0, 0, 0

    # Calculate precision, recall, and F0.5 score
    precision = cnt / len(prd_word_list)
    recall = cnt / len(cor_word_list)

    # Avoid division by zero in F0.5 calculation
    if (0.25 * precision + recall) == 0:
        return 0, 0, 0

    f_05 = 1.25 * (precision * recall) / (0.25 * precision + recall)
    return precision, recall, f_05


def my_train(gpus='cpu', model_path=None, test_file=None, eval_length=None, save_path=None, pb=False):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 데이터셋 로드
    dataset = load_datasets(test_file)

    # 랜덤 샘플링 적용
    if eval_length and eval_length < len(dataset['test']):
        # 데이터셋에서 eval_length만큼 랜덤 인덱스 선택
        indices = random.sample(range(len(dataset['test'])), eval_length)
        dataset['test'] = dataset['test'].select(indices)
        data_len = eval_length
    else:
        # eval_length가 없거나 데이터셋 크기보다 크면 전체 데이터 사용
        data_len = len(dataset['test'])

    device = torch.device(gpus)
    model.to(device)

    err_sentence_list = []
    cor_sentence_list = []
    prd_sentence_list = []
    precision_list = []
    recall_list = []
    f_05_list = []

    ngram = 2
    bar_length = 100
    _per_calc = 0.0

    print('=' * bar_length)
    for n in tqdm(range(data_len), disable=pb):
        err_sentence = dataset['test'][n]['err_sentence']
        err_sentence_list.append(err_sentence)
        cor_sentence = dataset['test'][n]['cor_sentence']
        cor_sentence_list.append(cor_sentence)
        tokenized = tokenizer(err_sentence, return_tensors='pt')
        input_ids = tokenized['input_ids']
        input_ids = input_ids.to(device)
        res = model.generate(
            inputs=input_ids,
            num_beams=20,
            num_return_sequences=2,
            temperature=2,
            repetition_penalty=0.2,
            length_penalty=0.2,
            no_repeat_ngram_size=2,
            max_length=input_ids.size()[1] + 5).cpu().tolist()[0]
        prd_sentence = tokenizer.decode(res).replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()
        _cnt = n + 1
        _per_calc = round(_cnt / data_len, 4)
        _now_time = datetime.now().__str__()
        _blank = ' ' * 30
        print(f'[{_now_time}] - [{_per_calc:6.1%} {_cnt:06,}/{data_len:06,}] - Evaluation Result')
        print(f'{_blank} >       TEST : {err_sentence}')
        print(f'{_blank} >    PREDICT : {prd_sentence}')
        print(f'{_blank} >      LABEL : {cor_sentence}')

        precision, recall, f_05 = calc_f_05(cor_sentence, prd_sentence, ngram)
        precision_list.append(precision)
        recall_list.append(recall)
        f_05_list.append(f_05)
        print(f'{_blank} >  PRECISION : {precision:6.3f}')
        print(f'{_blank} >     RECALL : {recall:6.3f}')
        print(f'{_blank} > F0.5 SCORE : {f_05:6.3f}')
        print('=' * bar_length)
        prd_sentence_list.append(prd_sentence)
        torch.cuda.empty_cache()

    _now_time = datetime.now().__str__()
    save_file_name = os.path.split(test_file)[-1].replace('.json', '') + '.csv'
    save_file_path = os.path.join(save_path, save_file_name)
    _df = pd.DataFrame({
        'err_sentence': err_sentence_list,
        'prd_sentence': prd_sentence_list,
        'cor_sentence': cor_sentence_list,
        'precision': precision_list,
        'recall': recall_list,
        'f_05': f_05_list
    })
    _df.to_csv(save_file_path, index=True)
    print(f'[{_now_time}] - Save Result File(.csv) - {save_file_path}')

    print('=' * bar_length)
    mean_precision = sum(precision_list) / len(precision_list)
    mean_recall = sum(recall_list) / len(recall_list)
    mean_f_05 = sum(f_05_list) / len(f_05_list)

    print(f'       Evaluation Ngram : {ngram}')
    print(f'      Average Precision : {mean_precision:6.3f}')
    print(f'         Average Recall : {mean_recall:6.3f}')
    print(f'     Average F0.5 score : {mean_f_05:6.3f}')
    print('=' * bar_length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_no", dest="gpu_no", type=int, action="store")
    parser.add_argument("--model_path", dest="model_path", type=str, action="store")
    parser.add_argument("--test_file", dest="test_file", type=str, action="store")
    parser.add_argument("--eval_length", dest="eval_length", type=int, action="store")
    parser.add_argument("-pb", dest="pb", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    save_path = './data/results'
    os.makedirs(save_path, exist_ok=True)

    gpu_no = 'cpu'
    if args.gpu_no or args.gpu_no == 0:
        gpu_no = f'cuda:{args.gpu_no}'

    if args.pb:
        args.pb = False
    else:
        args.pb = True

    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ========== Evaluation Start ==========')

    print(
        f'DEVICE : {gpu_no}, MODEL PATH : {args.model_path}, FILE PATH : {args.test_file}, DATA LENGTH : {args.eval_length}, SAVE PATH : {save_path}')
    my_train(gpu_no, model_path=args.model_path, test_file=args.test_file, eval_length=args.eval_length,
             save_path=save_path, pb=args.pb)

    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ========== Evaluation Finished ==========')
