from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datasets
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
import sys
import json
from datetime import datetime
import argparse
from omegaconf import OmegaConf
import random

# 한글 분리 및 조합 함수
def decompose_hangul(char):
    if not (0xAC00 <= ord(char) <= 0xD7A3):
        return char
    code = ord(char) - 0xAC00
    jongseong = code % 28
    jungseong = ((code - jongseong) // 28) % 21
    choseong = ((code - jongseong) // 28) // 21
    return choseong, jungseong, jongseong

def compose_hangul(choseong, jungseong, jongseong):
    return chr(0xAC00 + (choseong * 21 + jungseong) * 28 + jongseong)

# 인접 키 정의 (두벌식 자판 기준)
choseong_adjacent = {
    0: [1, 2, 6], 1: [0, 2], 2: [0, 1, 3], 3: [2, 4], 4: [3, 5],
    5: [4, 6], 6: [0, 5, 7], 7: [6, 8], 8: [7, 9], 9: [8]
}
jungseong_adjacent = {
    0: [1, 4], 1: [0, 2], 2: [1, 3], 3: [2], 4: [0, 5], 5: [4]
}

# 증강 함수
def substitute(char, choseong_adjacent, jungseong_adjacent):
    if not (0xAC00 <= ord(char) <= 0xD7A3):
        return char
    choseong, jungseong, jongseong = decompose_hangul(char)
    choice = random.choice(['choseong', 'jungseong'])
    if choice == 'choseong' and choseong in choseong_adjacent:
        new_choseong = random.choice(choseong_adjacent[choseong])
        return compose_hangul(new_choseong, jungseong, jongseong)
    elif choice == 'jungseong' and jungseong in jungseong_adjacent:
        new_jungseong = random.choice(jungseong_adjacent[jungseong])
        return compose_hangul(choseong, new_jungseong, jongseong)
    return char

def augment_substitute(sentence, prob=0.1):
    augmented = [substitute(char, choseong_adjacent, jungseong_adjacent)
                 if random.random() < prob else char
                 for char in sentence]
    return ''.join(augmented)

def augment_insert(sentence, prob=0.1):
    augmented = []
    for char in sentence:
        augmented.append(char)
        if random.random() < prob:
            new_choseong = random.choice(list(choseong_adjacent.keys()))
            new_jungseong = random.choice(list(jungseong_adjacent.keys()))
            augmented.append(compose_hangul(new_choseong, new_jungseong, 0))
    return ''.join(augmented)

def augment_delete(sentence, prob=0.1):
    if len(sentence) <= 1:
        return sentence
    augmented = [char for char in sentence if random.random() >= prob]
    if not augmented:
        augmented = [random.choice(sentence)]
    return ''.join(augmented)

def augment_transpose(sentence, prob=0.1):
    if len(sentence) < 2:
        return sentence
    augmented = list(sentence)
    for i in range(len(augmented) - 1):
        if random.random() < prob:
            augmented[i], augmented[i + 1] = augmented[i + 1], augmented[i]
    return ''.join(augmented)

def augment_sentence(sentence, prob=0.1):
    methods = [augment_substitute, augment_insert, augment_delete, augment_transpose]
    method = random.choice(methods)
    return method(sentence, prob)

# n-gram 및 F0.5 계산 함수
def get_ngram(text, n_gram):
    ngram_list = []
    text_length = len(text)
    for i in range(text_length - n_gram + 1):
        ngram_list.append(text[i:i + n_gram])
    return ngram_list

def calc_f_05(cor_sentence, prd_sentence, n_gram=2):
    prd_word_list = get_ngram(prd_sentence, n_gram)
    cor_word_list = get_ngram(cor_sentence, n_gram)
    if not cor_word_list or not prd_word_list:
        return 0, 0, 0
    cnt = 0
    for idx in range(len(prd_word_list)):
        start_idx = max(0, idx - 2)
        end_idx = min(len(cor_word_list), idx + 3)
        if prd_word_list[idx] in cor_word_list[start_idx:end_idx]:
            cnt += 1
    precision = cnt / len(prd_word_list) if prd_word_list else 0
    recall = cnt / len(cor_word_list) if cor_word_list else 0
    if precision + recall == 0:
        f_05 = 0
    else:
        f_05 = 1.25 * (precision * recall) / (0.25 * precision + recall)
    return precision, recall, f_05

# 데이터셋 생성 함수 (정확한 문장 추가 및 augment_prob 조정)
def make_dataset(train_data_path_list, validation_data_path_list, augment_prob=0.3):
    loaded_data_dict = {
        'train': {'err_sentence': [], 'cor_sentence': []},
        'validation': {'err_sentence': [], 'cor_sentence': []}
    }
    # train 데이터 로드
    for i, train_data_path in enumerate(train_data_path_list):
        with open(train_data_path, 'r') as f:
            _temp_json = json.load(f)
        loaded_data_dict['train']['err_sentence'].extend(
            list(map(lambda x: str(x['annotation']['err_sentence']), _temp_json['data'])))
        loaded_data_dict['train']['cor_sentence'].extend(
            list(map(lambda x: str(x['annotation']['cor_sentence']), _temp_json['data'])))
        print(f'train data {i} :', len(_temp_json['data']))

    # 원본 cor_sentence 저장
    original_cor_sentences = loaded_data_dict['train']['cor_sentence'].copy()

    # train 데이터 증강
    augmented_train = {'err_sentence': [], 'cor_sentence': []}
    for cor in original_cor_sentences:
        err = augment_sentence(cor, prob=augment_prob)
        augmented_train['err_sentence'].append(err)
        augmented_train['cor_sentence'].append(cor)
    loaded_data_dict['train']['err_sentence'].extend(augmented_train['err_sentence'])
    loaded_data_dict['train']['cor_sentence'].extend(augmented_train['cor_sentence'])

    # 정확한 문장 추가 (원본 데이터의 10%)
    num_original = len(original_cor_sentences)
    num_to_add = int(num_original * 0.1)
    indices = random.sample(range(num_original), num_to_add)
    for idx in indices:
        cor = original_cor_sentences[idx]
        loaded_data_dict['train']['err_sentence'].append(cor)
        loaded_data_dict['train']['cor_sentence'].append(cor)

    # validation 데이터 로드
    for i, validation_data_path in enumerate(validation_data_path_list):
        with open(validation_data_path, 'r') as f:
            _temp_json = json.load(f)
        loaded_data_dict['validation']['err_sentence'].extend(
            list(map(lambda x: str(x['annotation']['err_sentence']), _temp_json['data'])))
        loaded_data_dict['validation']['cor_sentence'].extend(
            list(map(lambda x: str(x['annotation']['cor_sentence']), _temp_json['data'])))
        print(f'validation data {i} :', len(_temp_json['data']))

    dataset_dict = {}
    for _trg in loaded_data_dict.keys():
        dataset_dict[_trg] = datasets.Dataset.from_dict(loaded_data_dict[_trg], split=_trg)
    dataset = datasets.DatasetDict(dataset_dict)
    return dataset

# 전처리 함수
def preprocess_function(df, tokenizer, src_col, tgt_col, max_length):
    inputs = df[src_col]
    targets = df[tgt_col]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_length, truncation=True)
    model_inputs["labels"] = labels['input_ids']
    return model_inputs

# 학습 함수 (compute_metrics 추가 및 파라미터 조정)
def train(config):
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Model Load Start ======')
    model = AutoModelForSeq2SeqLM.from_pretrained(config.pretrained_model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Model Load Finished ======')

    # F0.5 점수를 계산하는 compute_metrics 함수 정의
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = [[token if token != -100 else tokenizer.pad_token_id for token in label] for label in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        f_05_list = []
        for pred, label in zip(decoded_preds, decoded_labels):
            _, _, f_05 = calc_f_05(label, pred, n_gram=2)
            f_05_list.append(f_05)
        return {"f_05": sum(f_05_list) / len(f_05_list)}

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    print(f'[{_now_time}] ====== Data Load Start ======')
    dataset = make_dataset(config.train_data_path_list, config.validation_data_path_list, augment_prob=0.3)
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Data Load Finished ======')

    print(f'[{_now_time}] ====== Data Preprocessing Start ======')
    dataset_tokenized = dataset.map(
        lambda d: preprocess_function(d, tokenizer, config.src_col, config.tgt_col, config.max_length),
        batched=True, batch_size=config.per_device_train_batch_size)
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Data Preprocessing Finished ======')

    # 훈련 인자 설정 (생성 파라미터 및 F0.5 기반 모델 선택 추가)
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        fp16=config.fp16,
        weight_decay=0.1,  # 정규화 강화
        do_eval=config.do_eval,
        evaluation_strategy=config.evaluation_strategy,
        warmup_ratio=config.warmup_ratio,
        log_level=config.log_level,
        logging_dir=config.logging_dir,
        logging_strategy=config.logging_strategy,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model="f_05",  # F0.5 점수로 최적 모델 선택
        greater_is_better=True,
        dataloader_num_workers=config.dataloader_num_workers,
        group_by_length=config.group_by_length,
        report_to=config.report_to,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        predict_with_generate=True,  # 생성 기반 평가
        generation_num_beams=5,  # 빔 서치 활용
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_tokenized['train'],
        eval_dataset=dataset_tokenized['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file')
    args = parser.parse_args(sys.argv[1:])
    config = OmegaConf.load(args.config_file)
    save_path = './data/results'
    os.makedirs(save_path, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ========== Train Start ==========')
    print(f'DEVICE : {config.CUDA_VISIBLE_DEVICES}')
    print(f'MODEL NAME : {config.pretrained_model_name}')
    print(f'TRAIN FILE PATH :')
    for _path in config.train_data_path_list:
        print(f' - {_path}')
    print(f'VALIDATION FILE PATH :')
    for _path in config.validation_data_path_list:
        print(f' - {_path}')
    print(f'SAVE PATH : {config.output_dir}')
    train(config)
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ========== Train Finished ==========')