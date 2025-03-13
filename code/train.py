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
from utils import calc_f_05, augment_sentence


# 데이터셋 생성 함수
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


# 학습 함수
def train(config):
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Model Load Start ======')
    model = AutoModelForSeq2SeqLM.from_pretrained(config.pretrained_model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Model Load Finished ======')

    # F0.5 점수를 계산하는 compute_metrics 함수
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

    # 훈련 인자 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        fp16=config.fp16,
        weight_decay=0.1,
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
        metric_for_best_model="f_05",
        greater_is_better=True,
        dataloader_num_workers=config.dataloader_num_workers,
        group_by_length=config.group_by_length,
        report_to=config.report_to,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        predict_with_generate=True,
        generation_num_beams=5,
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
