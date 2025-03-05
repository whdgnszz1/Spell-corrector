from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from transformers import DataCollatorForSeq2Seq, get_linear_schedule_with_warmup
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
import os
import sys
import json
from datetime import datetime
import argparse
from omegaconf import OmegaConf
from datasets import Dataset, DatasetDict, Features, Value
import torch
import torch.nn.functional as F
import random
from torch.utils.data import WeightedRandomSampler, DataLoader
import Levenshtein

INCORRECT_CASES_FILE = 'incorrect_cases.json'


def load_candidates(candidate_file):
    with open(candidate_file, 'r') as f:
        json_dataset = json.load(f)
    candidates = [data['annotation']['cor_sentence'] for data in json_dataset['data']]
    return candidates


def remove_repetition(sentence):
    words = sentence.split()
    seen = set()
    result = []
    for word in words:
        if word not in seen:
            seen.add(word)
            result.append(word)
    return ' '.join(result)


def find_closest_candidates(raw_prd_sentence, candidates, top_n=1):
    distances = [(candidate, Levenshtein.distance(raw_prd_sentence, candidate)) for candidate in candidates]
    sorted_distances = sorted(distances, key=lambda x: x[1])
    return sorted_distances[:top_n]


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, weights=None, full_eval_dataset=None, candidate_file=None, is_reinforce=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_eval_dataset = full_eval_dataset
        self.candidates = load_candidates(candidate_file) if candidate_file else []
        self.is_reinforce = is_reinforce
        if weights is not None:
            self.sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        else:
            self.sampler = None

    def get_train_dataloader(self):
        if self.sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=self.sampler,
                collate_fn=self.data_collator
            )
        return super().get_train_dataloader()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = self.create_optimizer()
        num_warmup_steps = int(self.args.warmup_ratio * num_training_steps)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return self.optimizer, self.lr_scheduler

    def evaluate(self, *args, **kwargs):
        eval_result = super().evaluate(*args, **kwargs)
        model = self.model
        model.eval()
        predictions = []
        for batch in self.get_eval_dataloader():
            inputs = {k: v.to(self.args.device) for k, v in batch.items() if k != 'labels'}
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=4,
                    bos_token_id=0,
                    eos_token_id=1,
                    pad_token_id=3,
                    decoder_start_token_id=1,
                    forced_eos_token_id=1
                )
            predictions.extend(generated_ids.cpu().tolist())

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = self.eval_dataset['labels']
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        incorrect_cases = []
        for i, (raw_pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
            prd_sentence = remove_repetition(raw_pred)
            cor_word_count = len(label.split())
            prd_words = prd_sentence.split()
            prd_sentence = ' '.join(prd_words[:cor_word_count])
            closest_candidates = find_closest_candidates(prd_sentence, self.candidates, top_n=1)
            prd_sentence, _ = closest_candidates[0]
            edit_distance = Levenshtein.distance(prd_sentence, label)
            if edit_distance > 2:
                incorrect_cases.append({
                    'err_sentence': self.full_eval_dataset[i]['err_sentence'],
                    'cor_sentence': label,
                    'case': self.full_eval_dataset[i]['case'],
                    'edit_distance': edit_distance
                })

        if incorrect_cases:
            with open(INCORRECT_CASES_FILE, 'w') as f:
                json.dump(incorrect_cases, f, ensure_ascii=False, indent=4)
            print(f"Saved {len(incorrect_cases)} incorrect cases with edit distance > 2 to {INCORRECT_CASES_FILE}")

        return eval_result

    def train(self, resume_from_checkpoint=None, **kwargs):
        super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)
        if not self.is_reinforce:
            incorrect_cases = self.collect_incorrect_cases()
            if incorrect_cases:
                print(
                    f"Starting reinforcement learning for {len(incorrect_cases)} incorrect cases with edit distance > 2.")
                self.reinforce_learning(incorrect_cases)

    def collect_incorrect_cases(self):
        if os.path.exists(INCORRECT_CASES_FILE):
            with open(INCORRECT_CASES_FILE, 'r') as f:
                incorrect_cases = json.load(f)
            return incorrect_cases
        return []

    def reinforce_learning(self, incorrect_cases):
        reinforced_dataset = {
            'err_sentence': [case['err_sentence'] for case in incorrect_cases],
            'cor_sentence': [case['cor_sentence'] for case in incorrect_cases],
            'case': [case['case'] for case in incorrect_cases],
            'weight': [case['edit_distance'] for case in incorrect_cases]
        }
        reinforced_dataset = Dataset.from_dict(reinforced_dataset)
        reinforced_dataset_tokenized = reinforced_dataset.map(
            lambda d: preprocess_function(d, self.tokenizer, 'err_sentence', 'cor_sentence',
                                          self.args.generation_max_length),
            batched=True,
            batch_size=self.args.per_device_train_batch_size
        )
        weights = reinforced_dataset_tokenized['weight']
        train_dataset_processed = reinforced_dataset_tokenized.remove_columns(
            ['err_sentence', 'cor_sentence', 'case', 'weight']
        )
        reinforced_trainer = CustomSeq2SeqTrainer(
            model=self.model,
            args=self.args,
            train_dataset=train_dataset_processed,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
            weights=weights,
            full_eval_dataset=self.full_eval_dataset,
            candidate_file='../datasets/dataset_test_case3_1000.json',
            is_reinforce=True
        )
        reinforced_trainer.train()


def make_dataset(train_data_path_list, test_data_path_list):
    loaded_data_dict = {
        'train': {'err_sentence': [], 'cor_sentence': [], 'case': [], 'score': [], 'weight': []},
        'validation': {'err_sentence': [], 'cor_sentence': [], 'case': []},
        'test': {'err_sentence': [], 'cor_sentence': [], 'case': []}
    }

    for i, train_data_path in enumerate(train_data_path_list):
        with open(train_data_path, 'r') as f:
            _temp_json = json.load(f)
        for x in _temp_json['data']:
            err = x['annotation'].get('err_sentence', '')
            cor = x['annotation'].get('cor_sentence', '')
            case = str(x['annotation'].get('case', ''))
            if err and cor and case in ['1', '2', '3']:
                loaded_data_dict['train']['err_sentence'].append(err)
                loaded_data_dict['train']['cor_sentence'].append(cor)
                loaded_data_dict['train']['case'].append(case)
                loaded_data_dict['train']['score'].append(1.0)
                loaded_data_dict['train']['weight'].append(1.0)
        print(f'train data {i} :', len(_temp_json['data']))

    test_data = []
    for i, test_data_path in enumerate(test_data_path_list):
        with open(test_data_path, 'r') as f:
            _temp_json = json.load(f)
        for x in _temp_json['data']:
            err = x['annotation'].get('err_sentence', '')
            cor = x['annotation'].get('cor_sentence', '')
            case = str(x['annotation'].get('case', ''))
            if err and cor and case in ['1', '2', '3']:
                test_data.append({
                    'err_sentence': err,
                    'cor_sentence': cor,
                    'case': case
                })
        print(f'test data {i} :', len(_temp_json['data']))

    random.shuffle(test_data)
    val_size = int(len(test_data) * 0.1)
    val_data = test_data[:val_size]
    test_data = test_data[val_size:]

    for item in val_data:
        loaded_data_dict['validation']['err_sentence'].append(item['err_sentence'])
        loaded_data_dict['validation']['cor_sentence'].append(item['cor_sentence'])
        loaded_data_dict['validation']['case'].append(item['case'])

    for item in test_data:
        loaded_data_dict['test']['err_sentence'].append(item['err_sentence'])
        loaded_data_dict['test']['cor_sentence'].append(item['cor_sentence'])
        loaded_data_dict['test']['case'].append(item['case'])

    if os.path.exists(INCORRECT_CASES_FILE):
        with open(INCORRECT_CASES_FILE, 'r') as f:
            incorrect_cases = json.load(f)
        incorrect_set = set((case['err_sentence'], case['cor_sentence'], case['case']) for case in incorrect_cases)
        for i, (err, cor, case) in enumerate(zip(
                loaded_data_dict['train']['err_sentence'],
                loaded_data_dict['train']['cor_sentence'],
                loaded_data_dict['train']['case']
        )):
            if (err, cor, case) in incorrect_set:
                loaded_data_dict['train']['weight'][i] = 2.0
        print(f"Assigned weights to {len(incorrect_cases)} incorrect cases in training data.")

    features = Features({
        'err_sentence': Value('string'),
        'cor_sentence': Value('string'),
        'case': Value('string'),
        'score': Value('float32'),
        'weight': Value('float32')
    })

    dataset_dict = {
        'train': Dataset.from_dict(loaded_data_dict['train'], split='train', features=features),
        'validation': Dataset.from_dict(loaded_data_dict['validation'], split='validation', features=Features({
            'err_sentence': Value('string'),
            'cor_sentence': Value('string'),
            'case': Value('string')
        })),
        'test': Dataset.from_dict(loaded_data_dict['test'], split='test', features=Features({
            'err_sentence': Value('string'),
            'cor_sentence': Value('string'),
            'case': Value('string')
        }))
    }
    return DatasetDict(dataset_dict)


def preprocess_function(df, tokenizer, src_col, tgt_col, max_length):
    case_map = {
        '1': '[KOR_TO_ENG]',
        '2': '[ENG_TO_KOR]',
        '3': '[TYPO]'
    }
    inputs = [f"{case_map.get(case, '[UNKNOWN]')} {err}" for case, err in zip(df['case'], df[src_col])]
    targets = df[tgt_col]
    tokenized = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
    model_inputs = {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': tokenized['labels']
    }
    return model_inputs


def train(config):
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Model Load Start ======')
    global tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(config.pretrained_model_name).to(
        "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[KOR_TO_ENG]', '[ENG_TO_KOR]', '[TYPO]']})
    model.resize_token_embeddings(len(tokenizer))

    model.generation_config = GenerationConfig(
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=3,
        decoder_start_token_id=1,
        forced_eos_token_id=1
    )
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Model Load Finished ======')

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    print(f'[{_now_time}] ====== Data Load Start ======')
    dataset = make_dataset(config.train_data_path_list, config.test_data_path_list)
    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Data Load Finished ======')

    print(f'[{_now_time}] ====== Data Preprocessing Start ======')
    dataset_tokenized = dataset.map(
        lambda d: preprocess_function(d, tokenizer, config.src_col, config.tgt_col, config.max_length),
        batched=True,
        batch_size=config.per_device_train_batch_size
    )

    weights = dataset_tokenized['train']['weight']
    train_dataset_processed = dataset_tokenized['train'].remove_columns(
        ['err_sentence', 'cor_sentence', 'case', 'score', 'weight']
    )
    eval_dataset_processed = dataset_tokenized['validation'].remove_columns(
        ['err_sentence', 'cor_sentence', 'case']
    )
    full_eval_dataset = dataset_tokenized['validation']

    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ====== Data Preprocessing Finished ======')

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=100,
        gradient_accumulation_steps=2,
        warmup_ratio=0.05,
        fp16=True if torch.cuda.is_available() else False,
        weight_decay=config.weight_decay,
        do_eval=config.do_eval,
        evaluation_strategy="steps",
        log_level="info",
        logging_dir=config.logging_dir,
        logging_strategy="steps",
        logging_steps=10,
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=1,
        load_best_model_at_end=True,
        dataloader_num_workers=0,
        group_by_length=True,
        report_to=None,
        ddp_find_unused_parameters=False,
        label_smoothing_factor=0.05,
        predict_with_generate=True,
        generation_max_length=50,
    )

    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_processed,
        eval_dataset=eval_dataset_processed,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        weights=weights,
        full_eval_dataset=full_eval_dataset,
        candidate_file='../datasets/dataset_test_case3_1000.json',
        is_reinforce=False
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"Final model saved to {config.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file')
    args = parser.parse_args(sys.argv[1:])

    config_file = args.config_file
    config = OmegaConf.load(config_file)

    save_path = './data/results'
    os.makedirs(save_path, exist_ok=True)

    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ========== Train Start ==========')

    print(f'DEVICE : {"cuda" if torch.cuda.is_available() else "cpu"}')
    print(f'MODEL NAME : {config.pretrained_model_name}')
    print(f'TRAIN FILE PATH :')
    for _path in config.train_data_path_list:
        print(f' - {_path}')
    print(f'TEST FILE PATH :')
    for _path in config.test_data_path_list:
        print(f' - {_path}')
    print(f'SAVE PATH : {config.output_dir}')
    train(config)

    _now_time = datetime.now().__str__()
    print(f'[{_now_time}] ========== Train Finished ==========')
