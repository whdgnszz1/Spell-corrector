import re
from Levenshtein import distance as levenshtein_distance


def post_process(raw_prediction, err_sentence):
    cleaned = re.sub(r'(\b\w+\b)(\s+\1)+', r'\1', raw_prediction)
    cleaned = re.sub(r'(.)\1+', r'\1', cleaned)
    target_length = len(err_sentence)
    if len(cleaned) > target_length * 1.5:
        cleaned = cleaned[:target_length]
    return cleaned.strip()


def find_closest_candidate(post_processed_prd, candidates):
    """post_processed 결과와 후보군 중 편집 거리가 가장 짧은 값을 반환"""
    min_distance = float('inf')
    closest_candidate = None
    for candidate in candidates:
        dist = levenshtein_distance(post_processed_prd, candidate)
        if dist < min_distance:
            min_distance = dist
            closest_candidate = candidate
    return closest_candidate


def get_ngram(text, n_gram):
    ngram_list = []
    text_length = len(text)
    for i in range(text_length - n_gram + 1):
        ngram_list.append(text[i:i + n_gram])
    return ngram_list


def calc_f_05(cor_sentence, prd_sentence, n_gram):
    prd_word_list = get_ngram(prd_sentence, n_gram)
    cor_word_list = get_ngram(cor_sentence, n_gram)
    if not cor_word_list:
        return 0, 0, 0
    cnt = 0
    for idx in range(len(prd_word_list)):
        start_idx = max(0, idx - 2)
        end_idx = min(len(cor_word_list), idx + 3)
        if prd_word_list[idx] in cor_word_list[start_idx:end_idx]:
            cnt += 1
    if not prd_word_list:
        return 0, 0, 0
    precision = cnt / len(prd_word_list)
    recall = cnt / len(cor_word_list)
    if (0.25 * precision + recall) == 0:
        return 0, 0, 0
    f_05 = 1.25 * (precision * recall) / (0.25 * precision + recall)
    return precision, recall, f_05
