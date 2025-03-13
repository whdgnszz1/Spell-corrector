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
