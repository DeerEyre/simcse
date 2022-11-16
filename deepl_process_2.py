import jieba
import numpy as np
import re
import jsonlines
import random
import pandas as pd
original_sentence = []
similartiy_sentence = []
contractive_sentence = []

def get_word_vector(s1,s2):
    """
    :param s1: 句子1
    :param s2: 句子2
    :return: 返回句子的余弦相似度
    """
    # 分词
    cut1 = jieba.cut(s1)
    cut2 = jieba.cut(s2)
    list_word1 = (','.join(cut1)).split(',')
    list_word2 = (','.join(cut2)).split(',')

    # 列出所有的词,取并集
    key_word = list(set(list_word1 + list_word2))
    # 给定形状和类型的用0填充的矩阵存储向量
    word_vector1 = np.zeros(len(key_word))
    word_vector2 = np.zeros(len(key_word))

    # 计算词频
    # 依次确定向量的每个位置的值
    for i in range(len(key_word)):
        # 遍历key_word中每个词在句子中的出现次数
        for j in range(len(list_word1)):
            if key_word[i] == list_word1[j]:
                word_vector1[i] += 1
        for k in range(len(list_word2)):
            if key_word[i] == list_word2[k]:
                word_vector2[i] += 1

    # 输出向量
    return word_vector1, word_vector2




def cos_dist(vec1,vec2):
    """
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 返回两个向量的余弦相似度
    """
    dist1=float(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    return dist1


with open('deepl_translate_60W.jsonl', 'r+', encoding='utf8') as f:
    for item in jsonlines.Reader(f):
        original_sentence.append(item['original_sentence'])
        similartiy_sentence.append(item['translate_back_sentence'])

for i in range(len(original_sentence)):
    print(i)
    candidate = random.sample(original_sentence, 30)
    score = []
    for contra in candidate:
        vec1, vec2 = get_word_vector(original_sentence[i], contra)
        dist1 = cos_dist(vec1, vec2)
        score.append(dist1)
    max_index = score.index(max(score))
    if candidate[max_index] == original_sentence[i]:
        score[max_index] = 0
        max_index = score.index(max(score))
    contractive_sentence.append(candidate[max_index])

dataframe = pd.DataFrame({'sen0': original_sentence, 'sen1': similartiy_sentence, 'hard_neg': contractive_sentence})
dataframe.to_csv("deepl_train_data2", index=False, sep=',')




