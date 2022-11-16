import re
import difflib
import numpy as np
from tqdm import tqdm
from loguru import logger
from pathlib2 import Path
import jsonlines
import jieba
import math
import text2vec

def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  

def seg_sentence(sen, stopwords):
    cut_sen = jieba.lcut(sen)
    ret = []
    for word in cut_sen:
        if word not in stopwords:
            ret.append(word)
    return ret

def string_cos_similarity(str1, str2):
    """
    计�~W�~W符串�~Z~D�~Y弦�~]离
    :param str1: �~W符串1
    :param str2: �~W符串2
    :return: �~T�~[~^�~[�似度
    """

    cut_str1 = list(str1.replace(" ", ""))
    cut_str2 = list(str2.replace(" ", ""))

    all_char = set(cut_str1 + cut_str2)

    freq_str1 = [cut_str1.count(x) for x in all_char]
    freq_str2 = [cut_str2.count(x) for x in all_char]

    sum_all = sum(map(lambda z, y: z * y, freq_str1, freq_str2))
    sqrt_str1 = math.sqrt(sum(x ** 2 for x in freq_str1))
    sqrt_str2 = math.sqrt(sum(x ** 2 for x in freq_str2))

    similarity = sum_all / (sqrt_str1 * sqrt_str2)
    return similarity

if __name__ == '__main__':
    import warnings 
    import datetime as dt
    from collections import defaultdict
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    path = 'pp.jsonl'
    stopwords = stopwordslist('stop_words.txt')
    with jsonlines.open(path, 'r') as reader:
        origins = []
        trans = []
        for line in reader:
            origin = line['str1']
            tran = line['str2']
            if len(origin) == 0 or len(tran) == 0: # 过滤空值
                continue
            origins.append(origin)
            trans.append(tran)
    #sim = text2vec.Similarity()
    start_time = dt.datetime.now()
    for sen1,sen2 in zip(origins,trans):
        print(sen1)
        print(sen2)
        cut_sen1 = seg_sentence(sen1, stopwords)
        cut_sen2 = seg_sentence(sen2, stopwords)
        diff1 = list(set(cut_sen1)-set(cut_sen2))
        diff2 = list(set(cut_sen2)-set(cut_sen1))
        logic = defaultdict(list)
        for ork1 in diff1:
            max = 0
            ork = ''
            for ork2 in diff2:
                #score = sim.get_score(ork1, ork2)
                #score = string_cos_similarity(ork1, ork2)
                #score = edit_similar(ork1, ork2)
                score = difflib.SequenceMatcher(None, ork1, ork2).ratio()
                if score > max:
                    max = score
                    ork = ork2
            if ork not in logic:
                logic[ork] = [ork1, max]
            else:
                score = logic[ork][1]
                if score < max:
                    logic[ork] = [ork1, max]
        for key in logic:
            if key == '':
                continue
            print(f'{logic[key][0]} change to {key}')
                #print(f'{ork1} change to {ork} is {max}')
                #print(f'{ork1} change to {ork2} is {string_cos_similarity(ork1, ork2)}')
                #print(f'{ork1} change to {ork2} is {synonyms.compare(ork1, ork2, seg=False)}')
    end_time = dt.datetime.now()
    print(f'working is end, time costs {(end_time - start_time).seconds}')
    #print(sim.get_score('循环', '回收'))
