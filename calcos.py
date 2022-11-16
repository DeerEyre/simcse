import re
import difflib
import math
import text2vec
import jieba
import jsonlines
from tqdm import tqdm
import numpy as np
from loguru import logger
from pathlib2 import Path
from data import load_data, split_sentence, save_data
from threading import Thread
from multiprocessing import Pool
import Levenshtein
import warnings
import datetime as dt
from collections import defaultdict
import multiprocessing
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#multiprocessing.set_start_method('spawn', force=True)
#sim = text2vec.Similarity()

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
stopwords = stopwordslist('stop_words.txt')

def seg_sentence(sen, stopwords):
    cut_sen = jieba.lcut(sen)
    ret = []
    for word in cut_sen:
        if word not in stopwords:
            ret.append(word)
    return ret

def string_cos_similarity(str1, str2):
    """
    计算字符串的余弦距离
    :param str1: 字符串1
    :param str2: 字符串2
    :return: 返回相似度
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

def pad_to_maxlen(list1, list2):
    len1 = len(list1)
    len2 = len(list2)
    max_len = max(len1, len2)
    if len(list1) < max_len:
        list1 = list1 + [""] * (max_len - len1)
    if len(list2) < max_len:
        list2 = list2 + [""] * (max_len - len2)
    return list1, list2

def filter_length_ratio(str1: str, str2: str) -> bool:
    """计算两个字符串的长度比值, 并且过滤掉比值太低的"""
    ratio1 = len(str1) / len(str2)
    ratio2 = len(str2) / len(str1)
    if ratio1 < 0.8 or ratio2 < 0.8:
        return False
    else:
        return True


def get_sentences_cos_similarity(list1, list2):
    sim_list = []
    logger.info("Start to calculate the similarities of two sentences")
    assert len(list1) == len(list2)
    for sent1, sent2 in tqdm(zip(list1, list2), total=len(list1)):
        #sims = []
        for str1 in sent1:
            each_sim = list(map(lambda str2: string_cos_similarity(str1, str2), sent2))
            max_sim = max(each_sim)
            max_index = np.argmax(np.array(each_sim))
            best_match = sent2[max_index]

            if 0.8 <= max_sim <= 0.9 and filter_length_ratio(str1, best_match):
                data = {
                    "max_sim": max_sim,
                    "str1": str1,
                    "str2": best_match
                }
                sim_list.append(data)  # 相似度最大值，和最大值对应的元素下标
        #sim_list.append(sims)
    return sim_list


def calculate_L_distance(list1, list2):
    all_data = []
    for sent1, sent2 in tqdm(zip(list1, list2), total=len(list1)):
        l_dist = Levenshtein.ratio(sent1, sent2)
        if 0.8 <= l_dist <= 0.9:
            data = {
                "Levenshtein distance": l_dist,
                "String1": sent1,
                "String2": sent2
            }
            all_data.append(data)

    return all_data

def extract_change(origins, trans):
    change_list = []
    for sen1,sen2 in zip(origins,trans):
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
        change = []
        for key in logic:
            if key == '':
                continue
            change.append((logic[key][0], key))

        data = {
            "src": sen1,
            "tgt": sen2,
            "words": change
        }
        change_list.append(data)
    return change_list


class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.ret = self.func(*self.args)

    def get_result(self):
        try:
            return self.ret
        except:
            return None


if __name__ == "__main__":
    path = 'sim_sentence.jsonl'
    save_path = 'extract_change.jsonl'
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    with jsonlines.open(path, 'r') as reader:
        origins = []
        trans = []
        for line in reader:
            origin = line['str1']
            tran = line['str2']
            if len(origin) == 0 or len(tran) == 0: # �~G滤空�~@�
                continue
            origins.append(origin)
            trans.append(tran)
    origin1, origin2, origin3, origin4, origin5 = origins[:150000], origins[150000:300000],\
            origins[300000:450000], origins[450000:600000], origins[600000:]

    tran1, tran2, tran3, tran4, tran5 = trans[:150000], trans[150000:300000],\
            trans[300000:450000], trans[450000:600000], trans[600000:]
    logger.info("Start to use multiple threads to process the data.")
    """
    thread1 = MyThread(get_sentences_cos_similarity, args=(origin1, tran1,))
    thread2 = MyThread(get_sentences_cos_similarity, args=(origin2, tran2,))
    thread3 = MyThread(get_sentences_cos_similarity, args=(origin3, tran3,))
    thread4 = MyThread(get_sentences_cos_similarity, args=(origin4, tran4,))
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread4.join()
    thread3.join()
    thread2.join()
    thread1.join()
    ret1 = thread1.get_result()
    ret2 = thread2.get_result()
    ret3 = thread3.get_result()
    ret4 = thread4.get_result()
    ret = ret1 + ret2 + ret3 + ret4
    """

    pool = Pool(5)
    ret = []
    ret.append(pool.apply_async(extract_change, (origin1, tran1,)))
    ret.append(pool.apply_async(extract_change, (origin2, tran2,)))
    ret.append(pool.apply_async(extract_change, (origin3, tran3,)))
    ret.append(pool.apply_async(extract_change, (origin4, tran4,)))
    ret.append(pool.apply_async(extract_change, (origin5, tran5,)))
    pool.close()
    pool.join()

    sim_list = []
    for i in ret:
        sim_list += i.get()

    logger.info(f"The length of similar pairs are {len(sim_list)}")

    #sim_list = get_sentences_cos_similarity(origins_split, trans_split)
    #logger.info(f"The length of similarity list is {len(sim_list)}")
    #print(sim_list[:2])
    print(sim_list[:2])
    logger.info(f'Start to save the similarity data')
    save_data(save_path, sim_list)
    logger.info("End of work!")
