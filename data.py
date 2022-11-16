import re
from tqdm import tqdm
from loguru import logger
from pathlib2 import Path
import jsonlines


def load_data(path):
    with jsonlines.open(path, 'r') as reader:
        origins = []
        trans = []
        for line in reader:
            origin = line['original_sentence']
            tran = line['translate_back_sentence']
            if len(origin) == 0 or len(tran) == 0: # 过滤空值
                continue
            origins.append(origin)
            trans.append(tran)

    return origins, trans

def load_format_data(path):
    with jsonlines.open(path, 'r') as reader:
        list1 = []
        list2 = []
        for line in reader:
            str1 = line["str1"]
            str2 = line["str2"]
            list1.append(str1)
            list2.append(str2)

    return list1, list2

def split_sentence(origins, trans):
    origins_split, trans_split = [], []
    #pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
    #pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|=|\_|\+|，|。|；|‘|’|【|】|·|！| |…|（|）'
    pattern = r'\.|!|。|…|！|\?|？'
    for origin, tran in tqdm(zip(origins, trans), total=len(origins)):
        ret_origin = re.split(pattern, origin)
        ret_tran = re.split(pattern, tran)
        # 过滤空元素
        ret_origin = list(filter(lambda x: x != '', ret_origin))
        ret_tran = list(filter(lambda x: x != '', ret_tran))
        # 过滤长度小于6的
        ret_origin = list(filter(lambda x: len(x) > 10, ret_origin))
        ret_tran = list(filter(lambda x: len(x) > 10, ret_tran))
        if len(ret_tran) == 0 or len(ret_origin) == 0:
            # 过滤掉异常值
            continue
        assert len(ret_origin) > 0
        assert len(ret_tran) > 0
        origins_split.append(ret_origin)
        trans_split.append(ret_tran)

    logger.info(f"The lengths of origins and trans are {len(origins_split)} , {len(trans_split)}")
    return origins_split, trans_split

def save_data(path: str, data: list) -> None :
    with jsonlines.open(path, 'w') as writer:
        for i in tqdm(data, total=len(data)):
            writer.write(i)


if __name__ == "__main__":
    project_path = Path.cwd()
    data_path = Path(project_path, 'data', '0.jsonl')
    #data_path = Path(project_path, '0.jsonl')
    logger.info(f"The data path is {data_path}, and then reads data")
    origins, trans = load_data(data_path)
    logger.info("Starting to split the origin sentences and translated sentences")
    origins_split, trans_split = split_sentence(origins, trans)
    logger.info(f"The length of origin data and trans data are {len(origins_split)}, {len(trans_split)}")
    print(origins_split[:3], '\n')
    print(trans_split[:3])
