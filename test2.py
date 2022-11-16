import torch
import os
from enum import Enum
from typing import List, Union, Optional
from tqdm.autonotebook import trange
import numpy as np
from transformers import AutoTokenizer, AutoModel
from loguru import logger
from tqdm.auto import tqdm, trange
import re
import difflib
import numpy as np
from tqdm import tqdm
from loguru import logger
from pathlib2 import Path
import jsonlines
import jieba
import math

class EncoderType(Enum):
    FIRST_LAST_AVG = 0
    LAST_AVG = 1
    CLS = 2
    POOLER = 3
    MEAN = 4

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return EncoderType[s]
        except KeyError:
            raise ValueError()


class SentenceModel:
    def __init__(
            self,
            model_name_or_path: str = "shibing624/text2vec-base-chinese",
            encoder_type: Union[str, EncoderType] = "MEAN",
            max_seq_length: int = 128,
            device: Optional[str] = None,
    ):
        """
        Initializes the base sentence model.
        :param model_name_or_path: The name of the model to load from the huggingface models library.
        :param encoder_type: The type of encoder to use, See the EncoderType enum for options:
            FIRST_LAST_AVG, LAST_AVG, CLS, POOLER(cls + dense), MEAN(mean of last_hidden_state)
        :param max_seq_length: The maximum sequence length.
        :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if GPU.
        bert model: https://huggingface.co/transformers/model_doc/bert.html?highlight=bert#transformers.BertModel.forward
        BERT return: <last_hidden_state>, <pooler_output> [hidden_states, attentions]
        Note that: in doc, it says <last_hidden_state> is better semantic summery than <pooler_output>.
        thus, we use <last_hidden_state>.
        """
        self.model_name_or_path = model_name_or_path
        encoder_type = EncoderType.from_string(encoder_type) if isinstance(encoder_type, str) else encoder_type
        if encoder_type not in list(EncoderType):
            raise ValueError(f"encoder_type must be in {list(EncoderType)}")
        self.encoder_type = encoder_type
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        device = "cuda"
        self.device = torch.device(device)
        logger.debug("Use device: {}".format(self.device))
        self.bert.to(self.device)
        self.results = {}  # Save training process evaluation result

    def __str__(self):
        return f"<SentenceModel: {self.model_name_or_path}, encoder_type: {self.encoder_type}, " \
               f"max_seq_length: {self.max_seq_length}>"

    def get_sentence_embeddings(self, input_ids, attention_mask, token_type_ids):
        """
        Returns the model output by encoder_type as embeddings.
        Utility function for self.bert() method.
        """
        model_output = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.encoder_type == EncoderType.FIRST_LAST_AVG:
            # Get the first and last hidden states, and average them to get the embeddings
            # hidden_states have 13 list, second is hidden_state
            first = model_output.hidden_states[1]
            last = model_output.hidden_states[-1]
            seq_length = first.size(1)  # Sequence length

            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # [batch, hid_size]
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # [batch, hid_size]
            final_encoding = torch.avg_pool1d(
                torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2),
                kernel_size=2).squeeze(-1)
            return final_encoding

        if self.encoder_type == EncoderType.LAST_AVG:
            sequence_output = model_output.last_hidden_state  # [batch_size, max_len, hidden_size]
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            return final_encoding

        if self.encoder_type == EncoderType.CLS:
            sequence_output = model_output.last_hidden_state
            return sequence_output[:, 0]  # [batch, hid_size]

        if self.encoder_type == EncoderType.POOLER:
            return model_output.pooler_output  # [batch, hid_size]

        if self.encoder_type == EncoderType.MEAN:
            """
            Mean Pooling - Take attention mask into account for correct averaging
            """
            token_embeddings = model_output.last_hidden_state  # Contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            final_encoding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9)
            return final_encoding  # [batch, hid_size]

    def encode(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 64,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: str = None,
        ):
        """
        Returns the embeddings for a batch of sentences.
        :param sentences: str/list, Input sentences
        :param batch_size: int, Batch size
        :param show_progress_bar: bool, Whether to show a progress bar for the sentences
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        """
        self.bert.eval()
        if device is None:
            device = self.device
        if convert_to_tensor:
            convert_to_numpy = False
        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_is_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(s) for s in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            # Compute sentences embeddings
            with torch.no_grad():
                embeddings = self.get_sentence_embeddings(
                    **self.tokenizer(sentences_batch, max_length=self.max_seq_length,
                                     padding=True, truncation=True, return_tensors='pt').to(device)
                )
            embeddings = embeddings.detach()
            if convert_to_numpy:
                embeddings = embeddings.cpu()
            all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

model = SentenceModel(model_name_or_path="./sentence_model")
def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

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

def get_score(sentence1, sentence2):
    """
    Get score between text1 and text2
    :param sentence1: str
    :param sentence2: str
    :return: float, score
    """
    res = 0.0
    sentence1 = sentence1.strip()
    sentence2 = sentence2.strip()
    if not sentence1 or not sentence2:
        return res
    emb1 = model.encode(sentence1)
    emb2 = model.encode(sentence2)
    res = cos_sim(emb1, emb2)[0]
    res = float(res)
    return res

if __name__ == '__main__':
    import warnings
    import datetime as dt
    import pdb
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
            if len(origin) == 0 or len(tran) == 0: # �~G滤空�~@�
                continue
            origins.append(origin)
            trans.append(tran)
    start_time = dt.datetime.now()
    for sen1,sen2 in zip(origins,trans):
        print(sen1)
        print(sen2)
        cut_sen1 = seg_sentence(re.sub('\W*', '', sen1), stopwords)
        cut_sen2 = seg_sentence(re.sub('\W*', '', sen2), stopwords)
        diff1 = list(set(cut_sen1)-set(cut_sen2))
        diff2 = list(set(cut_sen2)-set(cut_sen1))
        logic = defaultdict(list)
        #pdb.set_trace()
        for ork1 in diff1:
            max = 0
            ork = ''
            for ork2 in diff2:
                #score = sim.get_score(ork1, ork2)
                #score = string_cos_similarity(ork1, ork2)
                #score = edit_similar(ork1, ork2)
                #score = difflib.SequenceMatcher(None, ork1, ork2).ratio()
                score = get_score(ork1, ork2)
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
    end_time = dt.datetime.now()
    print(f'working is end, time costs {(end_time - start_time).seconds}')
