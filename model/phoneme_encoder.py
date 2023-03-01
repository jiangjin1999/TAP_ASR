from operator import index
import torch

from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, BertPreTrainedModel, BertModel, BertConfig
import torch.nn as nn
from copy import deepcopy
import numpy as np
import torch
import pypinyin
from g2p_en import G2p
g2p = G2p()



class phoneme_encoder(BertPreTrainedModel):
    def __init__(self, config):
        super(phoneme_encoder, self).__init__(config)
        self.pho_embeddings = nn.Embedding(phoneme_convertor.get_pho_size(), config.hidden_size, padding_idx=0)
        self.pho_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        pho_config = deepcopy(config)
        pho_config.num_hidden_layers = 4
        self.pho_model = BertModel(pho_config)



    def forward(self, pho_idx, pho_lens, input_ids):
        input_shape = input_ids.size() 
        pho_embeddings = self.pho_embeddings(pho_idx)
        pho_lens = pho_lens.cpu().numpy().tolist()
        try: 
            pho_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            input=pho_embeddings,
            lengths=pho_lens,
            batch_first=True,
            enforce_sorted=False,
        ) 
        except Exception as e:
            print('phoneme-encoder:')
            print(e)
        
        _, pho_hiddens = self.pho_gru(pho_embeddings)
        
        try:
            pho_hiddens = pho_hiddens.squeeze(0).reshape(input_shape[0], input_shape[1], -1).contiguous()
        except Exception as e:
            print(e)
        pho_hiddens = self.pho_model(inputs_embeds=pho_hiddens,)[0]

        return pho_hiddens

class pinyin_encoder(BertPreTrainedModel):
    def __init__(self, config):
        super(pinyin_encoder, self).__init__(config)
        self.pho_embeddings = nn.Embedding(pinyin_convertor.get_pho_size(), config.hidden_size, padding_idx=0)
        self.pho_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        pho_config = deepcopy(config)
        pho_config.num_hidden_layers = 4
        self.pho_model = BertModel(pho_config)



    def forward(self, pho_idx, pho_lens, input_ids):
        input_shape = input_ids.size() 
        pho_embeddings = self.pho_embeddings(pho_idx)
        pho_lens = pho_lens.cpu().numpy().tolist()
        pho_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            input=pho_embeddings,
            lengths=pho_lens,
            batch_first=True,
            enforce_sorted=False,
        )
        _, pho_hiddens = self.pho_gru(pho_embeddings)
        pho_hiddens = pho_hiddens.squeeze(0).reshape(input_shape[0], input_shape[1], -1).contiguous()
        pho_hiddens = self.pho_model(inputs_embeds=pho_hiddens,)[0]

        return pho_hiddens



class Pinyin2(object):
    def __init__(self):
        super(Pinyin2, self).__init__()
        pho_vocab = ['P']
        pho_vocab += [chr(x) for x in range(ord('1'), ord('5') + 1)]
        pho_vocab += [chr(x) for x in range(ord('a'), ord('z') + 1)]
        pho_vocab += ['U']
        assert len(pho_vocab) == 33
        self.pho_vocab_size = len(pho_vocab)
        self.pho_vocab = {c: idx for idx, c in enumerate(pho_vocab)}

    def get_pho_size(self):
        return self.pho_vocab_size

    @staticmethod
    def get_pinyin(c):
        if len(c) > 1:
            return 'U'
        s = pypinyin.pinyin(
            c,
            style=pypinyin.Style.TONE3,
            neutral_tone_with_five=True,
            errors=lambda x: ['U' for _ in x],
        )[0][0]
        if s == 'U':
            return s
        assert isinstance(s, str)
        assert s[-1] in '12345'
        s = s[-1] + s[:-1]
        return s

    def convert(self, chars): 
        pinyins = list(map(self.get_pinyin, chars)) # 对每个字进行 pinyin化 ，如果是非char（pad等），返回U。如果是一个字，返回pinyin。
        pinyin_ids = [list(map(self.pho_vocab.get, pinyin)) for pinyin in pinyins]# 把 拼音字母转化成，ids
        pinyin_lens = [len(pinyin) for pinyin in pinyins] # 每个char 的拼音的长度
        try:
            pinyin_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x) for x in pinyin_ids],
                batch_first=True,
                padding_value=0,
            )
        except Exception as e:
            # print('phoneme-encoder:')
            print(e)
        '''
            Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])'''
        return pinyin_ids, pinyin_lens
pinyin_convertor = Pinyin2()

class Phoneme2(object):
    def __init__(self):
        super(Phoneme2, self).__init__()
        pho_vocab = [' '] # 英文中的word 可能有多个 phoneme 用 空格 进行表示。
        pho_vocab += [chr(x) for x in range(ord('0'), ord('4') + 1)]
        pho_vocab += [chr(x) for x in range(ord('A'), ord('Z') + 1)]
        pho_vocab += ['\'']
        pho_vocab += ['.']
        pho_vocab += ['#'] # 表示为空
        pho_vocab += ['^'] # 表示为word内的phoneme的空
        pho_vocab += ['*'] # <s> 表示开始
        pho_vocab += ['**'] # <s> 表示结束
        assert len(pho_vocab) == 38
        self.pho_vocab_size = len(pho_vocab)
        self.pho_vocab = {c: idx for idx, c in enumerate(pho_vocab)}

    def get_pho_size(self):
        return self.pho_vocab_size

    @staticmethod
    def get_my_phoneme(c):
        if c == '<s>':
            return '*'
        if c == '</s>':
            return '**'
        if c == '<pad>':
            return '#'
        s = g2p(c)
        str_split = '^' #一个单词内的几部分音素内容，用空格分割开来。 
        s = str_split.join(s)
        assert isinstance(s, str)
        return s
    # def  split_chars_string_2_words(chars: str):
    #     char_list = chars.split(' ')
    #     chars_output = []
    #     for item in char_list:
    #         if 's>' in item:
    #             item_tmp = item.split('s>')
    #             item_tmp[0] = item_tmp[0] + 's>'
    #             chars_output = chars_output + item_tmp
    #         else:
    #             chars_output.append(item)



    def convert(self, chars): 
        # chars = split_chars_string_2_words(chars)# 把chars中的每个字符分开来
        pinyins = list(map(self.get_my_phoneme, chars)) # 对每个字进行 pinyin化 ，如果是非char（pad等），返回U。如果是一个字，返回pinyin。
        pinyin_ids = [list(map(self.pho_vocab.get, pinyin)) for pinyin in pinyins]# 把 拼音字母转化成，ids
        pinyin_lens = [len(pinyin) for pinyin in pinyin_ids] # 每个char 的拼音的长度
        
        try: 
            pinyin_list = [torch.tensor(x) for x in pinyin_ids]  
        except Exception as e:
            print(e)
        
        try: 
            pinyin_ids_pad = torch.nn.utils.rnn.pad_sequence(
                pinyin_list,
                batch_first=True,
                padding_value=0,
            )
        except Exception as e:
            print(e)
        '''
            Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])'''
        return pinyin_ids_pad, pinyin_lens
phoneme_convertor = Phoneme2()