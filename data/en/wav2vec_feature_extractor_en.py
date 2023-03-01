
from abc import ABC
from typing import List
from torch.utils.data import Dataset, Subset, DataLoader
from dataclasses import dataclass
import h5py
import numpy as np


from tap import Tap

import torchaudio
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor,
)
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
import torch
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


@dataclass
class AudioInputExample: 
    """
    Input Example for a single example
    """
    id: str = ""
    file: str = ""
    text: str = ""


class DataProcessor(ABC):
    """Abstract Data Processor Class which handle the different corpus"""

    def get_train_dataset(self) -> Dataset:
        """get_train_dataset
        """
        raise NotImplementedError

    def get_test_dataset(self) -> Dataset:
        raise NotImplementedError

    def get_dev_dataset(self) -> Dataset:
        raise NotImplementedError

    def get_labels(self) -> List[str]:
        pass

    def get_train_labels(self) -> List[str]:
        return self.get_labels()




class AudioDataProcessor(DataProcessor):
    """AudioDataProcessor
    """

    def __init__(self, data_dir) -> None:
        super().__init__()
        self.data_dir = data_dir

    def _read(self, file: str) -> List[AudioInputExample]:
        examples = []
        examples = []
        import json
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            keys_list = list(data['utts'].keys())
            examples = []
            for key in keys_list:
                id = key
                file = data['utts'][id]['input']['path']
                text = data['utts'][id]['output']['text']
                example = AudioInputExample(id, file, text) 
                examples.append(example)

        return examples

    def _load_dataset(self,) -> Dataset:
        file = os.path.join(self.data_dir)
        examples = self._read(file)
        indices = [i for i in range(len(examples))] 
        return Subset(examples, indices) 

    def get_train_dataset(self) -> Dataset:
        return self._load_dataset()
    
    # def string2dict(self, item: str) -> List:
    #     return {"id":item.split('\"')[3], "file":item.split('\"')[7], "text": item.split('\"')[11]}
        
    # def rewrite_audio_path(sefl, file_path: str):
    #     file_path = file_path.split('LibriSpeech')
    #     file_path = os.path.join(config.audio_data_current_path+file_path[1])
    #     return file_path

class Config(Tap):
    # dataset_path: str = '/home/users/jiangjin/jiangjin_bupt/peking/huawei_asr/Model_data-all/DATA/audio_data/librispeech/'
    # audio_data_current_path: str = '/home/users/jiangjin/jiangjin_bupt/Python/ASR/wenet/examples/librispeech/s0/export/LibriSpeech/'
    device: str = 'cuda'
    # max_seq_length: int = 64
    # audio_max_length: int = 65000
    wav2vec_pretrained_model = '/home/data/jiangjin/TAP_ASR/pretrained-model/wav2vec_pretrained_model/en/'
    f_wav2vec_path= '' 
    audio_data_dir: str =  ''#'/home/jiangjin/ASR_CORRECTION/ASR/fairseq/examples/speech_recognition/datasets/preprocessed_data/train.json'
    batch_size = 48
    max_length: int = 130000
    
    
    
    
    def get_device(self):
        """return the device"""
        return torch.device(self.device)
  

class Extractor:
    def __init__(
        self, config: Config,
        audio_processor: DataProcessor,
        # text_tokenizer: PreTrainedTokenizer,
        audio_tokenizer: Wav2Vec2Processor,
        wav2vec_model: Wav2Vec2Model,
        f_wav2vec_path,
        
    ) -> None: 
        self.config = config
        self.audio_tokenizer = audio_tokenizer
        # self.wav2vec_model = wav2vec_model
        # self.text_tokenizer = text_tokenizer
        self.resampler = torchaudio.transforms.Resample()
        self.wav2vec_model = wav2vec_model.to(self.config.get_device())
        self.audio_train_dataloader = self.create_dataloader(
                dataset=audio_processor.get_train_dataset(),
                shuffle=False,
                collate_fn=self.convert_audio_examples_to_features,
        )
        self.f_wav2vec_path = f_wav2vec_path

    def create_dataloader(self, dataset: Dataset, collate_fn, shuffle) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle, # self.config.shuffle,
            collate_fn=collate_fn,
            # sampler=torch.utils.data.distributed.DistributedSampler(dataset)
        )

    def convert_audio_examples_to_features(self, audio_examples: List[AudioInputExample]):
        "load audio from disk"
        speechs = []
        for example in audio_examples:
            speech, _ = torchaudio.load(example.file)
            speech = self.resampler(speech).squeeze().numpy()
            speechs.append(speech)

        # audio_tokenizer 为 wav2vecprocessor 可以将语音转化为向量
        speechs = self.audio_tokenizer(
            raw_speech=speechs,
            sampling_rate=16_000,
            return_tensors="pt",
            padding='max_length',
            max_length=self.config.max_length,
            truncation=True,
        )
        speechs = speechs.to(self.config.get_device())

        with torch.no_grad():
            speechs = self.wav2vec_model(**speechs)
        # 这里感觉可以 输出 list(last_hidden_states.shape)的info
        speechs = speechs.last_hidden_state
        # texts = [example.text for example in audio_examples]
        # encoded_features = []
        # encoded_features = self.text_tokenizer.batch_encode_plus(
        #     texts,
        #     max_length=self.config.max_seq_length,
        #     padding='max_length',
        #     return_tensors='pt',
        #     return_attention_mask=True
        # )


        # if not os.path.exists(self.f_wav2vec_path):
        #     f_wav2vec = h5py.File(self.f_wav2vec_path, 'w')
        # else:
        #     print('__________start open H5file-_____________')
        #     f_wav2vec = h5py.File(self.f_wav2vec_path, 'a')
        #     print('__________end open H5file-_____________')
        # print('__________start save-_____________')
        # for i in range(len(speechs)):
        #     # print(i)
        #     print('__________start save loop-_____________')
        #     f_wav2vec.create_dataset(audio_examples[i].id, data=speechs[i].detach().cpu().numpy())
        #     print('__________end save loop-_____________')

        # print('__________end save-_____________')
        # print('__________start close H5file-_____________')
        # f_wav2vec.close()
        # print('__________start close H5file-_____________')        


        if not os.path.exists(self.f_wav2vec_path):
            f_wav2vec = h5py.File(self.f_wav2vec_path, 'w')
        else:
            f_wav2vec = h5py.File(self.f_wav2vec_path, 'a')
        for i in range(len(speechs)):
            # print(i)
            if self.config.local_rank=='0':
                f_wav2vec.create_dataset(audio_examples[i].id, data=speechs[i].detach().cpu().numpy())
        f_wav2vec.close()

        
        return  speechs, audio_examples

    def result_save(self,):
        print(f'Extractor begin...')
        # self.audio_tokenizer[0].to(self.config.get_device())
        # self.audio_tokenizer[1].to(self.config.get_device())
        # self.audio_tokenizer[1].cuda()

        self.train_bar = tqdm(total=len(self.audio_train_dataloader))      
        
        # f_wav2vec=h5py.File("wav2vec_feature.hdf5","w")
        # group_speech=self.f_wav2vec.create_group("speechs")
        # group_label=self.f_wav2vec.create_group("labels")
        speechs_list = []
        for audio_batch in self.audio_train_dataloader:


            self.train_bar.update()



if __name__ == "__main__":
    config: Config = Config().parse_args(known_only=True)

    
    extractor = Extractor(
        config,
        audio_processor=AudioDataProcessor(
            config.audio_data_dir),
        audio_tokenizer=Wav2Vec2Processor.from_pretrained(
            config.wav2vec_pretrained_model),
        wav2vec_model=Wav2Vec2Model.from_pretrained(config.wav2vec_pretrained_model),   
        f_wav2vec_path=
    )
    extractor.result_save()