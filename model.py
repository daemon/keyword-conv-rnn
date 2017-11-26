import hashlib
import random
import os

from torch.autograd import Variable
from tqdm import tqdm
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from audio import preprocess_audio

class SimpleCache(dict):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.n_keys = 0

    def __setitem__(self, key, value):
        if key in self.keys():
            super().__setitem__(key, value)
        elif self.n_keys < self.limit:
            self.n_keys += 1
            super().__setitem__(key, value)
        return value

class AIShellDataset(data.Dataset):
    def __init__(self, audio_data, config):
        self.audio_files, self.audio_pinyin = audio_data
        self.config = config
        self._audio_cache = SimpleCache(config["cache_size"])
        self._file_cache = SimpleCache(config["cache_size"])
        self.labels = self.config["labels"]

    def preprocess(self, example):
        if random.random() < 0.7:
            try:
                return self._audio_cache[example]
            except KeyError:
                pass

        file_data = self._file_cache.get(example)
        data = librosa.core.load(example, sr=16000)[0] if file_data is None else file_data
        self._file_cache[example] = data
        data = torch.from_numpy(preprocess_audio(data, 40))
        self._audio_cache[example] = data
        return data

    def compute_label(self, pinyin):
        label = torch.zeros(len(self.labels))
        for p in pinyin:
            try:
                idx = self.labels.index(p)
                label[idx] = 1
            except ValueError:
                continue
        return label

    def __getitem__(self, idx):
        data = self.preprocess(self.audio_files[idx])
        label = self.compute_label(self.audio_pinyin[idx])
        return data, label

    def __len__(self):
        return len(self.audio_files)

    @classmethod
    def _find_all_wavs(cls, directory):
        wavs = []
        for fname in os.listdir(directory):
            full_name = os.path.join(directory, fname)
            if os.path.isdir(full_name):
                wavs.extend(cls._find_all_wavs(full_name))
            elif full_name.endswith(".wav"):
                wavs.append(full_name)
        return wavs

    @classmethod
    def splits(cls, config):
        data_dir = config["data_dir"]
        train_pct = config["train_pct"]
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]
        pos_prob = config["positive_prob"]
        labels = set(config["labels"])
        transcript = config["transcript"]

        transcript_data = {}
        with open(transcript) as f:
            for line in f.readlines():
                ident, pinyin = line.split(" ", 1)
                transcript_data[ident] = set(pinyin.split())

        wavs = cls._find_all_wavs(data_dir)
        sets = [[[], []], [[], []], [[], []]]
        m = hashlib.md5()

        print("Computing train/dev/test splits...")
        for wav in wavs:
            h = int(hashlib.md5(wav.encode()).hexdigest(), 16)
            bucket = h % 100
            if bucket < train_pct:
                bucket = 0
            elif bucket < train_pct + dev_pct:
                bucket = 1
            else:
                bucket = 2

            ident = os.path.basename(wav).split(".")[0]
            if ident not in transcript_data:
                continue
            positive = len(labels - transcript_data[ident]) < len(labels)
            if (positive and random.random() < pos_prob) or (not positive and random.random() > pos_prob):
                sets[bucket][0].append(wav)
                sets[bucket][1].append(transcript_data[ident])
        return (cls(sets[0], config), cls(sets[1], config), cls(sets[2], config))

    @staticmethod
    def default_config():
        return dict(data_dir="data", dev_pct=10, train_pct=80, test_pct=10, positive_prob=0.8, 
            transcript="data/transcript.txt", labels=["ba", "dì"])

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

class ConvRNNModel(SerializableModule):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        n_fmaps = config["n_feature_maps"]
        n_labels = len(config["labels"])
        fc_size = config["fc_size"]

        self.bi_rnn = nn.GRU(40, self.hidden_size, 1, batch_first=True, bidirectional=True)
        self.conv = nn.Conv2d(1, n_fmaps, (1, self.hidden_size * 2))
        self.fc1 = nn.Linear(n_fmaps + 2 * self.hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, n_labels)

    @staticmethod
    def default_config():
        return dict(hidden_size=200, n_feature_maps=200, fc_size=200)

    def forward(self, x):
        h_0 = Variable(torch.zeros(2, x.size(0), self.hidden_size)).cuda()
        rnn_seq, rnn_out = self.bi_rnn(x, h_0)
        rnn_out.data = rnn_out.data.permute(1, 0, 2)
        x = self.conv(rnn_seq.unsqueeze(1)).squeeze(3)
        x = F.relu(x)
        x = F.max_pool1d(x, x.size(2))
        out = [t.squeeze(1) for t in rnn_out.chunk(2, 1)]
        out.append(x)
        x = torch.cat(out, 1).squeeze(2)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

