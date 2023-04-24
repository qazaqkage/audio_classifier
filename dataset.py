import os

import pandas as pd
import numpy as np

import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram, Resample
from torchaudio import load as load_audio
from analyse import audio_file_dict
import torch.nn.functional as F
from analyse import spec, max_h, max_w


class EmotionDataset(Dataset):
    def __init__(self, audio_file_dict):
        self.audio_fie_dict = audio_file_dict

    def __getitem__(self, index):
        img = list(audio_file_dict.index)[index]
        img, _ = torchaudio.load(img)
        img = torch.mean(img, dim=0).unsqueeze(0)
        img = torchaudio.transforms.Spectrogram()(img)
        img = F.pad(img, [0, max_w - img.size(2), 0, max_h - img.size(1)])

        label = pd.get_dummies(audio_file_dict.emotion)[index]
        label = np.array(label)
        label = torch.from_numpy(label)
        return (img, label)

    def __len__(self):
        count = len(audio_file_dict)
        return count



class MyCustomDataset(Dataset):
	"""a custom pytorch Dataset for audio classifier"""
	def __init__(self, audio_file_dict):
		self.audio_fie_dict = audio_file_dict

	def __len__(self):
		count = len(audio_file_dict)
		return count

	def __getitem__(self, index):
		img = list(audio_file_dict.index)[index]
		img, _ = torchaudio.load(img)
		img = torch.mean(img, dim=0).unsqueeze(0)
		img = torchaudio.transforms.Spectrogram()(img)
		img = F.pad(img, [0, max_w - img.size(2), 0, max_h - img.size(1)])

		def labeler(name):
			if name == 'male':
				return (1)
			else:
				return (0)

		label = list(audio_file_dict.actor_sex)[index]
		label = np.array(labeler(label))
		label = torch.from_numpy(label)
		return (img, label)


	def _cut_if_necessary(self, signal):
		if signal.shape[1] > self.num_samples:
			signal = signal[:, :self.num_samples]
		return signal


	def _right_pad_if_necessary(self, signal):
		length_signal = signal.shape[1]
		if length_signal < self.num_samples:
			num_missing_samples = self.num_samples - length_signal
			last_dim_padding = (0, num_missing_samples)
			signal = torch.nn.functional.pad(signal, last_dim_padding)
		return signal


	def _resample_if_necessary(self, signal, sr):
		if sr != self.target_sample_rate:
			resampler = Resample(sr, self.target_sample_rate)
			signal = resampler(signal)
		return signal


	def _mix_down_if_necessary(self, signal):
		if signal.shape[0] > 1:
			signal = torch.mean(signal, dim=0, keepdim=True)
		return signal

	train_data = EmotionDataset(audio_file_dict=X_train)
	test_data = EmotionDataset(audio_file_dict=X_test)