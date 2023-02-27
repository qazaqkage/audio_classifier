import os
import argparse

import numpy as np

from tqdm import tqdm

import librosa


DST_SPEC = 'spectrograms'
DST_MELSPECS = 'melspectrograms'
DST_MFCC = 'mfcc'


def extract_spectrogram(y, sr=8000, n_fft=None) -> np.array:
	'''
	y = time series audio
	sr = sample rate (8000 by default)
	
	returns: np.array of spectrogram
	'''
	if n_fft:
		stft = librosa.stft(y, n_fft=n_fft)
	else:
		stft = librosa.stft(y)
	spectrogram = np.abs(stft)**2
	return spectrogram

def extract_melspectrogram(y, sr=8000, n_fft=2048, hop_length=512, win_length=None) -> np.array:
	'''
	y = time series audio
	sr = sample rate (8000 by default)
	TODO: define other parameters
	
	returns: np.array of melspectrogram
	'''
	melspectrogram = librosa.feature.melspectrogram(y, sr=sr)
	return melspectrogram

def extract_mfcc(y, sr=8000, n_mfcc=20):
	'''
	y = time series audio
	sr = sample rate (8000 by default)
	n_mfcc = numner of MFCC
	
	returns: np.array of mfcc
	'''
	mfcc = librosa.feature.mfcc(y=y, sr=sr)
	return mfcc

def cut_if_necessary(y, size=8000):
	'''
	cuts audios with duration over size
	y = time series audio
	size = duration of audio that we need (i.g. if sr=8000Hz then 1sec=8000, 0.5=4000)
	returns: np.array of y
	'''
	if y.shape[0] > size:
		y = y[:size]

	return y

def pad_if_necessary(y, size=8000):
	'''
	pads audios with duration less than size with zeros
	y = time series audio
	size = duration of audio that we need (i.g. if sr=8000Hz then 1sec=8000, 0.5=4000)
	returns: np.array of y
	'''
	if y.shape[0] < size:
		diff = size - y.shape[0]
		zeros = np.zeros((diff))
		y = np.concatenate([y, zeros])

	return y

def save_numpy(root_path, fname, arr):
	dst_path = os.path.join(root_path, fname)
	np.save(dst_path, arr)

def make_dirs(dst, list_dirs):
	os.makedirs(dst, exist_ok=True)
	for d in list_dirs:
		print(f'CREATED FOLDER: {d}')
		os.makedirs(os.path.join(dst, d), exist_ok=True)


def main(files, root, dst):
	make_dirs(dst, [DST_SPEC, DST_MELSPECS, DST_MFCC])

	for f in tqdm(files):
		fpath = os.path.join(root, f)
		fname = os.path.splitext(f)[0]
		
		y, sr = librosa.load(fpath, sr=None)
		
		y = cut_if_necessary(y)
		
		spec = extract_spectrogram(y)
		melpec = extract_melspectrogram(y)
		mfcc = extract_mfcc(y)
		
		save_numpy(os.path.join(dst, DST_SPEC), fname=fname, arr=spec)
		save_numpy(os.path.join(dst, DST_MELSPECS), fname=fname, arr=melpec)
		save_numpy(os.path.join(dst, DST_MFCC), fname=fname, arr=mfcc)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Prepare audio features')
	parser.add_argument("src_folder", type=str, help="[input] to a folder with raw wav files")
	parser.add_argument("dst_folder", type=str, help="[output]  to a folder to store features")
	args = parser.parse_args()

	files = os.listdir(args.src_folder)

	main(files, args.src_folder, args.dst_folder)