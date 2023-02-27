import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from glob import glob
import IPython.display as ipd


import librosa
from librosa import display as dp
from scipy.io import wavfile

import argparse
#--------------------------

dst_spec = 'spectrograms'
dst_melspecs = 'melspectrograms'
dst_mfcc = 'mfcc'
dst_rms = 'rms'
dst_chroma = 'chroma'
dst_tempogram = 'tempogram'

root = 'C:/Users/LEGION/Desktop/Jupyret/datasets/recordings/'
files = os.listdir(root)

ind = 0
f = os.path.join(root, files[ind])
print(f)


audio=glob('C:/Users/LEGION/Desktop/Jupyret/datasets/recordings/*')
print(ipd.Audio(audio[1623]))

print(f'{len(files)} FILES IN FOLDER')
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


def extract_rms(y) -> np.array:
	'''
    y = time series audio
    S = spectogram magnitude
    phase = position of a sound wave in time

    returns: array of rms
    '''
	S, phase = librosa.magphase(librosa.stft(y))
	rms = librosa.feature.rms(S=S)
	return rms


def extract_chromagram(y, sr=8000):
	'''
    y = time series audio
    sr = sample rate (8000 by default)

    returns: array of chromagram
    '''

	chroma = librosa.feature.chroma_stft(y=y, sr=sr)
	return chroma


def extract_tempogram(y, sr=8000, hop_length=512):
	'''
    y = time series audio
    sr = sample rate (8000 by default)
    hop_length = the length of the non-intersecting portion of window length (512 by default)

    returns: array of chromagram
    '''
	oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
	tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)

	return tempogram

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
	make_dirs(dst, [dst_rms, dst_mfcc, dst_spec, dst_chroma, dst_melspecs, dst_tempogram])

	for f in tqdm(files):
		fpath = os.path.join(root, f)
		fname = os.path.splitext(f)[0]
		
		y, sr = librosa.load(fpath, sr=None)
		
		y = cut_if_necessary(y)
		
		spec = extract_spectrogram(y)
		melpec = extract_melspectrogram(y)
		mfcc = extract_mfcc(y)
		rms = extract_rms(y)
		chroma = extract_chromagram(y)
		tempogram = extract_tempogram(y)

		save_numpy(os.path.join(dst, dst_spec), fname=fname, arr=spec)
		save_numpy(os.path.join(dst, dst_melspecs), fname=fname, arr=melpec)
		save_numpy(os.path.join(dst, dst_mfcc), fname=fname, arr=mfcc)
		save_numpy(os.path.join(dst, dst_rms), fname=fname, arr=rms)
		save_numpy(os.path.join(dst, dst_chroma), fname=fname, arr=chroma)
		save_numpy(os.path.join(dst, dst_tempogram), fname=fname, arr=tempogram)

		
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Prepare audio features')
	parser.add_argument("src_folder", type=str, help="[input] to a folder with raw wav files")
	parser.add_argument("dst_folder", type=str, help="[output]  to a folder to store features")
	args = parser.parse_args()

	files = os.listdir(args.src_folder)

	main(files, args.src_folder, args.dst_folder)