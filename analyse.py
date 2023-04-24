import os
import pandas as pd
import matplotlib.pyplot as plt
import torchaudio
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from dataset import MyCustomDataset
from torch.utils.data import DataLoader

actors = sorted(os.listdir(r'C:\Users\LEGION\Desktop\Jupyret\input\ravdess-emotional-speech-audio'))
audio_file_dict = {}

def dict_update():
    for actor in actors:
        actor_dir = os.path.join(r'C:\Users\LEGION\Desktop\Jupyret\input\ravdess-emotional-speech-audio',actor)
        actor_files = os.listdir(actor_dir)
        actor_dict = [i.replace(".wav","").split("-") for i in actor_files]
        dict_entry = {os.path.join(actor_dir,i):j for i,j in zip(actor_files,actor_dict)}
        return audio_file_dict.update(dict_entry)

audio_file_dict = dict_update()
audio_file_dict = pd.DataFrame(audio_file_dict).T
audio_file_dict.columns = ['modality','vocal_channel','emotion','emotional_intensity','statement','repetition','actor']

def show_dict():
    print(audio_file_dict)


def actor_f(num):
    if int(num) % 2 == 0:
        return ('female')
    else:
        return ('male')

#-----------------Configs for dataset-----------------------
modality = {'01':'full_av','02':'video_only','03':'audio_only'}
vocal_channel = {'01':'speech','02':'song'}
emotion = {'01':'neutral','02':'calm','03':'happy','04':'sad','05':'angry','06':'fearful','07':'disgust','08':'surprised'}
emotional_intensity = {'01':'normal','02':'strong'}
statement = {'01':'Kids are talking by the door','02':'Dogs are sitting by the door'}
reptition = {'01':'first_repitition','02':'second_repetition'}

audio_file_dict.modality = audio_file_dict.modality.map(modality)
audio_file_dict.vocal_channel = audio_file_dict.vocal_channel.map(vocal_channel)
audio_file_dict.emotion = audio_file_dict.emotion.map(emotion)
audio_file_dict.emotional_intensity = audio_file_dict.emotional_intensity.map(emotional_intensity)
audio_file_dict.statement = audio_file_dict.statement.map(statement)
audio_file_dict.repetition = audio_file_dict.repetition.map(reptition)
audio_file_dict['actor_sex'] = audio_file_dict.actor.apply(actor_f)

def plotting():
    fig, (ax1,ax2) = plt.subplots(2, 2,figsize=(12,8))
    ax1[0].barh(y=audio_file_dict.emotion.value_counts().index,width=audio_file_dict.emotion.value_counts().values)
    ax1[0].set_title('Emotion')
    ax1[1].bar(x=audio_file_dict.actor_sex.value_counts().index,height=audio_file_dict.actor_sex.value_counts().values)
    ax1[1].set_title('Actor Sex')
    ax2[0].bar(x=audio_file_dict.emotional_intensity.value_counts().index,height=audio_file_dict.emotional_intensity.value_counts().values)
    ax2[0].set_title('Emotional Intensity')
    ax2[1].bar(x=audio_file_dict.statement.value_counts().index,height=audio_file_dict.statement.value_counts().values)
    plt.xticks(rotation=45)
    ax2[1].set_title('Statement')
    fig.tight_layout()

audio_files = []

def audio_load():
    for i in list(audio_file_dict.index):
        i, _ = torchaudio.load(i)
        audio_files.append(i)

sample1, sample_rate1 = torchaudio.load('../input/ravdess-emotional-speech-audio/Actor_01/03-01-01-01-01-01-01.wav')

def minmax():
    maxlen = 0
    minlen = np.Inf
    for i in audio_files:
        if i.shape[1] > maxlen:
            maxlen = i.shape[1]
        if i.shape[1] < minlen:
            minlen = i.shape[1]
def sample_Spec():
    specgram = torchaudio.transforms.Spectrogram()(sample1)
    print("Shape of spectrogram: {}".format(specgram.size()))
    plt.figure()
    plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray')

def spec_append():
    spectrograms = []
    for i in audio_files:
        specgram = torchaudio.transforms.Spectrogram()(i)
        spectrograms.append(specgram)
        print(spectrograms[0].shape, spectrograms[1].shape, spectrograms[2].shape)
    max_width, max_height = max([i.shape[2] for i in spectrograms]), max([i.shape[1] for i in spectrograms])
    return spectrograms, max_width, max_height

spec, max_w, max_h = spec_append()
def batch(max_width, max_height, spectrograms):
    image_batch = [
        # The needed padding is the difference between the
        # max width/height and the image's actual width/height.
        F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)])
        for img in spectrograms
    ]

def get_shape(image_batch):
    print(image_batch[0].shape, image_batch[1].shape, image_batch[2].shape)

def cleaning(audio_files, spectrograms, image_batch, y):
    del audio_files, spectrograms, image_batch, y

def splitting(audio_file_dict):
    X_train, X_test = train_test_split(audio_file_dict, test_size=0.3)
    train_data = MyCustomDataset(audio_file_dict=X_train)
    test_data = MyCustomDataset(audio_file_dict=X_test)
    return train_data, test_data

def set_hyperparametrs():
    num_epochs = 50
    num_classes = 2
    batch_size = 16
    learning_rate = 0.000001
    return num_epochs, num_classes, batch_size, learning_rate

hyperparam = set_hyperparametrs()
train_d, test_d = splitting(audio_file_dict)
train_loader = DataLoader(dataset=train_d, batch_size=hyperparam[2], shuffle=True)
test_loader = DataLoader(dataset=test_d, batch_size=hyperparam[2], shuffle=False)


