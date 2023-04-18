import numpy as np
import pandas as pd
import os

modality = {'01':'full_av','02':'video_only','03':'audio_only'}
vocal_channel = {'01':'speech','02':'song'}
emotion = {'01':'neutral','02':'calm','03':'happy','04':'sad','05':'angry','06':'fearful','07':'disgust','08':'surprised'}
emotional_intensity = {'01':'normal','02':'strong'}
statement = {'01':'Kids are talking by the door','02':'Dogs are sitting by the door'}
reptition = {'01':'first_repitition','02':'second_repetition'}
def actor_f(num):
    if int(num)%2==0: return('female')
    else: return('male')

actors = sorted(os.listdir('../input/ravdess-emotional-speech-audio'))
actors.pop()
actors

audio_file_dict = {}
for actor in actors:
    actor_dir = os.path.join('../input/ravdess-emotional-speech-audio',actor)
    actor_files = os.listdir(actor_dir)
    actor_dict = [i.replace(".wav","").split("-") for i in actor_files]
    dict_entry = {os.path.join(actor_dir,i):j for i,j in zip(actor_files,actor_dict)}
    audio_file_dict.update(dict_entry)

audio_file_dict = pd.DataFrame(audio_file_dict).T
audio_file_dict.columns = ['modality','vocal_channel','emotion','emotional_intensity','statement','repetition','actor']
audio_file_dict

audio_file_dict.modality = audio_file_dict.modality.map(modality)
audio_file_dict.vocal_channel = audio_file_dict.vocal_channel.map(vocal_channel)
audio_file_dict.emotion = audio_file_dict.emotion.map(emotion)
audio_file_dict.emotional_intensity = audio_file_dict.emotional_intensity.map(emotional_intensity)
audio_file_dict.statement = audio_file_dict.statement.map(statement)
audio_file_dict.repetition = audio_file_dict.repetition.map(reptition)
audio_file_dict['actor_sex'] = audio_file_dict.actor.apply(actor_f)

audio_file_dict

import matplotlib.pyplot as plt
import seaborn as sns

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

import torchaudio

sample1, sample_rate1 = torchaudio.load('../input/ravdess-emotional-speech-audio/Actor_01/03-01-01-01-01-01-01.wav')
sample1, sample_rate1

sample1, sample_rate1 = torchaudio.load('../input/ravdess-emotional-speech-audio/Actor_01/03-01-01-01-01-01-01.wav')
sample1, sample_rate1

sample1.shape

sample2.shape

import torch

torch.mean(sample1), torch.std(sample1), torch.min(sample1), torch.max(sample1)

torch.mean(sample2), torch.std(sample2), torch.min(sample2), torch.max(sample2)

plt.plot(sample1.t().numpy())

plt.plot(sample2.t().numpy())

audio_files = []
for i in list(audio_file_dict.index):
    i, _ = torchaudio.load(i)
    audio_files.append(i)

maxlen = 0
minlen = np.Inf
for i in audio_files:
    if i.shape[1]>maxlen:
        maxlen = i.shape[1]
    if i.shape[1]<minlen:
        minlen = i.shape[1]

minlen, maxlen

specgram = torchaudio.transforms.Spectrogram()(sample1)

print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray')

spectrograms = []
for i in audio_files:
    specgram = torchaudio.transforms.Spectrogram()(i)
    spectrograms.append(specgram)

spectrograms[0].shape,spectrograms[1].shape,spectrograms[2].shape,

max_width, max_height = max([i.shape[2] for i in spectrograms]), max([i.shape[1] for i in spectrograms])

import torch.nn.functional as F

image_batch = [
    # The needed padding is the difference between the
    # max width/height and the image's actual width/height.
    F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)])
    for img in spectrograms
]

image_batch[0].shape, image_batch[1].shape, image_batch[2].shape,

plt.imshow(image_batch[0][0].log2())

image_batch = torch.cat(image_batch,0)

del audio_files, spectrograms


class EmotionDataset(Dataset):
    def __init__(self, audio_file_dict):
        self.audio_fie_dict = audio_file_dict

    def __getitem__(self, index):
        img = list(audio_file_dict.index)[index]
        img, _ = torchaudio.load(img)
        img = torch.mean(img, dim=0).unsqueeze(0)
        img = torchaudio.transforms.Spectrogram()(img)
        img = F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)])

        label = pd.get_dummies(audio_file_dict.emotion)[index]
        label = np.array(label)
        label = torch.from_numpy(label)
        return (img, label)

    def __len__(self):
        count = len(audio_file_dict)
        return count

train_data = EmotionDataset(audio_file_dict=X_train)
test_data = EmotionDataset(audio_file_dict=X_test)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(242688, 1000)
        self.fc2 = nn.Linear(1000, 8)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = ConvNet()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.cuda()

# Train the model
total_step = len(train_loader)

for epoch in range(num_epochs):
    loss_list = []
    acc_list = []
    for i, (images, labels) in enumerate(train_loader):
        # Run the forward pass
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)
    print(f'epoch: {epoch}: acc:',np.mean(acc_list),'loss: ',np.mean(loss_list))

model.eval()

preds = []
outcome = []
labs = []
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        labs.append(labels)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        preds.append(predicted)
        c = (predicted == labels).squeeze()
        outcome.append(c)

outcome = torch.stack(outcome).view(-1).cpu().numpy()

print('Accuracy on test set after 50 epochs: ',100*round(outcome.sum()/len(outcome),2),'%')

