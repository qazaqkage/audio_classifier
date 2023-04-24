import numpy as np
import torch
import torch.nn as nn
from analyse import hyperparam, train_loader, test_loader

learning_rate = hyperparam[3]
num_epochs = hyperparam[0]
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

