import os
import time
import csv

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloader import generate_test_train_dataloader
from GAN import Discriminator


class DiscriminatorTraining:

    def __init__(self, model, image_dir, batch_size, num_workers, lr=5e-4):
        self.phases = ['train', 'val', 'test']

        # set up device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        cudnn.benchmark = True

        if self.device == 'cuda:0':
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        self.batch_size = batch_size

        # distribute model on multiple gpus
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)

        self.net = model

        self.net.to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, verbose=True)

        datasets = generate_test_train_dataloader(image_dir, batch_size, num_workers)

        self.dataloader = {phase: data for phase, data in zip(self.phases, datasets)}

        self.stats = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        self.best_loss = float('inf')

        print('Initialized')

    def train(self, epochs, save_path):
        model_save_path = os.path.join(save_path, 'initial_discriminator_model_trained.pth')
        csv_save_path = os.path.join(save_path, 'initial_discriminator_model_trained_stats.csv')
        for epoch in range(epochs):
            start = time.strftime("%H:%M:%S")
            print('Epoch {}/{} | Start Time: {}'.format(epoch, epochs - 1, start))
            print('-' * 10)
            for phase in self.phases[:2]:
                print('Starting phase: %s' % phase)
                print('-' * 10)
                self.run_epoch(phase)

                with open(csv_save_path, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.stats.keys())
                    writer.writerows(zip(*self.stats.values()))

            self.scheduler.step(self.stats['val_loss'][-1])

            # save only the best model
            if self.stats['val_loss'][-1] < self.best_loss:
                state = {
                    "epoch": epoch,
                    "best_loss": self.best_loss,
                    "state_dict": self.net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }

                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = self.stats['val_loss'][-1]
                torch.save(state, model_save_path)

        self.run_epoch('test')

    def run_epoch(self, phase):
        if phase == 'train':
            self.net.train(True)  # Set trainind mode = true
            dataloader = self.dataloader[phase]
        else:
            self.net.train(False)  # Set model to evaluate mode
            dataloader = self.dataloader[phase]

        running_loss = 0.0
        running_acc = 0.0

        step = 0
        total = 0

        # iterate over data
        for i, batch in enumerate(dataloader):
            images, labels = self.generate_batch(batch)
            img = images.float().to(self.device)
            label = labels.to(self.device)
            step += 1
            total += img.size()[0]

            # forward pass
            if phase == 'train':
                # zero the gradients
                self.optimizer.zero_grad()
                outputs = self.net(img)
                loss = self.criterion(outputs, label)

                # the backward pass frees the graph memory, so there is no
                # need for torch.no_grad in this training pass
                loss.backward()
                self.optimizer.step()

            else:
                with torch.no_grad():
                    outputs = self.net(img)
                    loss = self.criterion(outputs, label)

            acc = (torch.max(outputs.data, dim=1).indices == label).sum()

            running_acc += acc.item() * dataloader.batch_size * 2
            running_loss += loss.item() * dataloader.batch_size * 2

            if step % 100 == 0:
                print('Current step: {}  Loss: {}  Acc: {}'.format(step, loss, acc))

        epoch_loss = running_loss / total
        epoch_acc = running_acc / total

        print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))

        if phase == 'train':
            self.stats['train_loss'].append(epoch_loss)
            self.stats['train_acc'].append(epoch_acc)
        elif phase == 'val':
            self.stats['val_loss'].append(epoch_loss)
            self.stats['val_acc'].append(epoch_acc)

        return epoch_loss, epoch_acc

    @staticmethod
    def generate_batch(batch):
        orig_images = batch['original_image']
        edit_images = batch['edited_image']
        labels = torch.LongTensor([0] * orig_images.size()[0] + [1] * edit_images.size()[0])

        images = torch.cat((orig_images, edit_images), dim=0)

        # shuffle data
        shuff = torch.randperm(images.size()[0])
        images = images[shuff]
        labels = labels[shuff]

        return images, labels


if __name__ == '__main__':
    img_dir = r'D:\fivek_dataset'

    if torch.cuda.is_available():
        detected_gpus = torch.cuda.device_count()
        batch_size = 25 * detected_gpus
    else:
        batch_size = 10

    num_workers = 4
    epochs = 1

    save_path = './'

    model = Discriminator()
    trainer = DiscriminatorTraining(model, img_dir, batch_size, num_workers)
    trainer.train(epochs, save_path)
