import os
import time
import csv

import argparse
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloader import generate_test_train_dataloader
from GAN import Discriminator


class DiscriminatorTraining:

    def __init__(self, model, image_dir, batch_size, num_workers, lr=5e-4, no_cuda=False,number_images = 5000):
        self.phases = ['train', 'val', 'test']

        # set up device
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and not no_cuda) else "cpu")
        print(self.device)
        cudnn.benchmark = True
        
        if str(self.device) == 'cuda:0':
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
        #self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, verbose=True)

        datasets = generate_test_train_dataloader(image_dir, batch_size, num_workers,number_images=number_images)

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

            #self.scheduler.step(self.stats['val_loss'][-1])

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

            running_acc += acc.item() #* dataloader.batch_size * 2
            running_loss += loss.item()#* dataloader.batch_size * 2

            if step % 100 == 0:
                print('Current step: {}  Loss: {}  Acc: {}'.format(step, loss/labels.size()[0], acc/labels.size()[0]))

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
    parser = argparse.ArgumentParser()
    # The directory the images are located in.
    parser.add_argument('--img_dir', type=str, default=os.getcwd())
    # Number of image pairs per batch .
    parser.add_argument('--batch_size',type=int, default = None)
    # Number of workers for retrieving images from the dataset.
    parser.add_argument('--num_workers',type=int,default = 2)
    # Number of epochs to train for.
    parser.add_argument('--epochs',type=int,default = 10000)
    # Flag to disable CUDA
    parser.add_argument('--no_cuda',type=bool,default = False)
    # Number of images to use in train/test/val total.
    parser.add_argument('--number_images',type=int,default = 5000)
    # Flag to plot image examples each epoch.
    parser.add_argument('--plot',type=bool,default = True)
    # Path so save generative model. 
    parser.add_argument('--save_path',type = str, default = './')
    # Load in a previous descriminator
    parser.add_argument('--load_prev',type = bool, default = False)
    # The path to the discriminator model.
    parser.add_argument('--pretrained_discriminator_path', type=str, default='initial_discriminator_model_trained.pth')
    args = parser.parse_args()
    


    # If the batch size is not specified, assign defaults depending on number of GPUs. 
    if args.batch_size == None:
        if torch.cuda.is_available():
            detected_gpus = torch.cuda.device_count()
            batch_size = 25 * detected_gpus
            try:
                torch.multiprocessing.set_start_method('spawn')
            except:
                pass
        else:
            batch_size = 10
    else:
        batch_size = args.batch_size

    model = Discriminator()
    if args.load_prev:
        state = torch.load(args.pretrained_discriminator_path, map_location=lambda storage, loc: storage)
        state_dict = {k.replace('module.',''): v for k, v in state["state_dict"].items()}
        model.load_state_dict(state_dict)
        print('Loaded previous best model')

    # 
    trainer = DiscriminatorTraining(model, args.img_dir, batch_size, args.num_workers, no_cuda = args.no_cuda,number_images = args.number_images)
    trainer.train(args.epochs, args.save_path)
