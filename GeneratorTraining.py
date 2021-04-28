import os
import time
import csv

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloader import generate_test_train_dataloader
from GAN import Discriminator, Generator, Loss


class GeneratorTraining:

    def __init__(self, generator_model, discriminator_model, image_dir, batch_size, num_workers, lr=5e-4,no_cuda = False, number_images = 5000):
        self.phases = ['train', 'val', 'test']

        # set up device
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and not no_cuda) else "cpu")
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
            generator_model = torch.nn.DataParallel(generator_model)
            discriminator_model = torch.nn.DataParallel(discriminator_model)

        self.g_net = generator_model
        self.g_net.to(self.device)

        self.d_net = discriminator_model
        self.d_net.to(self.device)

        self.gen_loss = Loss()
        self.criterion = torch.nn.CrossEntropyLoss()

        self.g_optimizer = optim.Adam(self.g_net.parameters(), lr=lr)
        self.g_scheduler = ReduceLROnPlateau(self.g_optimizer, mode='min', patience=3, verbose=True)

        self.d_optimizer = optim.Adam(self.d_net.parameters(), lr=lr)
        self.d_scheduler = ReduceLROnPlateau(self.d_optimizer, mode='min', patience=3, verbose=True)

        datasets = generate_test_train_dataloader(image_dir, batch_size, num_workers,number_images=number_images)

        self.dataloader = {phase: data for phase, data in zip(self.phases, datasets)}

        self.stats = {'gen_train_loss': [], 'gen_val_loss': [], 'dis_train_acc': [], 'dis_val_acc': [], 'dis_train_loss': [], 'dis_val_loss': []}

        self.g_best_loss = float('inf')
        self.d_best_loss = float('inf')

        print('Initialized')

    def train(self, epochs, save_path):
        model_save_path = os.path.join(save_path, 'gan_model_trained.pth')
        csv_save_path = os.path.join(save_path, 'gan_model_trained_stats.csv')
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

            self.g_scheduler.step(self.stats['gen_val_loss'][-1])
            self.d_scheduler.step(self.stats['dis_val_loss'][-1])

            # save only the best model
            if self.stats['gen_val_loss'][-1] < self.g_best_loss:
                state = {
                    "epoch": epoch,
                    "best_gen_loss": self.g_best_loss,
                    "generator_state_dict": self.g_net.state_dict(),
                    "generator_optimizer": self.g_optimizer.state_dict(),
                    "discriminator_state_dict": self.d_net.state_dict(),
                    "discriminator_optimizer": self.d_optimizer.state_dict(),
                }

                print("******** New optimal found, saving state ********")
                state["best_gen_loss"] = self.gen_best_loss = self.stats['gen_val_loss'][-1]
                torch.save(state, model_save_path)

        self.run_epoch('test')

    def run_epoch(self, phase):
        if phase == 'train':
            self.g_net.train(True)  # Set trainind mode = true
            self.d_net.train(True)
            dataloader = self.dataloader[phase]
        else:
            self.g_net.train(False)  # Set model to evaluate mode
            self.d_net.train(False)
            dataloader = self.dataloader[phase]

        g_running_loss = 0.0
        d_running_loss = 0.0
        d_running_acc = 0.0

        step = 0
        total = 0

        # iterate over data
        for i, batch in enumerate(dataloader):
            #images, labels = self.generate_batch(batch)
            orig_img = batch['original_image'].float().to(self.device)
            edit_img = batch['edited_image'].float().to(self.device)
            # label = labels.to(self.device)
            step += 1
            total += orig_img.size()[0]

            # forward pass
            if phase == 'train':
                # zero the gradients
                self.g_optimizer.zero_grad()
                self.d_optimizer.zero_grad()
                enhanced_img = self.g_net(orig_img)

                images, labels = self.generate_batch((orig_img, enhanced_img))

                # might need to feed discriminator grayscale images to focus on image texture
                outputs = self.d_net(images)

                dis_loss = self.criterion(outputs, labels)
                gen_loss = self.gen_loss.total_loss(enhanced_img, orig_img, dis_loss)

                # the backward pass frees the graph memory, so there is no
                # need for torch.no_grad in this training pass
                dis_loss.backward(retain_graph = True)
                gen_loss.backward()
                self.g_optimizer.step()
                self.d_optimizer.step()

            else:
                with torch.no_grad():
                    enhanced_img = self.g_net(orig_img)

                    images, labels = self.generate_batch((orig_img, enhanced_img))

                    # might need to feed discriminator grayscale images to focus on image texture
                    outputs = self.d_net(images)

                    dis_loss = self.criterion(outputs, labels)
                    gen_loss = self.gen_loss.total_loss(enhanced_img, orig_img, dis_loss)

            dis_acc = (torch.max(outputs.data, dim=1).indices == labels).sum()/orig_img.size()[0]

            d_running_acc += dis_acc.item()# * dataloader.batch_size * 2
            d_running_loss += dis_loss.item()# * dataloader.batch_size * 2

            g_running_loss += gen_loss.item()# * dataloader.batch_size

            if step % 100 == 0:
                print('Current step: {}  Generator Loss: {}  Discriminator Loss: {} Discriminator Acc: {}'.format(step, gen_loss, dis_loss, dis_acc))

        g_epoch_loss = g_running_loss / total
        d_epoch_loss = d_running_loss / (total * 2)
        d_epoch_acc = d_running_acc / (total * 2)

        print('Epoch Results for {} - Generator Loss: {}  Discriminator Loss: {} Discriminator Acc: {}'.format(phase, g_epoch_loss, d_epoch_loss, d_epoch_acc))

        if phase == 'train':
            self.stats['gen_train_loss'].append(g_epoch_loss)
            self.stats['dis_train_loss'].append(d_epoch_loss)
            self.stats['dis_train_acc'].append(d_epoch_acc)
        elif phase == 'val':
            self.stats['gen_val_loss'].append(g_epoch_loss)
            self.stats['dis_val_loss'].append(d_epoch_loss)
            self.stats['dis_val_acc'].append(d_epoch_acc)

        return g_epoch_loss, d_epoch_loss, d_epoch_acc

    @staticmethod
    def generate_batch(batch):
        #try:
        #    orig_images = batch['original_image']
        #    edit_images = batch['edited_image']
        #except:
        orig_images = batch[0]
        edit_images = batch[1]
        labels = torch.LongTensor([0] * orig_images.size()[0] + [1] * edit_images.size()[0])

        images = torch.cat((orig_images, edit_images), dim=0)

        # shuffle data
        shuff = torch.randperm(images.size()[0])
        images_ = images[shuff]
        labels_ = labels[shuff]

        return images_, labels_


if __name__ == '__main__':
    img_dir = os.getcwd()

    # path to pretrained discriminator model
    pretrained_discriminator_path = 'initial_discriminator_model_trained.pth'

    if torch.cuda.is_available():
        detected_gpus = torch.cuda.device_count()
        batch_size = 25 * detected_gpus
    else:
        batch_size = 10

    num_workers = 4
    #batch_size = 2
    epochs = 100

    no_cuda = True
    number_images = (0,5000)
    save_path = './'

    discriminator_model = Discriminator()

    state = torch.load(pretrained_discriminator_path, map_location=lambda storage, loc: storage)
    # might need to do this
    # state_dict = {k[7:]: v for k, v in state["state_dict"].items()}
    discriminator_model.load_state_dict(state["state_dict"])

    generator_model = Generator()

    trainer = GeneratorTraining(generator_model, discriminator_model, img_dir, batch_size, num_workers,no_cuda = no_cuda, number_images =number_images)
    trainer.train(epochs, save_path)
