import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler

import argparse

import matplotlib.pyplot as plt

import vae
import models
from train_eeg_data import EEGDataset


class Importer(object):
    """docstring for Importer"""
    def __init__(self, parser):
        self.args = parser.parse_args()

    def import_model(self, model_path, data_path):
        data_loader = self.import_data(data_path)
        data_iter = iter(data_loader)
        x = data_iter.next().to('cpu').numpy()
        data_length = x.shape[1]
        n_channel = x.shape[2]
        input_size = n_channel*data_length
        latent_size = self.args.latent_size
        encoder = models.FullyConnecteEncoder(input_size, latent_size)
        decoder = models.FullyConnecteDecoder(latent_size, input_size)
        if self.args.device == None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = args.device
        model = vae.VariationalAutoEncoder(encoder, decoder, 
                                           path_to_model=self.args.path_to_model,
                                           device=device, 
                                           n_epoch=self.args.num_epoch,
                                           beta_steps=10, 
                                           beta_min=self.args.beta_min, 
                                           beta_max=self.args.beta_max,
                                           snapshot=self.args.snapshot, 
                                           lr=0.001)
        model = model.to(self.args.device)
        model.load_model(model_path)
        #sd_model = torch.load(file_path, map_location="cpu")
        #model.load_state_dict(sd_model)
        return model, data_loader

    def import_data(self, data_path):
        data = EEGDataset(data_path, device="cpu")
        data_loader = DataLoader(data, batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        return data_loader




class Evaluator(object):

    def __init__(self, model, data_loader):
        super(Evaluator, self).__init__()
        self.model = model
        self.data_loader = data_loader


    def input_reconstruction(self):
        for x in data_loader:
            x_recon = self.model.forward(x)[0]
            x = x.detach().numpy()
            x_recon = x_recon.detach().numpy()

            for subject in range(x.shape[0]):
                colors = []
                sub_x = x[subject]
                sub_x_recon = x_recon[subject]
                for channel in range(sub_x.shape[0]):
                    sub_x[channel] += 100*channel
                    sub_x_recon[channel] += 100*channel
                    plt.plot(sub_x[channel], color="C0")
                    plt.plot(sub_x_recon[channel], color="C1", linestyle="dashed")
                plt.show()


    def get_latents(self, input_data):
        mu, logstd = self.model.encoder(input_data)
        var = torch.exp(logstd)
        eps = torch.randn_like(var).to(self.device)
        latent_data = eps.mul(var).add(mu)
        return latent_data


    def latent_correlations(self):
        input_data = self.data_loader[0]
        latent_data = self.get_latents(input_data)
        print(latent_data.shape)

        # visualize latents
        #visualize(latent_data)


    def subject_representations(self, labels):
        input_data = self.data_loader[0]
        latent_data = self.get_latents(input_data)
        

        # train and visualize t-SNE
        #tsne(latent_data, labels)




parser = argparse.ArgumentParser(description='VAE training using EEG dataset')
parser.add_argument('--train', default=False,action='store_true', help='set it to train the model')
parser.add_argument('--eval', default=False, action='store_true', help='set it  to evaluate the model')
parser.add_argument('--path-to-model', default="./", type=str, help='the path to save/load the model')
parser.add_argument('--path-to-data', default=None, type=str, help='the path to load the data')
parser.add_argument('--model-filename', default=None, type=str, help='the filename of the model to be evaluated')
parser.add_argument('--latent-size', default=5, type=int, help='the dimension of the latent variable')
parser.add_argument('--num-epoch', default=10000, type=int, help='the number of epochs to train the model')
parser.add_argument('--snapshot', default=100, type=int, help='the number of epochs to make a snapshot of the training')
parser.add_argument('--batch-size', default=100, type=int, help='the batch size')
parser.add_argument('--beta-min', default=0.0, type=float, help='the starting value for beta')
parser.add_argument('--beta-max', default=0.01, type=float, help='the final value for beta')
parser.add_argument('--device', default=None, type=str, help='the device for training, cpu or cuda')

model_path = "vae_99999.mdl"
data_path = "./data/snippets.npy"
importer = Importer(parser)
model, data_loader = importer.import_model(model_path, data_path)

#data = importer.import_data(data_path)[:1000]
#print("Imported model and data")
#print(data.shape)

evaluator = Evaluator(model, data_loader)
#evaluator.input_reconstruction()
evaluator.input_reconstruction()
#evaluator.subject_representations()