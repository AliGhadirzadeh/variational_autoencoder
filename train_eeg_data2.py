import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import models
import vae

import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='VAE training using EEG dataset')
parser.add_argument('--train', default=False,action='store_true', help='set it to train the model')
parser.add_argument('--eval', default=False, action='store_true', help='set it  to evaluate the model')
parser.add_argument('--path-to-model', default=None, type=str, help='the path to save/load the model')
parser.add_argument('--path-to-data', default=None, type=str, help='the path to load the data')
parser.add_argument('--model-filename', default=None, type=str, help='the filename of the model to be evaluated')
parser.add_argument('--latent-size', default=5, type=int, help='the dimension of the latent variable')
parser.add_argument('--num-epoch', default=10000, type=int, help='the number of epochs to train the model')
parser.add_argument('--crossvalidation', default=False, type=bool, help='whether to crossvalidate')
parser.add_argument('--patience', default=3, type=int, help='patience parameter for crossvalidation')
parser.add_argument('--snapshot', default=100, type=int, help='the number of epochs to make a snapshot of the training')
parser.add_argument('--batch-size', default=100, type=int, help='the batch size')
parser.add_argument('--beta-min', default=0.0, type=float, help='the starting value for beta')
parser.add_argument('--beta-max', default=0.01, type=float, help='the final value for beta')
parser.add_argument('--device', default=None, type=str, help='the device for training, cpu or cuda')

class EEGDataset(Dataset):
    def __init__(self, data_filename, shuffle=True, device="cpu"):
        self.device = device
        numpy_data = np.load(data_filename)
        numpy_data = numpy_data[:1000]
        self.length = numpy_data.shape[1]
        self.channels = numpy_data.shape[2]
        if shuffle:
            np.random.shuffle(numpy_data)
        self.data = torch.from_numpy(numpy_data).float().to(self.device)
        self.num_samples = self.data.shape[0]
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    args = parser.parse_args()
    latent_size = args.latent_size
    if args.device == None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
    print('Device is: {}'.format(device))

    print('Loading data...')
    # Data at /home/sebgho/eeg_project/raw_data/data/snippets/snippets.npy
    file_path = args.path_to_data
    data_set = EEGDataset(file_path, device=device)

    # Create samplers for training data and test data
    test_fraction = 0.2
    val_fraction = 0.1
    indices = list(range(len(data_set)))
    test_val_split = int(np.floor(test_fraction * len(data_set)))
    val_train_split = int(np.floor((test_fraction + val_fraction) * len(data_set)))

    test_indices = indices[:test_val_split]
    val_indices = indices[test_val_split:val_train_split]
    train_indices = indices[val_train_split:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)


    # Create data loaders for test and training data
    train_loader = DataLoader(data_set, batch_size=100,
                              sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(data_set, batch_size=100,
                             sampler=val_sampler, num_workers=4)
    test_loader = DataLoader(data_set, batch_size=100,
                             sampler=test_sampler, num_workers=4)

    print('Creating models...')
    # Create autoencoder
    encoder = models.FullyConnecteEncoder(data_set.channels*data_set.length,
                                          latent_size)
    decoder = models.FullyConnecteDecoder(latent_size,
                                          data_set.channels*data_set.length)

    v_autoencoder = vae.VariationalAutoEncoder(encoder, decoder,
                                               path_to_model=args.path_to_model,
                                               device=device,
                                               n_epoch=args.num_epoch,
                                               crossvalidation=args.crossvalidation,
                                               patience=args.patience,
                                               beta_steps=10, 
                                               beta_min=args.beta_min, 
                                               beta_max=args.beta_max,
                                               snapshot=args.snapshot, 
                                               lr=0.001)
    v_autoencoder = v_autoencoder.to(device)

    print('Training model...')
    # Train model
    if args.train:
        v_autoencoder.train_vae(train_loader, val_loader)

    # Evaluate model
    if args.eval:
        if not args.train:
            v_autoencoder.load_model(args.model_filename)
        v_autoencoder.evaluate(test_loader)

        # visualize few restored data
        #z,_ = v_autoencoder.encode(x)
        #xhat = v_autoencoder.decode(z)
        #nsample = min(xhat.shape[0],5)

        #img = x[:nsample,:].reshape(-1,img_size)
        #img_hat = xhat[:nsample,:].reshape(-1,img_size)
        #plt.subplot(1,2,1)
        #plt.imshow(img,cmap='gray')
        #plt.subplot(1,2,2)
        #plt.imshow(img_hat,cmap='gray')
        #plt.show()
