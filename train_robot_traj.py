import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import autoencoder_models
import variational_autoencoder as vae
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='VAE training for robot motion trajectories')
parser.add_argument('--train', default=False, action='store_true', help='set it to train the model')
parser.add_argument('--eval', default=False, action='store_true', help='set it  to evaluate the model')
parser.add_argument('--conditional', default=False, action='store_true', help='set it to make it conditional vae')
parser.add_argument('--path-to-model', default='', type=str, help='the path to save/load the model')
parser.add_argument('--path-to-data', default='', type=str, help='the path to load the data')
parser.add_argument('--path-to-cdata', default='', type=str, help='the path to load the condition data')
parser.add_argument('--model-filename', default=None, type=str, help='the filename of the model to be evaluated')
parser.add_argument('--latent-size', default=5, type=int, help='the dimension of the latent variable')
parser.add_argument('--num-epoch', default=10000, type=int, help='the number of epochs to train the model')
parser.add_argument('--snapshot', default=100, type=int, help='the number of epochs to make a snapshot of the training')
parser.add_argument('--batch-size', default=100, type=int, help='the batch size')
parser.add_argument('--beta-min', default=0.0, type=float, help='the starting value for beta')
parser.add_argument('--beta-max', default=0.01, type=float, help='the final value for beta')
parser.add_argument('--beta-steps', default=20, type=int, help='the number of steps to increase beta')
parser.add_argument('--kl-loss-target', default=0.0, type=float, help='the target value of the kl-loss')
parser.add_argument('--device', default='cpu', type=str, help='the device for training, cpu or cuda')


class TrajDataset(Dataset):
    def __init__(self, data_filename, cdata_filename=''):
        self.data = torch.from_numpy(np.load(data_filename)).float()
        if cdata_filename:
            self.cdata = torch.from_numpy(np.load(cdata_filename)).float()
            self.cond_data_available = True
        else:
            self.cond_data_available = False
        self.num_samples = self.data.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.cond_data_available:
            return self.data[idx], self.cdata[idx]
        else:
            return self.data[idx]


if __name__ == '__main__':
    args = parser.parse_args()
    latent_size = args.latent_size
    device = args.device
    print('Device is: {}'.format(device))

    traj_dataset = TrajDataset(args.path_to_data, args.path_to_cdata)
    trajectory_loader = DataLoader(traj_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    traj_iter = iter(trajectory_loader)
    x, c = traj_iter.next()
    n_joints = x.shape[1]
    traj_length = x.shape[2]
    condition_size = c.shape[1]

    # creating the autoencoder
    if args.conditional:
        encoder = autoencoder_models.FullyConnectedConditionalEncoder(n_joints*traj_length, condition_size, latent_size)
        decoder = autoencoder_models.FullyConnectedConditionalDecoder(latent_size, condition_size, n_joints*traj_length)
    else:
        encoder = autoencoder_models.FullyConnectedEncoder(n_joints * traj_length, latent_size)
        decoder = autoencoder_models.FullyConnectedDecoder(latent_size, n_joints * traj_length)
    v_autoencoder = vae.VariationalAutoEncoder(encoder, decoder, conditional=args.conditional, path_to_model=args.path_to_model,
                                                device=device,n_epoch=args.num_epoch,
                                                beta_steps=args.beta_steps, beta_min=args.beta_min, beta_max=args.beta_max,
                                                snapshot=args.snapshot, lr=0.001, target_kl_loss=args.kl_loss_target)
    v_autoencoder = v_autoencoder.to(device)

    # train the model
    if args.train:
        v_autoencoder.train_vae(trajectory_loader)

    # evaluate the model
    if args.eval:
        if not args.train:
            v_autoencoder.load_model(args.model_filename)
        lv, _ = v_autoencoder.evaluate(trajectory_loader)

        if args.latent_size ==1:
            plt.hist(lv, bins='auto')
            plt.show()

        elif args.latent_size ==2:
            plt.plot(lv[:,0], lv[:,1],'.')
            plt.show()

        else:
            for i in range(args.latent_size):
                for j in range(i+1, args.latent_size):
                    plt.figure()
                    plt.plot(lv[:,i], lv[:,j],'.')
            plt.show()

        # visualize few restored data
        z,_ = v_autoencoder.encode(x)
        xhat = v_autoencoder.decode(z).reshape(x.shape)



        nsample = min(x.shape[0],3)
        x = x[:nsample, :, :]
        xhat = xhat[:nsample, : ,:]


        #traj = x[:nsample].reshape(-1,traj_length)
        #traj_hat = xhat[:nsample].reshape(-1,traj_length)
        for i in range(nsample):
            plt.subplot(nsample,1,i+1)
            for j in range(7):
                if x.shape[1] == 8:
                    plt.plot(x[i][0], x[i][j+1], color='b')
                    plt.plot(x[i][0], xhat[i][j+1], color='r')
                else:
                    plt.plot(np.arange(x.shape[2]), x[i][j], color='b')
                    plt.plot(np.arange(x.shape[2]), xhat[i][j], color='r')
        plt.show()
