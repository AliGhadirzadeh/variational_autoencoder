import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import autoencoder_models
import variational_autoencoder as vae
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='VAE training for robot motion trajectories')
parser.add_argument('--train', default=False, action='store_true', help='set it to train the model')
parser.add_argument('--eval', default=False, action='store_true', help='set it  to evaluate the model')
parser.add_argument('--path-to-model', default='', type=str, help='the path to save/load the model')
parser.add_argument('--path-to-data', default='', type=str, help='the path to load the data')
parser.add_argument('--model-filename', default=None, type=str, help='the filename of the model to be evaluated')
parser.add_argument('--latent-size', default=1, type=int, help='the dimension of the latent variable')
parser.add_argument('--num-epoch', default=1000, type=int, help='the number of epochs to train the model')
parser.add_argument('--snapshot', default=10, type=int, help='the number of epochs to make a snapshot of the training')
parser.add_argument('--step-length', default=1, type=int, help='the time-steps for the prediction')
parser.add_argument('--batch-size', default=1000, type=int, help='the batch size')
parser.add_argument('--beta-min', default=0.00005, type=float, help='the starting value for beta')
parser.add_argument('--beta-max', default=0.00005, type=float, help='the final value for beta')
parser.add_argument('--beta-steps', default=1, type=int, help='the number of steps to increase beta')
parser.add_argument('--kl-loss-target', default=0.0, type=float, help='the target value of the kl-loss')
parser.add_argument('--device', default='cpu', type=str, help='the device for training, cpu or cuda')


class JointActionDataset(Dataset):
    def __init__(self, data_filename, device, step_length=1):
        if 'unrolled' not in data_filename:
            traj_data = torch.from_numpy(np.load(data_filename)).float()
            num_traj = traj_data.shape[0]
            len_traj = traj_data.shape[1]
            self.num_joints = traj_data.shape[2]
            self.num_samples = num_traj * (len_traj - 1)
            self.data = torch.zeros(self.num_samples, self.num_joints)
            self.cdata = torch.zeros(self.num_samples, 1)
            ctr = 0
            for i in range(num_traj):
                for j in range((len_traj - step_length)):
                    self.cdata[ctr] = torch.tensor([j])  # torch.cat((traj_data[i][j], torch.tensor([j])))
                    self.data[ctr] = traj_data[i][j + step_length] - traj_data[i][j]
                    ctr += 1
            np.save('data_unrolled.npy', self.data)
            np.save('cdata_unrolled.npy', self.cdata)
            print('data saved')
        else:
            self.data = torch.from_numpy(np.load('data_unrolled.npy'))
            self.cdata = torch.from_numpy(np.load('cdata_unrolled.npy'))
            self.num_samples = self.data.shape[0]
            self.num_joints = self.data.shape[1]
            print('data loaded')
        self.data = self.data.to(device)
        self.cdata = self.cdata.to(device)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.cdata[idx]


def print_tensor(t, t_name):
    s = t_name
    for i in range(t.shape[0]):
        s += ', {:0.3f}'.format(t[i])
    print(s)


if __name__ == '__main__':
    args = parser.parse_args()
    latent_size = args.latent_size
    device = args.device
    print('Device is: {}'.format(device))
    joint_action_dataset = JointActionDataset(args.path_to_data, device, args.step_length)
    joint_action_loader = DataLoader(joint_action_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    n_joints = joint_action_dataset.num_joints

    encoder = autoencoder_models.FullyConnectedConditionalEncoder(n_joints, 1, latent_size)
    decoder = autoencoder_models.FullyConnectedConditionalDecoder(latent_size, 1, n_joints)
    vae_model = vae.VariationalAutoEncoder(encoder, decoder, conditional=True,
                                           path_to_model=args.path_to_model,
                                           device=device,
                                           n_epoch=args.num_epoch,
                                           beta_steps=args.beta_steps,
                                           beta_min=args.beta_min,
                                           beta_max=args.beta_max,
                                           snapshot=args.snapshot,
                                           lr=0.001,
                                           target_kl_loss=args.kl_loss_target).to(device)
    if args.train:
        vae_model.train_vae(joint_action_loader)

    if args.eval:
        vae_model.load_model('vae_0039.mdl')
        vae_model.eval()

        action_iter = iter(joint_action_loader)
        x = joint_action_dataset.data[0]
        c = joint_action_dataset.cdata[0]
        for idx in range(76):
            print_tensor(joint_action_dataset.data[idx], 'x')
            c[7] = float(idx)
            #print_tensor(c, 'c')
            z = torch.normal(mean=torch.zeros(1, 2), std=torch.ones(1, 2))
            c = joint_action_dataset.cdata[idx]
            x_hat = vae_model.decode(z, c.view(1,-1))
            c[0:7] = torch.from_numpy(x_hat)

            print_tensor(x_hat[0], 'x_hat')





