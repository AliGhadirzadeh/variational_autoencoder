import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import models
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, path_to_model='vae_model/', device='cpu', n_epoch=10000,
                 beta_interval=10, beta_min=0, beta_max=0.01, snapshot=100, lr = 1e-3):

        super(VariationalAutoEncoder,self).__init__()
        assert(encoder.output_size == decoder.input_size)
        self.latent_size = encoder.output_size
        self.device = device
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder

        self.deploy = False

        self.model_dir=path_to_model

        self.epoch = 0
        self.n_epoch = n_epoch
        self.beta =  beta_min #nn.Parameter(torch.Tensor([beta_min]), requires_grad=False)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_interval = beta_interval
        self.beta_idx = 0

        self.snapshot = snapshot # number of epoch

        self.init_param()

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):

        mu, logstd = self.encoder(x)

        if not self.deploy:
            std = torch.exp(logstd)
            eps = torch.randn_like(std).to(self.device)
            z = eps.mul(std).add(mu)
        else:
            z = mu

        xhat = self.decoder(z)
        xhat = xhat.view(x.size())
        return xhat, mu, logstd

    def loss(self,x,xhat,mu,logsd):
        #print("is cuda" ,x.is_cuda, xhat.is_cuda, mu.is_cuda, logsd.is_cuda)
        renonstruction_loss = F.mse_loss(xhat, x)
        var = (logsd.exp()) ** 2
        kld = 0.5 * torch.sum(-2*logsd + mu ** 2 + var , 1) - 0.5
        kld_loss = kld.mean()
        return renonstruction_loss + self.beta * kld_loss, renonstruction_loss, kld_loss

    def update_beta(self):
        epoch_to_update = (self.beta_idx+1.0)/self.beta_interval*self.n_epoch
        if self.epoch > epoch_to_update:
            self.beta = (self.beta_idx+1.0)/self.beta_interval*(self.beta_max-self.beta_min)
            self.beta_idx += 1
            print ("beta updated - new value: ", self.beta)

    def train_vae(self, data_loader):
        self.train()
        self.deploy = False
        optimizer = optim.Adam(self.parameters(), self.lr)

        self.hist_losses = np.zeros((self.n_epoch,3))
        for self.epoch in range(self.n_epoch):
            self.update_beta()
            sum_loss = 0.0
            sum_reconst_loss = 0.0
            sum_kdl_loss = 0.0
            for x in data_loader:
                if isinstance(x, list):
                    x = x[0].to(self.device)
                optimizer.zero_grad()
                xhat,  mu, logsd = self.forward(x)
                loss, reconst_loss, kdl_loss = self.loss(x, xhat, mu, logsd)
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
                sum_reconst_loss += reconst_loss.item()
                sum_kdl_loss += kdl_loss.item()

            print('[%d] loss: %.6e' %(self.epoch + 1, sum_loss / len(data_loader)))
            print('\treconst_loss: %.6e' %(sum_reconst_loss / len(data_loader)))
            print('\tkdl_loss: %.6e' %(sum_kdl_loss / len(data_loader)))
            self.hist_losses[self.epoch] = np.array([sum_loss, sum_reconst_loss, sum_kdl_loss])/ len(data_loader)

            if self.epoch % self.snapshot == (self.snapshot-1) or self.epoch == (self.n_epoch-1):
                self.save_model()


    def evaluate(self, data_loader):
        self.eval()
        self.deploy = True
        sum_loss = 0.0
        sum_reconst_loss = 0.0
        sum_kdl_loss = 0.0

        batch_size = data_loader.batch_size
        latent_var = np.zeros((len(data_loader)*batch_size, self.latent_size))

        for i, x in enumerate(data_loader):
            if isinstance(x, list):
                x = x[0].to(self.device)
            xhat,  mu, logsd = self.forward(x)
            _, reconst_loss, kdl_loss = self.loss(x, xhat, mu, logsd)
            sum_reconst_loss += reconst_loss.item()
            sum_kdl_loss += kdl_loss.item()
            lv = np.copy(mu.detach().numpy())
            c_batch_size = lv.shape[0]
            latent_var[i*batch_size:i*batch_size+c_batch_size,:] = lv

        print('reconst_loss: %.6e' %(sum_reconst_loss / len(data_loader)))
        print('kdl_loss: %.6e' %(sum_kdl_loss / len(data_loader)))

        if self.latent_size < 10:
            for i in range(self.latent_size):
                plt.subplot(self.latent_size*100+11+i)
                plt.hist(latent_var[:,i], bins=20)
                plt.xlim((-1,1))
            plt.show()


    def decode(self, z):
        """
        The function decodes the latent variable into the original signal
        Args:
            z (np.ndarray or tensor): the input latent variable
        Returns:
            xhat (np.ndarray): the decoded signal
        """
        if isinstance(z,np.ndarray):
            z = torch.from_numpy(z).detach()
        xhat = self.decoder(z).detach().numpy()
        return xhat

    def encode(self, x):
        """
        The function encodes the input data into a latent variable
        Args:
            x (np.ndarray or tensor): the input data
        Returns:
            mu (np.ndarray): the mean of the encoded variable
            logsd (np.ndarray): the log(sd) of the encoded variable
        """
        if isinstance(x,np.ndarray):
            x = torch.from_numpy(x)
        mu, logsd = self.encoder(x)
        return mu.detach().numpy(), logsd.detach().numpy()

    def save_model(self):
        filepath = os.path.join(self.model_dir, 'vae_{:04d}.mdl'.format(self.epoch))
        torch.save(self.state_dict(), filepath)
        plt_labels = ['loss', 'reconst. loss', 'kdl loss']
        for i in range(3):
            plt.plot(np.arange(self.epoch),self.hist_losses[:self.epoch,i], label=plt_labels[i])
        plt.xlabel('epoch')
        plt.ylabel('losses')
        plt.legend(loc='upper right')
        filepath = os.path.join(self.model_dir, 'loss_{:04d}.jpg'.format(self.epoch))
        plt.savefig(filepath)
        plt.clf()

    def load_model(self, filename):
        filepath = os.path.join(self.model_dir, filename)
        self.load_state_dict(torch.load(filepath,map_location=self.device) )
        self.eval()
