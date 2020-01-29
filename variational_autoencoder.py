import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import autoencoder_models
import matplotlib as mpl
if not "DISPLAY" in os.environ:
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, path_to_model='vae_model/', device='cpu', n_epoch=10000,
                 beta_steps=10, beta_min=0, beta_max=0.01, snapshot=100, lr = 1e-4, target_kl_loss=0):

        super(VariationalAutoEncoder,self).__init__()
        assert(encoder.output_size == decoder.input_size)
        self.latent_size = encoder.output_size
        self.device = device
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder

        self.model_dir=path_to_model

        self.epoch = 0
        self.n_epoch = n_epoch
        self.beta =  beta_min #nn.Parameter(torch.Tensor([beta_min]), requires_grad=False)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_steps = beta_steps
        self.beta_idx = 0

        self.target_kl_loss = target_kl_loss
        self.average_kl_loss = 0

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
        var = torch.exp(logstd)
        eps = torch.randn_like(var).to(self.device)
        z = eps.mul(var).add(mu)
        xhat = self.decoder(z)
        xhat = xhat.view(x.size())

        return xhat, mu, logstd, z

    def forward_deploy(self, x):

        mu, logstd = self.encoder(x)
        var = torch.exp(logstd)
        eps = torch.randn_like(var).to(self.device)
        z = eps.mul(var).add(mu)
        xhat = self.decoder(mu)
        xhat = xhat.view(x.size())

        return xhat, mu, logstd, z

    def loss(self,x,xhat,mu,logsd):
        renonstruction_loss = F.mse_loss(xhat, x)
        var = torch.exp(logsd)

        kld = torch.sum(-logsd + (mu ** 2)*0.5 + var ,1) - self.latent_size

        kld_loss = kld.mean()

        return renonstruction_loss + self.beta * kld_loss, renonstruction_loss, kld_loss

    def update_beta(self):
        epoch_to_update = (self.beta_idx+1.0)/self.beta_steps*self.n_epoch
        if self.epoch > epoch_to_update:
            if self.average_kl_loss > self.target_kl_loss:
                self.beta = (self.beta_idx+1.0)/self.beta_steps*(self.beta_max-self.beta_min)
                self.beta_idx += 1
                print ("beta updated - new value: ", self.beta)

    def train_vae(self, data_loader):
        self.train()
        optimizer = optim.Adam(self.parameters(), self.lr)

        self.hist_losses = np.zeros((self.n_epoch,3))
        for self.epoch in range(self.n_epoch):
            self.update_beta()
            sum_loss = 0.0
            sum_reconst_loss = 0.0
            sum_kl_loss = 0.0
            for x in data_loader:
                if isinstance(x, list):
                    x = x[0].to(self.device)
                optimizer.zero_grad()
                xhat,  mu, logsd, z = self.forward(x)
                loss, reconst_loss, kl_loss = self.loss(x, xhat, mu, logsd)
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
                sum_reconst_loss += reconst_loss.item()
                sum_kl_loss += kl_loss.item()

            if self.epoch == 0:
                self.average_kl_loss = sum_kl_loss / len(data_loader)
            else:
                self.average_kl_loss = 0.8*self.average_kl_loss+0.2*(sum_kl_loss / len(data_loader))

            print('[%d] loss: %.6e' %(self.epoch + 1, sum_loss / len(data_loader)))
            print('\treconst_loss: %.6e' %(sum_reconst_loss / len(data_loader)))
            print('\tkl_loss: %.6e' %(sum_kl_loss / len(data_loader)))
            print('\tbeta: %.6e' %(self.beta))
            print('\tlearning rate: %.6e' % (self.get_lr(optimizer)))
            self.hist_losses[self.epoch] = np.array([sum_loss, sum_reconst_loss, sum_kl_loss])/ len(data_loader)

            if self.epoch % self.snapshot == (self.snapshot-1) or self.epoch == (self.n_epoch-1):
                self.save_model()

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def evaluate(self, data_loader):
        self.eval()
        sum_loss = 0.0
        sum_reconst_loss = 0.0
        sum_kl_loss = 0.0

        batch_size = data_loader.batch_size
        latent_var = np.zeros((len(data_loader)*batch_size, self.latent_size))
        labels = np.zeros(len(data_loader)*batch_size)
        for i, x in enumerate(data_loader):
            label =[]
            if isinstance(x, list):
                label=x[1].to(self.device)
                x = x[0].to(self.device)
            xhat,  mu, logsd, z = self.forward_deploy(x)
            _, reconst_loss, kl_loss = self.loss(x, xhat, mu, logsd)
            sum_reconst_loss += reconst_loss.item()
            sum_kl_loss += kl_loss.item()
            lv = np.copy(z.detach().numpy())
            c_batch_size = lv.shape[0]
            latent_var[i*batch_size:i*batch_size+c_batch_size,:] = lv
            if len(label)>0:
                labels[i*batch_size:i*batch_size+c_batch_size] = label

        print('reconst_loss: %.6e' %(sum_reconst_loss / len(data_loader)))
        print('kl_loss: %.6e' %(sum_kl_loss / len(data_loader)))

        return latent_var, labels


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
        z = z.float()
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
        """
        Saving the neural network parameters as well as the training curves
        Args:
        Returns:
        """
        filepath = os.path.join(self.model_dir, 'vae_{:04d}.mdl'.format(self.epoch))
        torch.save(self.state_dict(), filepath)
        plt_labels = ['loss', 'reconst. loss', 'kdl loss']
        for i in range(3):
            plt.subplot(3,1,i+1)
            plt.plot(np.arange(self.snapshot-1),self.hist_losses[self.epoch-self.snapshot+1:self.epoch,i], label=plt_labels[i])
            plt.ylabel(plt_labels[i])
            plt.xlabel('epoch')

        filepath = os.path.join(self.model_dir, 'loss_{:04d}.jpg'.format(self.epoch))
        plt.savefig(filepath)
        plt.clf()
        print("model saved successfully")

    def load_model(self, filename):
        filepath = os.path.join(self.model_dir, filename)
        self.load_state_dict(torch.load(filepath,map_location=self.device) )
        self.eval()
