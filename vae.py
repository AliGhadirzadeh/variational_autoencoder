import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

class FullyConnecteEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnecteEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4_mean = nn.Linear(250, output_size)
        self.fc4_logsd = nn.Linear(250, output_size)
        self.tanh = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(500)
        self.bn3 = nn.BatchNorm1d(250)

    def forward(self, x):
        x=x.view(-1,self.input_size)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        mean = self.tanh(self.fc4_mean(x))
        logsd = self.fc4_logsd(x)
        return mean, logsd

class FullyConnecteDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnecteDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 250)
        self.fc3 = nn.Linear(250, 500)
        self.fc4 = nn.Linear(500, 1000)
        self.fc5 = nn.Linear(1000, output_size)

        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(250)
        self.bn3 = nn.BatchNorm1d(500)
        self.bn4 = nn.BatchNorm1d(1000)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.sigmoid(self.fc5(x))
        return x


class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, path_to_model='vae_model/', device='cpu', n_epoch=10000,
                 beta_interval=10, beta_min=1.0e-6, beta_max=1.0e-0, lr = 1e-3):

        super(VariationalAutoEncoder, self).__init__()
        assert(encoder.output_size == decoder.input_size)
        self.latent_size = encoder.output_size
        self.device = device
        self.lr = lr
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), self.lr)

        self.deploy = False

        self.model_dir=path_to_model

        self.epoch = 0
        self.n_epoch = n_epoch
        self.beta = beta_min
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_interval = beta_interval
        self.beta_idx = 0

        self.snapshot = 100 # number of epoch

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

        self.hist_losses = np.zeros((self.n_epoch,3))
        for self.epoch in range(self.n_epoch):
            self.update_beta()
            sum_loss = 0.0
            sum_reconst_loss = 0.0
            sum_kdl_loss = 0.0
            for x in data_loader:
                if isinstance(x, list):
                    x = x[0].to(self.device)
                self.optimizer.zero_grad()
                xhat,  mu, logsd = self.forward(x)
                loss, reconst_loss, kdl_loss = self.loss(x, xhat, mu, logsd)
                loss.backward()
                self.optimizer.step()
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

        for i in range(self.latent_size):
            plt.subplot(510+i+1)
            plt.hist(latent_var[:,i], bins=20)
            plt.xlim((-1,1))
        plt.show()

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
        self.load_state_dict(torch.load(filepath) )
        self.eval()
