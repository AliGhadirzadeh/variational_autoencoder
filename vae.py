import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    def __init__(self, data_loader, encoder, decoder, device='cpu', n_epoch=10000,
                 beta_interval=10, beta_min=1.0e-6, beta_max=1.0e-0, lr = 1e-3):

        super(VariationalAutoEncoder, self).__init__()
        assert(encoder.output_size == decoder.input_size)
        self.latent_size = encoder.output_size
        self.device = device
        self.lr = lr
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), self.lr)
        self.data_loader = data_loader

        self.deploy = False

        self.n_epoch = n_epoch
        self.beta = beta_min
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_interval = beta_interval
        self.beta_idx = 0

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

    def loss(self, x):
        xhat,  mu, logsd = self.forward(x)
        renonstruction_loss = F.mse_loss(xhat, x)
        var = (logsd.exp()) ** 2
        kld = 0.5 * torch.sum(-2*logsd + mu ** 2 + var , 1) - 0.5
        kld_loss = kld.mean()
        return renonstruction_loss + self.beta * kld_loss, renonstruction_loss, kld_loss

    def update_beta(self, epoch):
        epoch_to_update = (self.beta_idx+1.0)/self.beta_interval*self.n_epoch
        if epoch > epoch_to_update:
            self.beta = (self.beta_idx+1.0)/self.beta_interval*(self.beta_max-self.beta_min)
            self.beta_idx += 1
            print ("beta updated - new value: ", self.beta)

    def train(self):
        for epoch in range(self.n_epoch):
            self.update_beta(epoch)
            sum_loss = 0.0
            sum_reconst_loss = 0.0
            sum_kdl_loss = 0.0
            for x in self.data_loader:
                if isinstance(x, list):
                    x = x[0]
                self.optimizer.zero_grad()
                loss, reconst_loss, kdl_loss = self.loss(x.to(self.device))
                loss.backward()
                self.optimizer.step()
                sum_loss += loss.item()
                sum_reconst_loss += reconst_loss.item()
                sum_kdl_loss += kdl_loss.item()

            print('[%d] loss: %.6e' %(epoch + 1, sum_loss / len(self.data_loader)))
            print('\treconst_loss: %.6e' %(sum_reconst_loss / len(self.data_loader)))
            print('\tkdl_loss: %.6e' %(sum_kdl_loss / len(self.data_loader)))

        #torch.save(self.model.state_dict(), 'model.mdl')
