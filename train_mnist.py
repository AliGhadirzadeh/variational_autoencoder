import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import vae

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True,transform=transform)
mnistloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True, num_workers=0)

#dataiter = iter(mnistloader)
#img, label = dataiter.next()
#imshow(torchvision.utils.make_grid(img))

encoder = vae.FullyConnecteEncoder(28*28,5)
decoder = vae.FullyConnecteDecoder(5,28*28)
v_autoencoder = vae.VariationalAutoEncoder(mnistloader, encoder, decoder, device='cpu',n_epoch=10)

v_autoencoder.train()
