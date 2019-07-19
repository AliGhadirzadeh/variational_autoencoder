import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import models
import vae

import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='VAE training using MNIST dataset')
parser.add_argument('--train', default=False,action='store_true', help='set it to train the model')
parser.add_argument('--eval', default=False, action='store_true', help='set it  to evaluate the model')
parser.add_argument('--path-to-model', default=None, type=str, help='the path to save/load the model')
parser.add_argument('--model-filename', default=None, type=str, help='the filename of the model to be evaluated')
parser.add_argument('--latent-size', default=5, type=int, help='the dimension of the latent variable')
parser.add_argument('--image-size', default=12, type=int, help='MNIST images are resized by the --image-size')
parser.add_argument('--show-sample-images', default=False, action='store_true', help='set it to show a number of sampled images from the dataset')
parser.add_argument('--num-epoch', default=10000, type=int, help='the number of epochs to train the model')
parser.add_argument('--snapshot', default=100, type=int, help='the number of epochs to make a snapshot of the training')
parser.add_argument('--batch-size', default=100, type=int, help='the batch size')
parser.add_argument('--beta-min', default=0.0, type=float, help='the starting value for beta')
parser.add_argument('--beta-max', default=0.01, type=float, help='the final value for beta')
parser.add_argument('--beta-steps', default=20, type=int, help='the number of steps to increase beta')
parser.add_argument('--device', default=None, type=str, help='the device for training, cpu or cuda')

if __name__ == '__main__':
    args = parser.parse_args()

    img_size = args.image_size
    latent_size = args.latent_size

    # define the data transform
    transform = transforms.Compose(
        [transforms.Resize(img_size),
         transforms.ToTensor()])

    # construct the train and test data loader
    size_test_data = 10000

    trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True,transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

    #trainset_part = torch.utils.data.random_split(trainset, [len(trainset)-size_test_data, size_test_data])[0]
    testset_part = torch.utils.data.random_split(testset, [size_test_data, len(testset)-size_test_data])[0]

    mnist_test_loader = torch.utils.data.DataLoader(testset_part, batch_size=args.batch_size, shuffle=True, num_workers=0)
    mnist_train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,shuffle=True, num_workers=0)

    # making the autoencoder
    #encoder = models.FullyConnecteEncoder(img_size*img_size,latent_size)
    encoder = models.ConvolutionalEncoder(img_size,img_size,1,latent_size)
    decoder = models.FullyConnecteDecoder(latent_size,img_size*img_size)

    if args.device == None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
    print('Device is: {}'.format(device))

    v_autoencoder = vae.VariationalAutoEncoder(encoder, decoder,path_to_model=args.path_to_model,
                                                device=device,n_epoch=args.num_epoch,
                                                beta_steps=args.beta_steps, beta_min=args.beta_min, beta_max=args.beta_max,
                                                snapshot=args.snapshot, lr=0.001)
    v_autoencoder = v_autoencoder.to(device)


    # show some sample images
    if args.show_sample_images:
        dataiter = iter(mnist_train_loader)
        x, _ = dataiter.next()
        x = x.numpy()
        nsample = min(x.shape[0],5)
        img = x[:nsample,:].reshape(-1,img_size)

        plt.imshow(img,cmap='gray')
        plt.show()

    # train the model
    if args.train:
        v_autoencoder.train_vae(mnist_train_loader)

    # evaluate the model
    if args.eval:
        if not args.train:
            v_autoencoder.load_model(args.model_filename)
        lv, label = v_autoencoder.evaluate(mnist_test_loader)

        # visualize the latent space
        if latent_size > 2:
            for i in range(latent_size):
                plt.subplot(latent_size, 1, i+1)
                for j in range(10):
                    idx = np.where(label == j)
                    plt.hist(lv[idx[0],i], bins=50,stacked=False,histtype='step')
        else:
            for j in range(10):
                idx = np.where(label == j)
                plt.plot(lv[idx[0],0], lv[idx[0],1],'.', label=str(j))
        plt.legend()
        plt.show()

        # visualize few restored data
        dataiter = iter(mnist_test_loader)
        x, _ = dataiter.next()
        x=x.numpy()
        z,_ = v_autoencoder.encode(x)
        xhat = v_autoencoder.decode(z)
        nsample = min(xhat.shape[0],5)

        img = x[:nsample,:].reshape(-1,img_size)
        img_hat = xhat[:nsample,:].reshape(-1,img_size)
        plt.subplot(1,2,1)
        plt.imshow(img,cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(img_hat,cmap='gray')
        plt.show()
