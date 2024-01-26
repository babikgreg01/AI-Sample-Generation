import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 64, 1)
    

class CVAE(nn.Module):
    def __init__(self, h_dim=16384, z_dim=2):
        super(CVAE, self).__init__()
        self.encoder = nn.Sequential(
            # input shape: (bs, 1, 1024, 16)
            nn.Conv2d(1, 32, kernel_size=3, stride=(2,1), padding=1),
            nn.ReLU(),
            # input shape: (bs, 32, 512, 16)
            nn.Conv2d(32, 64, kernel_size=3, stride=(2,2), padding=1),
            nn.ReLU(),
            # input shape: (bs, 64, 256, 8)
            nn.Conv2d(64, 128, kernel_size=3, stride=(2,2), padding=1),
            nn.ReLU(),
            # input shape: (bs, 128, 128, 4)
            nn.Conv2d(128, 256, kernel_size=3, stride=(2,2), padding=1),
            nn.ReLU(),
            # input shape: (bs, 256, 64, 2)
            nn.Conv2d(256, 512, kernel_size=3, stride=(2,2), padding=1),
            nn.ReLU(),
            # input shape: (bs, 512, 32, 1)
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
           

        self.decoder = nn.Sequential(
            UnFlatten(),
            # input shape: (bs, 512, 32, 1)
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=(2,2), padding=1, output_padding=(1,1)),
            nn.ReLU(),
            # input shape: (bs, 256, 64, 2)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=(2,2), padding=1, output_padding=(1,1)),
            nn.ReLU(),
            # input shape: (bs, 128, 128, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=(2,2), padding=1, output_padding=(1,1)),
            nn.ReLU(),
            # input shape: (bs, 64, 256, 8)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=(2,2), padding=1, output_padding=(1,1)),
            nn.ReLU(),
            # input shape: (bs, 32, 512, 16)
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0)),
            nn.Sigmoid(),
            # output shape: (bs, 1, 1024, 16)
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = to_var(torch.randn(*mu.size()))
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
    return BCE + KLD