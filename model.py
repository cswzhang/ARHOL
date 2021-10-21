# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GINConv


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, layers, gin_layer):
        super().__init__()
        self.encoder = nn.ModuleList()
        for i in range(gin_layer):
            if i == 0:
                self.encoder.append(GINConv(MLP(input_dim, hidden_dim, hidden_dim, layers), 'sum', 0, True))
            elif i == gin_layer - 1:
                self.encoder.append(GINConv(MLP(hidden_dim, hidden_dim, out_dim, layers), 'sum', 0, True))
            else:
                self.encoder.append(GINConv(MLP(hidden_dim, hidden_dim, hidden_dim, layers), 'sum', 0, True))

    def forward(self, g, h):
        # x, edge_index = data.x, data.edge_index
        for i, layer in enumerate(self.encoder):
            h = layer(g, h)
        return h


class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, layers):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(layers):
            if i == 0:
                self.encoder.append(nn.Linear(input_dim, hidden_dim))
                self.decoder.append(nn.Linear(out_dim, hidden_dim))
            elif i == layers - 1:
                self.encoder.append(nn.Linear(hidden_dim, out_dim))
                self.decoder.append(nn.Linear(hidden_dim, input_dim))
            else:
                self.encoder.append(nn.Linear(hidden_dim, hidden_dim))
                self.decoder.append(nn.Linear(hidden_dim, hidden_dim))

    def encode(self, h):
        for i, layer in enumerate(self.encoder):
            h = layer(h)
            h = F.relu(h)
        return h

    def forward(self, h):
        h = self.encode(h)
        out = h
        for i, layer in enumerate(self.decoder):
            out = layer(out)
            out = F.relu(out)
        return h, out

    def get_embedding(self, h):
        h = self.encode(h)
        return h


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, layers):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(layers - 1):
            if i == 0:
                self.encoder.append(nn.Linear(input_dim, hidden_dim))
                self.decoder.append(nn.Linear(out_dim, hidden_dim))
            else:
                self.encoder.append(nn.Linear(hidden_dim, hidden_dim))
                self.decoder.append(nn.Linear(hidden_dim, hidden_dim))
        self.decoder.append(nn.Linear(hidden_dim, input_dim))
        self.mu = nn.Linear(hidden_dim, out_dim)
        self.sigma = nn.Linear(hidden_dim, out_dim)

        for i, layer in enumerate(self.encoder):
            nn.init.xavier_normal_(self.encoder[i].weight.data)
            nn.init.normal_(self.encoder[i].bias.data)
        for i, layer in enumerate(self.decoder):
            nn.init.xavier_normal_(self.decoder[i].weight.data)
            nn.init.normal_(self.decoder[i].bias.data)
        nn.init.xavier_normal_(self.mu.weight.data)
        nn.init.normal_(self.mu.bias.data)
        nn.init.xavier_normal_(self.sigma.weight.data)
        nn.init.normal_(self.sigma.bias.data)

    def encode(self, h):
        for i, layer in enumerate(self.encoder):
            h = layer(h)
            h = F.relu(h)
        mu = self.mu(h)
        sigma = self.sigma(h).exp()
        epsilon = torch.from_numpy(np.random.normal(0, 1, sigma.size())).float().cuda()
        z = mu + epsilon * sigma
        return z

    def forward(self, h):
        h = self.encode(h)
        out = h
        for i, layer in enumerate(self.decoder):
            out = layer(out)
            out = F.relu(out)
        return h, out


class Adversary(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, layers):
        super().__init__()
        self.generator = nn.ModuleList()
        for i in range(layers):
            if i == 0:
                self.generator.append(nn.Linear(input_dim, hidden_dim))
            elif i == layers - 1:
                self.generator.append(nn.Linear(hidden_dim, out_dim))
            else:
                self.generator.append(nn.Linear(hidden_dim, hidden_dim))

        self.discriminator = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def encode(self, h):
        for i, layer in enumerate(self.generator):
            h = layer(h)
            h = F.relu(h)
        return h

    def forward(self, pos, neg):
        neg = self.encode(neg)
        pos_lbl = self.discriminator(pos)
        neg_lbl = self.discriminator(neg)
        return pos_lbl, neg_lbl


class ARHOL(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, pos_dim, layers):
        super().__init__()
        self.ae = AE(input_dim, hidden_dim, out_dim, layers)
        self.generator = GIN(out_dim, hidden_dim, pos_dim, layers, layers)

        self.discriminator = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features, pos, g):
        h, out = self.ae(features)
        pos_lbl = self.discriminator(pos)

        h = self.generator(g, h)

        neg_lbl = self.discriminator(h.detach())
        loss_ae = F.mse_loss(out, features)
        loss_d = 0.5 * (F.binary_cross_entropy_with_logits(pos_lbl, torch.ones_like(pos_lbl)) +
                        F.binary_cross_entropy_with_logits(neg_lbl, torch.zeros_like(neg_lbl)))
        loss_g = F.binary_cross_entropy_with_logits(self.discriminator(h), torch.ones_like(neg_lbl))
        return loss_ae, loss_g, loss_d

    def get_embedding(self, features):
        h, _ = self.ae(features)

        return h.cpu().data.numpy()
