# coding=utf-8
import argparse
import os
import random
import time

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch

from model import ARHOL
from task import Task


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='brazil-flights', type=str)
    parser.add_argument('--hidden', default=256, type=int)
    parser.add_argument('--dimension', default=128, type=int)
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--gin-layers', default=2, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--model', default='ARHOL', choices=['ARHOL', 'AE', 'AE+ReFeX', 'GDV+ReFeX'], type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--l2', default=0.0, type=float)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument('--loop', default=100, type=int)
    return parser.parse_args()


def run(args):
    print(args)

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dgl.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    task = Task('clf')
    label = np.loadtxt('dataset/clf/{}.lbl'.format(args.dataset), delimiter=' ', dtype=int)
    graph = nx.read_edgelist('dataset/clf/{}.edge'.format(args.dataset), nodetype=int)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    g = dgl.from_networkx(graph).to('cuda:0')
    print(nx.info(graph))
    features = np.loadtxt('cache/{}.out'.format(args.dataset), dtype=float)

    features = np.log(features + 1)

    features = torch.from_numpy(features).float().cuda()
    refex = pd.read_csv('cache/{}_features.csv'.format(args.dataset)).values
    refex = torch.from_numpy(refex).float().cuda()

    save_path = 'embed/motif-ad/'
    if args.alpha == 0:
        save_path = 'embed/motif-ae/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = ARHOL(features.shape[1], args.hidden, args.dimension, refex.shape[1], args.layers).cuda()

    optimizer_g = torch.optim.Adam([{"params": model.ae.parameters()}, {"params": model.generator.parameters()}],
                                   lr=args.lr,
                                   weight_decay=args.l2)

    optimizer_d = torch.optim.Adam(
        [{"params": model.discriminator.parameters()}], lr=args.lr,
        weight_decay=args.l2)
    model.train()
    total_time = 0
    for i in range(args.epochs):
        start = time.time()
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        loss_ae, loss_g, loss_d = model(features, refex, g)
        loss = loss_ae + args.alpha * (loss_g + loss_d)
        loss.backward()
        optimizer_g.step()
        optimizer_d.step()
        print("Epoch {}: Loss: {} Time: {}".format(i + 1, loss, time.time() - start))
        total_time += time.time() - start

        if (i + 1) % 5 == 0:
            embed = model.get_embedding(features)
            task.classfication(embed, label, 0.7, loop=args.loop)

    print("Total training time: {}".format(total_time))

    model.eval()
    embed = model.get_embedding(features)

    columns = ['id'] + ['x_' + str(i) for i in range(embed.shape[1])]
    ids = np.array(range(embed.shape[0])).reshape((-1, 1))
    embedding = np.concatenate([ids, embed], axis=1)
    embedding = pd.DataFrame(embedding, columns=columns)
    embedding.to_csv('{}/{}_{}.emb'.format(save_path, args.dataset, args.dimension), index=False)

    task.classfication(embed, label, 0.7, loop=args.loop)


if __name__ == '__main__':
    args = parse_args()
    run(args)
