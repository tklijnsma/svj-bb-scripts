import os
import os.path as osp
import math
import tqdm

import numpy as np
import torch
import torch.nn.functional as F

import bbefp
from time import strftime
from sklearn.metrics import confusion_matrix


def main():


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)
    ckpt_dir = strftime('testckpts_znn_%b%d_%H%M%S')

    n_epochs = 20

    batch_size = 16
    train_loader = bbefp.dataset.get_loader_ptetaphie('data/train/merged.npz', batch_size)
    test_loader = bbefp.dataset.get_loader_ptetaphie('data/test/merged.npz', batch_size)

    epoch_size = len(train_loader.dataset)
    model = bbefp.networks.DynamicReductionNetwork(hidden_dim=64, k=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = bbefp.networks.CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.1, policy="cosine")
    bbefp.networks.print_model_summary(model)

    nsig = 0


    for data in tqdm.tqdm(train_loader, total=len(train_loader)):
        nsig += data.y.sum()
    s_over_n = float(nsig/len(train_loader.dataset))
    print('sig/total=', s_over_n)
    loss_weights = torch.tensor([s_over_n, 1.-s_over_n]).to(device)
    # loss_weights = torch.tensor([1., 4.]).to(device)

    def write_checkpoint(checkpoint_number=None, best=False):
        ckpt = 'ckpt_best.pth.tar' if best else 'ckpt_{0}.pth.tar'.format(checkpoint_number)
        ckpt = osp.join(ckpt_dir, ckpt)
        os.makedirs(ckpt_dir, exist_ok=True)
        if best: print('Saving epoch {0} as new best'.format(checkpoint_number))
        torch.save(dict(model=model.state_dict()), ckpt)

    def train(epoch):
        print('Training epoch', epoch)
        model.train()
        scheduler.step()

        for data in tqdm.tqdm(train_loader, total=len(train_loader)):
            data = data.to(device)
            optimizer.zero_grad()
            result = model(data)
            log_probabilities = torch.nn.functional.log_softmax(result, dim=1)
            # pred = probabilities.argmax(1)
            # print('probabilities=', probabilities)
            # print('pred=', pred)
            # print('data.y=', data.y)
            # raise Exception

            # print('y =', data.y)
            # print('result =', result)
            # print('Sizes of y and result:', data.y.size(), result.size())

            loss = F.nll_loss(log_probabilities, data.y, weight=loss_weights)
            loss.backward()

            #print(torch.unique(torch.argmax(result, dim=-1)))
            #print(torch.unique(data.y))

            optimizer.step()
            scheduler.batch_step()

    def test():
        with torch.no_grad():
            model.eval()

            correct = 0
            n_test = len(test_loader.dataset)
            pred = np.zeros(n_test, dtype=np.int8)
            truth = np.zeros(n_test, dtype=np.int8)

            for i, data in enumerate(test_loader):
                data = data.to(device)
                result = model(data)
                probabilities = torch.exp(torch.nn.functional.log_softmax(result, dim=1))
                predictions = torch.argmax(probabilities, dim=-1)
                # print('probabilities=', probabilities)
                # print('pred=', pred)
                # print('data.y=', data.y)

                correct += predictions.eq(data.y).sum().item()
                pred[i*batch_size:(i+1)*batch_size] = predictions.cpu()
                truth[i*batch_size:(i+1)*batch_size] = data.y.cpu()

            print(confusion_matrix(truth, pred, labels=[0,1]))
            acc = correct / n_test
            print(
                'Epoch: {:02d}, Test acc: {:.4f}'
                .format(epoch, acc)
                )
            return acc


    test_accs = []

    best_test_acc = 0.0
    for epoch in range(1, 1+n_epochs):
        train(epoch)
        test_acc = test()
        write_checkpoint(epoch)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            write_checkpoint(epoch, best=True)
        test_accs.append(test_acc)

    np.savez(osp.join(ckpt_dir, 'testaccs.npz'), testaccs=test_accs)

if __name__ == '__main__':
    main()
