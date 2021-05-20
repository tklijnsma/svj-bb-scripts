import bbefp

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from time import strftime


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    efps = [467]

    batch_size = 16
    train_loader = bbefp.dataset.get_loader_ptm_plus_efps(
        'data/train/merged.npz',
        'precalc/efp_May04.npz',
        efps, batch_size
        )
    test_loader = bbefp.dataset.get_loader_ptm_plus_efps(
        'data/test/merged.npz',
        'precalc/efp_test_May05.npz',
        efps, batch_size
        )

    model = bbefp.networks.PTMNet(2 + len(efps))
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)


    def train(i_epoch):
        print('Training epoch', i_epoch)
        model.train()
        for data in tqdm.tqdm(train_loader, total=len(train_loader)):
            data, y = data
            data = data.to(device)
            optimizer.zero_grad()
            result = model(data)
            log_probabilities = torch.nn.functional.log_softmax(result, dim=1)
            loss = F.nll_loss(log_probabilities, y)
            loss.backward()
            optimizer.step()

    def test():
        with torch.no_grad():
            model.eval()
            correct = 0
            n_test = len(test_loader.dataset)
            pred = np.zeros(n_test, dtype=np.int8)
            truth = np.zeros(n_test, dtype=np.int8)
            for i, data in enumerate(test_loader):
                data, y = data
                data = data.to(device)
                result = model(data)
                probabilities = torch.exp(torch.nn.functional.log_softmax(result, dim=1))
                predictions = torch.argmax(probabilities, dim=-1)
                correct += predictions.eq(y).sum().item()
                pred[i*batch_size:(i+1)*batch_size] = predictions.cpu()
                truth[i*batch_size:(i+1)*batch_size] = y.cpu()
            print(confusion_matrix(truth, pred, labels=[0,1]))
            acc = correct / n_test
            return acc

    ckpt = strftime(
        'ckpt_%b%d_%H%M%S_efps_{}_best.pth.tar'
        .format('_'.join([str(efp) for efp in efps]))
        )
    best_acc = 0.
    for i_epoch in range(150):
        train(i_epoch)
        acc = test()
        print(
            'Epoch: {:02d}, Test acc: {:.4f}'
            .format(i_epoch, acc)
            )
        if acc > best_acc:
            print(f'New best acc {acc:.3} (was {best_acc:.3f})')
            torch.save(dict(model=model.state_dict()), ckpt)
            best_acc = acc
        else:
            print(f'Did not improve from best_acc {best_acc:.3f}')


if __name__ == '__main__':
    main()

