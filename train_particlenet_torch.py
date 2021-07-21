import bbefp
from time import strftime

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from time import strftime

from torch_geometric.data import DataLoader

import sys, os.path as osp, os, glob, shutil
sys.path.append(osp.abspath('weaver'))
from utils.nn.model.ParticleNet import ParticleNet, FeatureConv

import awkward as ak

class ParticleNetTagger(nn.Module):
    """
    Wrapper for the weaver ParticleNet architecture
    """

    def __init__(
        self,
        pf_features_dims=5,
        num_classes=2,
        conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
        fc_params=[(128, 0.1)],
        use_fusion=True,
        use_fts_bn=True,
        use_counts=True,
        pf_input_dropout=None,
        sv_input_dropout=None,
        for_inference=False,
         **kwargs
         ):
        super(ParticleNetTagger, self).__init__(**kwargs)
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.pf_conv = FeatureConv(pf_features_dims, 32)
        self.pn = ParticleNet(
            input_dims=32,
            num_classes=num_classes,
            conv_params=conv_params,
            fc_params=fc_params,
            use_fusion=use_fusion,
            use_fts_bn=use_fts_bn,
            use_counts=use_counts,
            for_inference=for_inference
            )

    def forward(self, pf_points, pf_features, pf_mask):
        if self.pf_input_dropout:
            pf_mask = (self.pf_input_dropout(pf_mask) != 0).float()
            pf_points *= pf_mask
            pf_features *= pf_mask
        features = self.pf_conv(pf_features * pf_mask) * pf_mask
        mask = pf_mask
        return self.pn(pf_points, features, mask)


def main():
    # do_checkpoints = False # For debugging
    do_checkpoints = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    model = ParticleNetTagger().to(device)
    bbefp.networks.print_model_summary(model)

    batch_size = 32
    load = lambda merged_file: torch.utils.data.DataLoader(
        bbefp.dataset.ParticleNetDataset(merged_file),
        batch_size=batch_size, shuffle=True
        )
    train_loader = load('data/train/merged.awkd')
    test_loader = load('data/test/merged.awkd')

    epoch_size = len(train_loader.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = bbefp.networks.CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.1, policy="cosine")

    nsig = 0
    for *_, y in tqdm.tqdm(train_loader, total=len(train_loader)):
        nsig += y.sum()
    s_over_n = float(nsig/len(train_loader.dataset))
    print('sig/total=', s_over_n)
    loss_weights = torch.tensor([s_over_n, 1.-s_over_n]).to(device)


    def train(epoch):
        print('Training epoch', epoch)
        model.train()
        scheduler.step()

        i = 0

        for points, features, mask, y in tqdm.tqdm(train_loader, total=len(train_loader)):
            points = points.to(device)
            features = features.to(device)
            mask = mask.to(device)
            y = y.squeeze().to(device)

            optimizer.zero_grad()
            result = model(points, features, mask)
            log_probabilities = torch.nn.functional.log_softmax(result, dim=1)

            loss = F.nll_loss(log_probabilities, y, weight=loss_weights)
            loss.backward()

            optimizer.step()
            scheduler.batch_step()


    def test():
        with torch.no_grad():
            model.eval()

            correct = 0
            n_test = len(test_loader.dataset)
            pred = np.zeros(n_test, dtype=np.int8)
            truth = np.zeros(n_test, dtype=np.int8)

            for i, (points, features, mask, y) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
                points = points.to(device)
                features = features.to(device)
                mask = mask.to(device)
                y = y.squeeze().to(device)

                result = model(points, features, mask)
                probabilities = torch.exp(torch.nn.functional.log_softmax(result, dim=1))
                predictions = torch.argmax(probabilities, dim=-1)

                correct += predictions.eq(y).sum().item()
                pred[i*batch_size:(i+1)*batch_size] = predictions.cpu()
                truth[i*batch_size:(i+1)*batch_size] = y.cpu()

            print(confusion_matrix(truth, pred, labels=[0,1]))
            acc = correct / n_test
            print(
                'Epoch: {:02d}, Test acc: {:.4f}'
                .format(epoch, acc)
                )
            return acc

    ckpt_dir = strftime('ckpts_torchpnet_%b%d')
    def write_checkpoint(checkpoint_number=None, best=False):
        ckpt = 'ckpt_best.pth.tar' if best else 'ckpt_{0}.pth.tar'.format(checkpoint_number)
        ckpt = osp.join(ckpt_dir, ckpt)
        if best: print('Saving epoch {0} as new best'.format(checkpoint_number))
        if do_checkpoints:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(dict(model=model.state_dict()), ckpt)

    test_accs = []

    n_epochs = 200
    best_test_acc = 0.0
    for epoch in range(1, 1+n_epochs):
        train(epoch)
        test_acc = test()
        write_checkpoint(epoch)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            write_checkpoint(epoch, best=True)
        test_accs.append(test_acc)


if __name__ == '__main__':
    main()