import bbefp

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from time import strftime

from torch_geometric.data import DataLoader

import sys, os.path as osp, os
sys.path.append(osp.abspath('weaver'))
from utils.nn.model.ParticleNet import ParticleNet, FeatureConv


class ParticleNetTagger(nn.Module):

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
        # pf_points = data.x
        # pf_features = data.features
        # pf_mask = data.mask
        if self.pf_input_dropout:
            pf_mask = (self.pf_input_dropout(pf_mask) != 0).float()
            pf_points *= pf_mask
            pf_features *= pf_mask
        # points = torch.cat((pf_points, sv_points), dim=2)
        # features = torch.cat((self.pf_conv(pf_features * pf_mask) * pf_mask, self.sv_conv(sv_features * sv_mask) * sv_mask), dim=2)
        # mask = torch.cat((pf_mask, sv_mask), dim=2)
        # print('pf_points:', pf_points.size())
        # print('pf_features:', pf_features.size())
        # print('pf_mask:', pf_mask.size())
        # print('pf_features * pf_mask:', (pf_features * pf_mask).size())
        features = self.pf_conv(pf_features * pf_mask) * pf_mask
        # print('features:', features.size())
        mask = pf_mask
        # print('mask:', mask.size())
        return self.pn(pf_points, features, mask)


class ParticleNetDataset(torch.utils.data.Dataset):
    def __init__(self, merged_npz, transform=None, target_transform=None):
        d, self.y = bbefp.dataset.get_data_particlenet(merged_npz, padding=True)
        self.coords = d['points'].astype(np.float32)
        self.features = d['features'].astype(np.float32)
        self.masks = d['mask'].astype(np.float32)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.coords[idx].T, self.features[idx].T, self.masks[idx].T, self.y[idx]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    # model = bbefp.networks.ParticleNet()
    model = ParticleNetTagger()

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        ParticleNetDataset('data/train/merged.npz'),
        batch_size=batch_size, shuffle=False
        )
    test_loader = torch.utils.data.DataLoader(
        ParticleNetDataset('data/test/merged.npz'),
        batch_size=batch_size, shuffle=False
        )
    # train_loader = DataLoader(
    #     bbefp.dataset.ParticleNetDataset(
    #         'data/train/merged.npz', padding=True
    #         ),
    #     batch_size=batch_size, shuffle=False
    #     )
    # test_loader = DataLoader(
    #     bbefp.dataset.ParticleNetDataset(
    #         'data/test/merged.npz', padding=True),
    #     batch_size=batch_size, shuffle=False
    #     )

    epoch_size = len(train_loader.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = bbefp.networks.CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.1, policy="cosine")
    bbefp.networks.print_model_summary(model)


    def train(epoch):
        print('Training epoch', epoch)
        model.train()
        scheduler.step()

        i = 0

        for points, features, mask, y in tqdm.tqdm(train_loader, total=len(train_loader)):
            points = points.to(device)
            features = features.to(device)
            mask = mask.to(device)
            y = y.to(device)

            # data = data.to(device)
            optimizer.zero_grad()
            result = model(points, features, mask)
            log_probabilities = torch.nn.functional.log_softmax(result, dim=1)
            # pred = probabilities.argmax(1)
            # print('probabilities=', probabilities)
            # print('pred=', pred)
            # print('data.y=', data.y)
            # raise Exception

            # print('y =', data.y)
            # print('result =', result)
            # print('Sizes of y and result:', data.y.size(), result.size())

            loss = F.nll_loss(log_probabilities, y) #, weight=loss_weights)
            loss.backward()

            #print(torch.unique(torch.argmax(result, dim=-1)))
            #print(torch.unique(data.y))

            optimizer.step()
            scheduler.batch_step()

            i += 1
            if i == 50: break

    def test():
        with torch.no_grad():
            model.eval()

            correct = 0
            n_test = len(test_loader.dataset)
            pred = np.zeros(n_test, dtype=np.int8)
            truth = np.zeros(n_test, dtype=np.int8)

            for i, (points, features, mask, y) in enumerate(tqdm.tqdm(test_loader, total=len(test_loader))):
                points = points.to(device)
                features = features.to(device)
                mask = mask.to(device)
                y = y.to(device)
                result = model(points, features, mask)
                probabilities = torch.exp(torch.nn.functional.log_softmax(result, dim=1))
                predictions = torch.argmax(probabilities, dim=-1)
                # print('probabilities=', probabilities)
                # print('pred=', pred)
                # print('data.y=', data.y)

                correct += predictions.eq(y).sum().item()
                pred[i*batch_size:(i+1)*batch_size] = predictions.cpu()
                truth[i*batch_size:(i+1)*batch_size] = y.cpu()

                if i == 49: break

            print(confusion_matrix(truth, pred, labels=[0,1]))
            acc = correct / n_test
            print(
                'Epoch: {:02d}, Test acc: {:.4f}'
                .format(epoch, acc)
                )
            return acc


    test_accs = []

    n_epochs = 5
    best_test_acc = 0.0
    for epoch in range(1, 1+n_epochs):
        train(epoch)
        test_acc = test()
        # write_checkpoint(epoch)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # write_checkpoint(epoch, best=True)
        test_accs.append(test_acc)




if __name__ == '__main__':
    main()