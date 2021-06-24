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
        # print('pf_conv(pf_features * pf_mask):', self.pf_conv(pf_features * pf_mask).size())
        features = self.pf_conv(pf_features * pf_mask) * pf_mask
        # print('features:', features.size())
        mask = pf_mask
        # print('mask:', mask.size())
        return self.pn(pf_points, features, mask)



class ParticleNetDataset(torch.utils.data.Dataset):
    def __init__(
        self, merged_npz,
        lefts=None, rights=None, n_events_max=None,
        force_reproc=False, n_constituents=200, pad=True,
        transform=None, target_transform=None
        ):
        self.procdir = osp.join((osp.dirname(osp.abspath(merged_npz))), 'pnet_proc')
        print('procdir:', self.procdir)

        if force_reproc and osp.isdir(self.procdir):
            print(f'Reprocessing - removing existing {self.procdir}')
            shutil.rmtree(self.procdir)

        if not osp.isdir(self.procdir):
            os.makedirs(self.procdir)
            print('Processing', merged_npz, '-->', self.procdir)

            data, y = bbefp.dataset.data_pnet_as_akarray(merged_npz, nmax=n_events_max)
            n_events = len(y)
            n_features = data[0].shape[0]

            # Normalize per feature
            if lefts is None and rights is None:
                lefts = []
                rights = []
                for i_feature in range(n_features):
                    feature = data[:,i_feature,:].flatten()
                    q10 = np.quantile(feature, .05)
                    q90 = np.quantile(feature, .95)
                    fmin = np.min(feature)
                    fmax = np.max(feature)
                    left = q10 if abs(fmin/q10) < .7 else fmin
                    right = q90 if abs(q90/fmax) < .7 else fmax
                    print(
                        f'Normalizing feature {i_feature} from'
                        f'({left:.3f}, {right:.3f} --> (0, 1)'
                        )
                    lefts.append(left)
                    rights.append(right)
            self.lefts = lefts
            self.rights = rights

            for i in tqdm.tqdm(range(n_events), total=n_events):
                features = np.array(data[i].tolist())
                n_constituents_this = features.shape[1]

                # Normalize
                for i_feature in range(n_features):
                    features[i_feature] = (
                        (features[i_feature] - lefts[i_feature])
                        / (rights[i_feature] - lefts[i_feature])
                        )

                # Fix size
                if features.shape[1] > n_constituents:
                    # Remove constituents if too many
                    features = features[:,:n_constituents]
                else:
                    # Zero pad otherwise
                    needed_extra_zeroes = n_constituents - n_constituents_this
                    features = np.pad(features, ((0,0),(0,needed_extra_zeroes)), 'constant')

                mask = np.zeros((1, n_constituents))
                mask[:,:min(n_constituents_this, n_constituents)] = 1.

                assert features[:2].shape == (2, 200)

                # Make tensors and store
                tosave = dict(
                    coords = torch.from_numpy(features[:2].astype(np.float32)),
                    features = torch.from_numpy(features[2:].astype(np.float32)),
                    mask = torch.from_numpy(mask.astype(np.float32)),
                    y = torch.from_numpy(y[i:i+1].astype(np.long))
                    )
                torch.save(tosave, self.procdir + f'/data_{i}.pt')

            print('Done processing')

        self.proc_files = list(sorted(glob.iglob(osp.join(self.procdir, '*.pt'))))


    def __len__(self):
        return len(self.proc_files)

    def __getitem__(self, idx):
        return torch.load(self.proc_files[idx])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reproc', action='store_true')
    parser.add_argument('-n', '--nowrite', action='store_true')
    args = parser.parse_args()
    if args.reproc:
        ParticleNetDataset('data/train/merged.npz', force_reproc=True)
        ParticleNetDataset('data/test/merged.npz', force_reproc=True)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    model = ParticleNetTagger().to(device)

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        ParticleNetDataset('data/train/merged.npz'),
        batch_size=batch_size, shuffle=True
        )
    test_loader = torch.utils.data.DataLoader(
        ParticleNetDataset('data/test/merged.npz'),
        batch_size=batch_size, shuffle=True
        )

    epoch_size = len(train_loader.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = bbefp.networks.CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.1, policy="cosine")
    bbefp.networks.print_model_summary(model)


    nsig = 0
    for data in tqdm.tqdm(train_loader, total=len(train_loader)):
        nsig += data['y'].sum()
    s_over_n = float(nsig/len(train_loader.dataset))
    print('sig/total=', s_over_n)
    loss_weights = torch.tensor([s_over_n, 1.-s_over_n]).to(device)


    def train(epoch):
        print('Training epoch', epoch)
        model.train()
        scheduler.step()

        i = 0

        for data in tqdm.tqdm(train_loader, total=len(train_loader)):
            points = data['coords'].to(device)
            features = data['features'].to(device)
            mask = data['mask'].to(device)
            y = data['y'].squeeze().to(device)

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

            for i, data in enumerate(tqdm.tqdm(test_loader, total=len(test_loader))):
                points = data['coords'].to(device)
                features = data['features'].to(device)
                mask = data['mask'].to(device)
                y = data['y'].squeeze().to(device)

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
        if not args.nowrite:
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