#!/usr/bin/env python3
"""Evaluate a trained Deep SVDD checkpoint on the test split, broken down by data source.

The AUC logged during training uses the validation split (normal ROFFO scans that are also in
the training set versus synthetic atrophy) and says little about how well CN and AD are
separated. This script uses the real test split and reports, per source (OASIS / ADNI / HEC)
and overall (synthetic scans excluded), the AUC of:

  dist-to-c    ||f(x) - c||^2 with the center c saved in the checkpoint (checkpoints written
               before c was saved do not have it)
  mahalanobis  distance of f(x) to the training distribution of f (needs no center)
  volume       lower summed volume of both structures = more anomalous
  asymmetry    |L - R| / (L + R)
  vol+asym     both combined (z-scored with training statistics)

The last three need no trained network: they are the reference a learned model has to beat.

Run it from src/trainer (paths are relative to it):

    cd src/trainer
    python ../../tools/evaluate_svdd.py --structure HIP
    python ../../tools/evaluate_svdd.py --structure HIP --results ../../runs/HIP/results
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'trainer'))

import DataModule  # noqa: E402
import Models  # noqa: E402
from utils import yaml  # noqa: E402

CONFIG = 'AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDD.yml'
SOURCES = [('r_sub-OAS', 'OASIS'), ('r_ADNI', 'ADNI'), ('r_HEC', 'HEC'), ('SYNTHETIC', 'SYNTH'),
           ('r_IXI', 'IXI'), ('r_ROFFO', 'ROFFO')]


def source(name):
    for prefix, label in SOURCES:
        if name.startswith(prefix):
            return label
    return 'OTHER'


class Names(torch.utils.data.Dataset):
    def __init__(self, dataset, names):
        self.dataset, self.names = dataset, names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return self.dataset.get(self.names[i])


@torch.no_grad()
def run(net, dataset, names, workers, device):
    loader = torch.utils.data.DataLoader(Names(dataset, names), batch_size=24, num_workers=workers)
    outputs, labels, volumes = [], [], []
    for x, y in loader:
        outputs.append(net([x[0].to(device), x[1].to(device), x[2].to(device)]).cpu())
        labels.append(y[0].numpy())
        volumes.append(torch.stack([x[1], x[2]], 1).numpy())
    return (torch.cat(outputs).numpy().astype(np.float64), np.concatenate(labels),
            np.concatenate(volumes).astype(np.float64))


def mahalanobis(train_out, test_out, lam=1e-3):
    cov = train_out.T @ train_out / len(train_out)
    cov = cov + lam * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    return np.einsum('ij,jk,ik->i', test_out, np.linalg.inv(cov), test_out)


def auc(y, score, mask):
    return roc_auc_score(y[mask], score[mask]) if mask.sum() and len(set(y[mask])) > 1 else float('nan')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--structure', required=True, choices=['THA', 'PAL', 'PUT', 'HIP', 'AMY'])
    parser.add_argument('--config', default=CONFIG, help='SVDD config file in configs/ (default: %(default)s); '
                        'its name (without .yml) gives the results folder <structure>_<name>')
    parser.add_argument('--checkpoint', default=None, help='SVDD checkpoint (default: the latest last.ckpt)')
    parser.add_argument('--results', default=os.path.join(ROOT, 'src', 'results'), help='results folder')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        exp = os.path.splitext(args.config)[0]
        pattern = os.path.join(args.results, '%s_%s' % (args.structure, exp), 'version_*', 'checkpoints', 'last.ckpt')
        found = glob.glob(pattern)
        if not found:
            sys.exit('no checkpoint found: ' + pattern)
        ckpt_path = max(found, key=os.path.getmtime)
    print('checkpoint:', ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    center = ckpt.get('svdd_center')

    cfg = yaml.load(os.path.join(ROOT, 'configs', args.config), structure=args.structure)
    dm = DataModule.get(cfg['DataModule'])
    dm.setup()
    net = Models.get_SVDD(cfg['Model'])
    net.load_state_dict({k[len('_model.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('_model.')})
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device).eval()

    # right entries only: the order used at test time (the network is antisymmetric)
    train_names = [n for n in dm._train_data.filenames if n.endswith('right')]
    test_names = [n for n in dm._test_data.filenames if not n.startswith('SYNTHETIC')]
    print('train entries: %d | test entries (no synthetic): %d' % (len(train_names), len(test_names)))
    out_tr, _, vol_tr = run(net, dm._train_data, train_names, args.workers, device)
    out_te, y, vol_te = run(net, dm._test_data, test_names, args.workers, device)
    src = np.array([source(n) for n in test_names])

    feat = lambda v: np.stack([v.sum(1), np.abs(v[:, 0] - v[:, 1]) / v.sum(1)], 1)
    f_tr, f_te = feat(vol_tr), feat(vol_te)
    z = (f_te - f_tr.mean(0)) / f_tr.std(0)

    scores = {}
    if center is not None:
        scores['dist-to-c'] = ((out_te - center.numpy()) ** 2).sum(1)
    else:
        print('NOTE: this checkpoint has no saved center, dist-to-c is not available')
    scores['mahalanobis'] = mahalanobis(out_tr, out_te)
    scores['volume'] = -z[:, 0]
    scores['asymmetry'] = z[:, 1]
    scores['vol+asym'] = -z[:, 0] + z[:, 1]

    print('network output on train: mean norm %.4g, std across samples %.4g'
          % (np.linalg.norm(out_tr, axis=1).mean(), out_tr.std(0).mean()))
    groups = ['OASIS', 'ADNI', 'HEC']
    print('\n%-12s %s   ALL' % ('AUC', ' '.join('%-6s' % g for g in groups)))
    for name, s in scores.items():
        print('%-12s %s   %.3f' % (name, ' '.join('%.3f ' % auc(y, s, src == g) for g in groups), roc_auc_score(y, s)))
    for g in groups:
        m = src == g
        print('%s: n=%d (CN %d, anomalous %d)' % (g, m.sum(), (y[m] == 0).sum(), (y[m] == 1).sum()))


if __name__ == '__main__':
    main()

def auc(y, score, mask):
    return roc_auc_score(y[mask], score[mask]) if mask.sum() and len(set(y[mask])) > 1 else float('nan')
