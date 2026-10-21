#!/usr/bin/env python3
"""Pre-flight check: is everything needed to train present?

For every structure it verifies that
  * the split file exists and the data folder it points to exists,
  * every file listed in training/validation/test exists as .nii (and so does its
    left/right counterpart, which the pair loader always reads),
  * every listed file has a label in data/labels/all_labels_anomaly_detection.pkl.

Only needs PyYAML (no torch), so it can run on a login node:

    python tools/check_data.py                 # all structures
    python tools/check_data.py HIP THA         # some of them
    python tools/check_data.py --quick         # only folders and split files

Exit code is 0 when everything is fine, 1 otherwise.
"""
import argparse
import os
import pickle
import sys

import yaml

ALL_STRUCTURES = ['THA', 'PAL', 'PUT', 'HIP', 'AMY']
DATA_CONFIG = 'data_both_hippocampi_flippedSVDDDifferenceVolume.yml'
LABELS_FILE = 'all_labels_anomaly_detection.pkl'


def resolve(text, root, structure):
    return text.replace('{root}', root).replace('{structure}', structure)


def counterpart(name):
    if name.endswith('left'):
        return name[:-4] + 'right'
    if name.endswith('right'):
        return name[:-5] + 'left'
    return None


def split_names(value):
    return [n for n in (value or '').split(',') if n]


def check_structure(root, structure, labels, quick):
    problems = []
    with open(os.path.join(root, 'configs', DATA_CONFIG)) as f:
        splits_path = resolve(yaml.safe_load(f)['splits'], root, structure)
    if not os.path.isfile(splits_path):
        return ['split file not found: ' + splits_path], {}

    with open(splits_path) as f:
        split = yaml.safe_load(f)
    data_dir = resolve(split['path'], root, structure)
    if not os.path.isdir(data_dir):
        return ['data folder not found: ' + data_dir], {}

    counts = {}
    if quick:
        return problems, counts

    present = set(n[:-4] for n in os.listdir(data_dir) if n.endswith('.nii'))
    for part in ('training', 'validation', 'test'):
        names = split_names(split.get(part))
        counts[part] = len(names)
        missing = [n for n in names if n not in present]
        missing_pair = [n for n in names if counterpart(n) not in present]
        no_label = [n for n in names if n not in labels]
        for what, bad in (('missing file', missing), ('missing left/right counterpart', missing_pair),
                          ('no label', no_label)):
            if bad:
                problems.append('%s: %d %s (e.g. %s)' % (part, len(bad), what, bad[0]))
    return problems, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('structures', nargs='*', help='structures to check (default: all)')
    parser.add_argument('--quick', action='store_true', help='only check split files and data folders')
    parser.add_argument('--root', default=os.environ.get('DEEPSVDD_ROOT'),
                        help='repository root (default: $DEEPSVDD_ROOT or the parent of tools/)')
    args = parser.parse_args()
    unknown = [s for s in args.structures if s not in ALL_STRUCTURES]
    if unknown:
        parser.error('unknown structure(s): %s (valid: %s)' % (', '.join(unknown), ' '.join(ALL_STRUCTURES)))

    root = os.path.abspath(args.root or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    structures = args.structures or ALL_STRUCTURES
    print('root: %s' % root)

    labels_path = os.path.join(root, 'data', 'labels', LABELS_FILE)
    if not os.path.isfile(labels_path):
        print('ERROR: labels pickle not found: ' + labels_path)
        return 1
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    print('labels: %d entries' % len(labels))

    failed = 0
    for s in structures:
        problems, counts = check_structure(root, s, labels, args.quick)
        if problems:
            failed += 1
            print('%s: PROBLEMS' % s)
            for p in problems:
                print('    - ' + p)
        else:
            detail = ' train/val/test = %d/%d/%d' % (counts['training'], counts['validation'], counts['test']) if counts else ''
            print('%s: OK%s' % (s, detail))

    print('ALL OK' if not failed else '%d structure(s) with problems' % failed)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
