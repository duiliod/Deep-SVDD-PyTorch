"""
Add the missing OASIS entries (introduced by merge_oasis_splits.py) to the
labels pickle consumed at training time by
src/trainer/DataModule/Filter/LoadImageLabel.py
(hardcoded path: /home/duilio/Downloads/seg/data/labels/all_labels_anomaly_detection.pkl).

merge_oasis_splits.py only builds the split YAML files; it does NOT touch the
labels pickle. Since LoadImageLabel.apply_one() does `dictLabels.get(filename)`
with no default, any sampled filename missing from the pickle returns None,
which crashes torch's default_collate as soon as it is sampled in a training
batch. This script fixes that by adding CDR-based labels (0=CN, 1=AD) for
every OASIS filename referenced in any of the 5 merged splits that is not
already present in the pickle.
"""
import sys
import os
import pickle

import pandas as pd

sys.path.insert(0, "/home/duilio/Downloads/seg/src")
from trainer.utils import yaml as y
from trainer.utils.string_processing import string_to_list

CLINICAL_XLSX = "/home/duilio/Downloads/seg/data/oasis_clinicalData2.xlsx"
LABELS_PKL = "/home/duilio/Downloads/seg/data/labels/all_labels_anomaly_detection.pkl"

STRUCTURES = {
    "THA": "/media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/deepsvdd_THA_IXI",
    "PAL": "/media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/deepsvdd_PAL_IXI",
    "PUT": "/media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/deepsvdd_PUT_IXI",
    "HIP": "/media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/deepsvdd_HIP_IXI",
    "AMY": "/media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/deepsvdd_AMY_IXI",
}

SPLIT_FILENAME = "hippocampi_holdout(both)NOROFFO20_withOASIS.yml"


def get_cdr_lookup():
    df = pd.read_excel(CLINICAL_XLSX, sheet_name="ID_matching")
    lookup = {}
    for _, row in df.iterrows():
        lookup[(row["Subject"], row["ID"])] = row["Matched CDR"]
    return lookup


def basename_to_subject_session(basename: str):
    # same fixed-offset convention used across the codebase for OASIS filenames
    return basename[6:14], basename[19:24]


def strip_side(filename: str):
    if filename.endswith("left"):
        return filename[:-4]
    if filename.endswith("right"):
        return filename[:-5]
    return filename


def main():
    with open(LABELS_PKL, "rb") as fp:
        labels = pickle.load(fp)
    print("Loaded {} existing label entries".format(len(labels)))

    cdr_lookup = get_cdr_lookup()
    print("Loaded {} clinical records".format(len(cdr_lookup)))

    all_filenames = set()
    for structure_key, structure_root in STRUCTURES.items():
        split_path = os.path.join(structure_root, "split", SPLIT_FILENAME)
        d = y.load(split_path)
        for key in ["training", "validation", "test"]:
            all_filenames.update(string_to_list(d[key]))

    missing = sorted(f for f in all_filenames if f not in labels)
    print("Found {} filenames (across all 5 structures) missing from the labels pickle".format(len(missing)))

    added, unmatched, inconsistent = 0, 0, 0
    for filename in missing:
        basename = strip_side(filename)
        subject, session = basename_to_subject_session(basename)
        cdr = cdr_lookup.get((subject, session))
        if cdr is None:
            unmatched += 1
            print("  WARNING: no CDR match for {} (subject={}, session={})".format(filename, subject, session))
            continue
        if cdr == 0:
            label = 0
        elif cdr > 0.5:
            label = 1
        else:
            inconsistent += 1
            print("  WARNING: ambiguous CDR={} for {}".format(cdr, filename))
            continue
        labels[filename] = label
        added += 1

    print("Added {} new label entries ({} unmatched, {} ambiguous CDR)".format(added, unmatched, inconsistent))
    print("New total label entries: {}".format(len(labels)))

    with open(LABELS_PKL, "wb") as fp:
        pickle.dump(labels, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved updated pickle to {}".format(LABELS_PKL))


if __name__ == "__main__":
    main()
