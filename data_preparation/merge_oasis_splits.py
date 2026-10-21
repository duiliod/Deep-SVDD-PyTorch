"""
Build the OASIS CN/AD split (following the same convention as the existing
local split hippocampi_holdout(both)NOROFFO10.yml: 1500 unique CN sessions
(both left+right) go to training, and the remaining matched sessions
(right side only) go to test), then merge it into each of the 5 server
split files hippocampi_holdout(both)NOROFFO20.yml.

This mirrors the logic of prepare_for_anomaly_detection_OASIS() in
src/data_preparation/anomaly_detection_data_preparation.py, but sourced
from the newly extracted structure images (one per structure/server folder)
instead of the pre-existing OASIS/left,right raw masks.
"""
import sys
import os
from os import path

import pandas as pd

sys.path.insert(0, "/home/duilio/Downloads/seg/src")
from trainer.utils import yaml as y

CLINICAL_XLSX = "/home/duilio/Downloads/seg/data/oasis_clinicalData2.xlsx"

STRUCTURES = {
    "THA": "/media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/deepsvdd_THA_IXI",
    "PAL": "/media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/deepsvdd_PAL_IXI",
    "PUT": "/media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/deepsvdd_PUT_IXI",
    "HIP": "/media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/deepsvdd_HIP_IXI",
    "AMY": "/media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/deepsvdd_AMY_IXI",
}

BASE_SPLIT_FILENAME = "hippocampi_holdout(both)NOROFFO20.yml"
N_CN_TRAINING_SESSIONS = 1500  # same number used in the local reference split


def get_cdr_lookup():
    df = pd.read_excel(CLINICAL_XLSX, sheet_name="ID_matching")
    lookup = {}
    for _, row in df.iterrows():
        lookup[(row["Subject"], row["ID"])] = row["Matched CDR"]
    return lookup


def basename_to_subject_session(basename: str):
    # basename like r_sub-OAS30579_ses-d1232_T1w (fixed offsets, same convention
    # used across the codebase for OASIS filenames)
    try:
        subject = basename[6:14]
        session = basename[19:24]
        return subject, session
    except Exception:
        return None, None


def build_oasis_split_for_structure(structure_key: str, structure_root: str, cdr_lookup: dict):
    data_dir = path.join(structure_root, "data", "hippocampi_images")

    # list all available OASIS base names (session-level) that have BOTH left and right
    # images actually extracted for this structure
    left_files = {f[:-8] for f in os.listdir(data_dir) if f.startswith("r_sub-OAS") and f.endswith("left.nii")}
    right_files = {f[:-9] for f in os.listdir(data_dir) if f.startswith("r_sub-OAS") and f.endswith("right.nii")}
    both_sides = sorted(left_files & right_files)

    # classify each session by CDR using the clinical excel
    cn_sessions, ad_sessions = [], []
    for basename in both_sides:
        subject, session = basename_to_subject_session(basename)
        cdr = cdr_lookup.get((subject, session))
        if cdr is None:
            continue
        if cdr == 0:
            cn_sessions.append(basename)
        elif cdr > 0.5:
            ad_sessions.append(basename)

    cn_sessions = sorted(cn_sessions)
    ad_sessions = sorted(ad_sessions)

    n_train = min(N_CN_TRAINING_SESSIONS, len(cn_sessions))
    train_sessions = cn_sessions[:n_train]
    # remaining CN sessions + all AD sessions go to test (right side only, same
    # convention as the local reference split)
    test_sessions = cn_sessions[n_train:] + ad_sessions

    training_names = [b + "left" for b in train_sessions] + [b + "right" for b in train_sessions]
    test_names = [b + "right" for b in test_sessions]

    print("[{}] CN sessions: {}, AD sessions: {} -> training {} (left+right), test {} (right only)".format(
        structure_key, len(cn_sessions), len(ad_sessions), len(train_sessions), len(test_sessions)))

    return training_names, test_names


def merge_into_base_split(structure_root: str, new_training: list, new_test: list):
    split_dir = path.join(structure_root, "split")
    base_split_path = path.join(split_dir, BASE_SPLIT_FILENAME)
    merged_split_path = path.join(split_dir, "hippocampi_holdout(both)NOROFFO20_withOASIS.yml")

    base = y.load(base_split_path)
    base_training = base["training"].split(",") if base["training"] else []
    base_validation = base["validation"].split(",") if base["validation"] else []
    base_test = base["test"].split(",") if base["test"] else []

    merged_training = base_training + new_training
    merged_test = base_test + new_test

    # keep the same "path" value as the base split (it points to the remote
    # server data path, e.g. /home/ddeangeli/deepsvdd/data/hippocampi_images,
    # since these splits are meant to be deployed there)
    data_path = base["path"]

    y.save_split_file_in_yaml(
        merged_split_path,
        data_path,
        merged_training,
        base_validation,
        merged_test,
        True,
    )

    print("  -> wrote {} (training {}, validation {}, test {})".format(
        merged_split_path, len(merged_training), len(base_validation), len(merged_test)))


def main():
    cdr_lookup = get_cdr_lookup()
    print("Loaded {} clinical records".format(len(cdr_lookup)))

    for structure_key, structure_root in STRUCTURES.items():
        training_names, test_names = build_oasis_split_for_structure(
            structure_key, structure_root, cdr_lookup
        )
        merge_into_base_split(structure_root, training_names, test_names)


if __name__ == "__main__":
    main()
