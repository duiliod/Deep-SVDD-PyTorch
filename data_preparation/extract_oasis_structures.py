"""
Extract Thalamus, Pallidum, Putamen, Hippocampus and Amygdala (left/right) from
FastSurfer aseg.auto_noCCseg.mgz outputs for the OASIS cohort, preprocess them
exactly like anomaly_detection_data_preparation.py does (crop 64 cube around the
bounding box center + resize to 64x64x64), and save them directly into each
server deepsvdd_*_IXI/data/hippocampi_images folder.

FreeSurfer labels used (left / right):
    Thalamus:    10 / 49
    Putamen:     12 / 51
    Pallidum:    13 / 52
    Hippocampus: 17 / 53
    Amygdala:    18 / 54

Usage:
    python3 extract_oasis_structures.py
"""
import os
import sys
from os import path

import numpy as np
import nibabel as nb

sys.path.insert(0, "/home/duilio/Downloads/seg/src")
from utils_hippocampus.util import (
    crop_hipp_image_around_boundig_box,
    resize_image_to_match_hippocampal_boundig_box,
)

FASTSURFER_OUTPUT_DIR = "/media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/OASIS_fastsurfer_output"
ASEG_FILENAME = path.join("mri", "aseg.auto_noCCseg.mgz")

# label_left, label_right, min_voxels_to_keep (sanity threshold)
STRUCTURES = {
    "THA": {"name": "Thalamus", "left": 10, "right": 49, "min_voxels": 200,
            "out_dir": "/media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/deepsvdd_THA_IXI/data/hippocampi_images"},
    "PAL": {"name": "Pallidum", "left": 13, "right": 52, "min_voxels": 100,
            "out_dir": "/media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/deepsvdd_PAL_IXI/data/hippocampi_images"},
    "PUT": {"name": "Putamen", "left": 12, "right": 51, "min_voxels": 200,
            "out_dir": "/media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/deepsvdd_PUT_IXI/data/hippocampi_images"},
    "HIP": {"name": "Hippocampus", "left": 17, "right": 53, "min_voxels": 500,
            "out_dir": "/media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/deepsvdd_HIP_IXI/data/hippocampi_images"},
    "AMY": {"name": "Amygdala", "left": 18, "right": 54, "min_voxels": 100,
            "out_dir": "/media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/deepsvdd_AMY_IXI/data/hippocampi_images"},
}

IMAGE_SIZE = (64, 64, 64)
TOLERANCE = 0.25


def already_processed(sid: str) -> bool:
    """Check if this subject was already extracted for all structures (idempotency
    check used to make repeated incremental runs cheap)."""
    for cfg in STRUCTURES.values():
        left_path = path.join(cfg["out_dir"], "{}left.nii".format(sid))
        right_path = path.join(cfg["out_dir"], "{}right.nii".format(sid))
        # if neither side exists for this structure, it was not processed yet
        # (it's fine if a side is legitimately missing due to TOO_SMALL, but we
        # can't distinguish that cheaply here, so we only skip subjects where at
        # least the hippocampus - most reliable structure - was already written)
        if cfg is STRUCTURES["HIP"] and not (path.isfile(left_path) or path.isfile(right_path)):
            return False
    return True


def process_subject(subject_dir: str, log_fh):
    sid = path.basename(subject_dir.rstrip("/"))

    if already_processed(sid):
        return None

    aseg_path = path.join(subject_dir, ASEG_FILENAME)
    if not path.isfile(aseg_path):
        log_fh.write("MISSING_ASEG\t{}\n".format(sid))
        return False

    try:
        img = nb.load(aseg_path)
        data = img.get_fdata()
        affine = img.affine
    except Exception as exc:
        log_fh.write("LOAD_ERROR\t{}\t{}\n".format(sid, exc))
        return False

    any_saved = False
    for struct_key, cfg in STRUCTURES.items():
        for side, label in (("left", cfg["left"]), ("right", cfg["right"])):
            mask = (data == label)
            nonzero = int(np.count_nonzero(mask))
            if nonzero < cfg["min_voxels"]:
                log_fh.write("TOO_SMALL\t{}\t{}\t{}\t{}\n".format(sid, struct_key, side, nonzero))
                continue

            raw_struct_image = nb.Nifti1Image(mask.astype(float), affine)
            try:
                cropped = crop_hipp_image_around_boundig_box(raw_struct_image, tolerance=TOLERANCE)
                resized = resize_image_to_match_hippocampal_boundig_box(cropped, IMAGE_SIZE)
            except Exception as exc:
                log_fh.write("PREPROC_ERROR\t{}\t{}\t{}\t{}\n".format(sid, struct_key, side, exc))
                continue

            out_filename = "{}{}.nii".format(sid, side)
            out_path = path.join(cfg["out_dir"], out_filename)
            nb.save(resized, out_path)
            any_saved = True

    return any_saved


def main():
    makedirs_needed = {cfg["out_dir"] for cfg in STRUCTURES.values()}
    for d in makedirs_needed:
        os.makedirs(d, exist_ok=True)

    subjects = sorted(
        d for d in os.listdir(FASTSURFER_OUTPUT_DIR)
        if path.isdir(path.join(FASTSURFER_OUTPUT_DIR, d)) and d.startswith("r_sub-OAS")
    )
    print("Found {} FastSurfer subject output folders".format(len(subjects)))

    log_path = "/media/duilio/d24a1167-a40e-45c3-a3c7-5ea5da09d1ca/server/extract_oasis_structures.log"
    processed, skipped, already_done = 0, 0, 0
    with open(log_path, "a") as log_fh:
        for i, sid in enumerate(subjects):
            subject_dir = path.join(FASTSURFER_OUTPUT_DIR, sid)
            ok = process_subject(subject_dir, log_fh)
            if ok is None:
                already_done += 1
            elif ok:
                processed += 1
            else:
                skipped += 1
            if (i + 1) % 200 == 0:
                print("Progress {}/{} ({} newly processed, {} skipped, {} already done)".format(
                    i + 1, len(subjects), processed, skipped, already_done))
                log_fh.flush()

    print("DONE. Newly processed {}, skipped {}, already done {}.".format(
        processed, skipped, already_done))


if __name__ == "__main__":
    main()
