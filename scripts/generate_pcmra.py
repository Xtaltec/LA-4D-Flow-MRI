import logging
import os
from glob import glob
from ntpath import basename
from os.path import join

import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage
from dotenv import load_dotenv
from natsort import natsorted

from src.utils.data_readers import read_magnitude, read_velocity
from src.utils.matrixops import sp_to_mm
from src.utils.misc import join_create, not_all_exclude
from src.utils.parsing import parse_pcmra

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH", "./")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    """Main function to process PCMRA cases."""
    args = parse_pcmra()
    not_all = args.not_all
    exclude = args.exclude
    gamma = args.gamma

    path_data = join(BASE_PATH, "data")
    path_mag = join(path_data, "raw", "magnitude")
    path_vel = join(path_data, "raw", "velocity")
    path_an = join_create(path_data, "processed", "anatomy")
    path_flow = join_create(path_data, "processed", "flow", "all")
    path_img = join_create(path_data, "processed", "segmentation", "pcmra")

    mag_cases = glob(join(path_mag, "*"))
    vel_cases = glob(join(path_vel, "*"))

    cases = get_cases(mag_cases, vel_cases, not_all, exclude)

    for case in cases:
        logging.info(f"Processing case: {case}")
        try:
            process_case(case, path_mag, path_vel, path_an, path_flow, path_img, gamma)
        except Exception as e:
            logging.error(f"Error processing case {case}: {e}")


def get_cases(mag_cases, vel_cases, not_all, exclude):
    """Get the list of cases to process."""
    cases = [basename(i) for i in vel_cases]
    cases_mag = [basename(i) for i in mag_cases]
    cases = natsorted(list(set(cases) | set(cases_mag)))
    return not_all_exclude(cases, not_all, exclude)


def process_case(case, path_mag, path_vel, path_an, path_flow, path_img, gamma):
    """Process a single PCMRA case."""
    mag, mag_spacing = read_magnitude(case, path_mag, path_an)
    vel, spacing, origin = read_velocity(case, path_vel, path_flow, mag_spacing)
    vel_m = np.mean(np.linalg.norm(vel, axis=3), axis=3)

    if mag:
        mag_m = np.mean(mag.get_fdata(), axis=3)
        spacing = sp_to_mm(mag_spacing)
        mag_m, vel_m, spacing = match_shapes(mag_m, vel_m, spacing)
        pcmra = np.multiply(vel_m**gamma, mag_m)
        img = nib.Nifti1Image(pcmra, np.diag(tuple(spacing) + (1,)), header=mag.header)
    else:
        spacing = sp_to_mm(spacing)
        img = nib.Nifti1Image(vel_m, np.diag(spacing + (1,)))
        header_info = img.header
        header_info["pixdim"][1:4] = spacing

    img.to_filename(join(path_img, case + ".nii.gz"))


def match_shapes(mag_m, vel_m, spacing):
    """Match the shapes of magnitude and velocity data."""
    if all([j / i == 2 for i, j in zip(mag_m.shape, vel_m.shape)]):
        mag_m = ndimage.zoom(mag_m, 2, order=3)
        spacing = np.array(spacing) / 2
    elif mag_m.shape != vel_m.shape:
        mag_m, vel_m = crop_to_match(mag_m, vel_m)
    return mag_m, vel_m, spacing


def crop_to_match(mag_m, vel_m):
    """Crop magnitude and velocity data to match their shapes."""
    crop_mag = tuple(
        (slice((d - m) // 2, -(d - m) // 2 or None) if d > m else (slice(0, None, None)))
        for d, m in zip(mag_m.shape, vel_m.shape)
    )
    crop_vel = tuple(
        (slice((d - m) // 2, -(d - m) // 2 or None) if d > m else (slice(0, None, None)))
        for d, m in zip(vel_m.shape, mag_m.shape)
    )
    mag_m = mag_m[crop_mag]
    vel_m = vel_m[crop_vel]
    return mag_m, vel_m


if __name__ == "__main__":
    main()
