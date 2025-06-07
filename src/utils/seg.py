import json
import os
import subprocess
from glob import glob
from ntpath import basename
from os.path import isdir, join
from pathlib import Path
from shutil import copyfile
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from natsort import natsorted

from src.utils.misc import check_image, join_create


# %% Nifti to nnUNet processing
def nifti_to_nnUnet(
    path_images,
    path_masks,
    path_datasets,
    dataset_name,
    structure="",
    new_identifier=False,
    modality="0000",
):
    """Changes the name of all nifti files to the format required by nunet.
    It saves a .txt file with the pairings of old and new filenames. If folders
    have each a numeric identifier per case it is preserved otherways a new is
    provided automatically"""

    # Create output folder
    path_out = join_create(
        path_datasets, dataset_name + ("_" + structure if structure != "" else "")
    )

    # Generate image and mask output if valid path is provided
    path_nunet_i = (
        join_create(path_out, "images") if isdir(path_images) else print("No images provided")
    )
    path_nunet_m = (
        join_create(path_out, "masks") if isdir(path_masks) else print("No masks provided")
    )

    # Get all images and masks
    images = natsorted(glob(join(path_images, "*" + dataset_name + "_*.nii.gz")))
    masks = natsorted(glob(join(path_masks, structure, "*" + dataset_name + "_*.nii.gz")))

    # Get the IDs of the images and masks
    image_id = [basename(i).split(".")[0] for i in images]
    mask_id = [basename(m).split(".")[0] for m in masks]

    # Compute cases with both image and masks or only one of them
    common_id = sorted(list(set(mask_id) & set(image_id)))
    dif_im = sorted(list(set(image_id) - set(mask_id)))
    dif_mask = sorted(list(set(mask_id) - set(image_id)))

    # Create empty lists to store old and new names
    original_m, original_i, nunet_m, nunet_i = [], [], [], []
    counter = 0

    # Save common image and masks
    if len(common_id) > 0:
        for nid, cid in enumerate(common_id):
            # Find the image and mask paths
            i = [i for i in images if basename(i).split(".")[0] == cid][0]
            m = [m for m in masks if basename(m).split(".")[0] == cid][0]

            # If m is a folder, get the file inside
            m_file = glob(join(m, "*nii.gz")) if isdir(m) else m
            i_file = glob(join(i, "*nii.gz")) if isdir(i) else i

            # If there is more than one file in the folder, raise error
            check_image(i_file)
            check_image(m_file)

            # Get the new name
            i_file = i_file[0] if isinstance(i_file, list) else i_file
            m_file = m_file[0] if isinstance(m_file, list) else m_file

            # If case has a numeric identifier, use it
            case_id = cid if not new_identifier else dataset_name + "_" + str(nid).zfill(3)

            # Copy to output folder
            new_i = copyfile(m_file, join(path_nunet_m, case_id + ".nii.gz"))
            new_m = copyfile(i_file, join(path_nunet_i, case_id + "_" + modality + ".nii.gz"))

            # Save the original and new filenames
            original_m += [basename(m)]
            original_i += [basename(i)]
            nunet_m += [basename(new_i)]
            nunet_i += [basename(new_m)]

    counter += len(common_id)

    # Save images without masks
    if len(dif_im) > 0:
        for nid, cid in enumerate(dif_im):
            i = [i for i in images if basename(i).split(".")[0] == cid][0]
            i_file = glob(join(i, "*nii.gz")) if isdir(i) else i
            check_image(i_file)
            i_file = i_file[0] if isinstance(i_file, list) else i_file
            case_id = (
                cid if not new_identifier else dataset_name + "_" + str(nid + counter).zfill(3)
            )
            new_i = copyfile(i_file, join(path_nunet_i, case_id + "_" + modality + ".nii.gz"))
            original_i += [basename(i)]
            nunet_i += [basename(new_i)]

    counter += len(dif_im)

    # Save masks without images
    if len(dif_mask) > 0:
        for nid, cid in enumerate(dif_mask):
            m = [m for m in masks if basename(m).split(".")[0] == cid][0]
            m_file = glob(join(m, "*nii.gz")) if isdir(m) else m
            check_image(m_file)
            m_file = m_file[0] if isinstance(m_file, list) else m_file
            case_id = (
                cid if not new_identifier else dataset_name + "_" + str(nid + counter).zfill(3)
            )
            new_m = copyfile(m_file, join(path_nunet_m, case_id + ".nii.gz"))
            original_m += [basename(m)]
            nunet_m += [basename(new_m)]

    # Save the original and new filenames in a .txt file
    if len(masks) > 0:
        guide_m = pd.DataFrame({"Original": original_m, "nnUNet": nunet_m})
        guide_m.to_csv(
            join(path_nunet_m, "case_guide.txt"),
            header=None,
            index=None,
            sep=" ",
            mode="a",
        )
    if len(images) > 0:
        guide_i = pd.DataFrame({"Original": original_i, "nnUNet": nunet_i})
        guide_i.to_csv(
            join(path_nunet_i, "case_guide.txt"),
            header=None,
            index=None,
            sep=" ",
            mode="a",
        )


def nnUNet_dataset_generator(datasets, task_name, path_datasets, path_tasks, structure=""):
    path_out = join_create(path_tasks, task_name)

    # Create output paths
    path_tr_im, path_te_im, path_tr_la = (
        join(path_out, "imagesTr"),
        join(path_out, "imagesTs"),
        join(path_out, "labelsTr"),
    )
    Path(path_tr_im).mkdir(parents=True, exist_ok=True)
    Path(path_te_im).mkdir(parents=True, exist_ok=True)
    Path(path_tr_la).mkdir(parents=True, exist_ok=True)

    # List with with name guide; list with all cases; new id list
    guide, images, masks, new_id = np.empty((0, 2)), [], [], []

    for d in datasets:
        p = join(path_datasets, d + ("_" + structure if structure != "" else ""))
        images += glob(join(p, "images", "*.nii.gz"))
        masks += glob(join(p, "masks", "*.nii.gz"))
        guide = np.vstack(
            (
                guide,
                pd.read_csv(glob(join(p, "images", "*.txt"))[0], sep=" ", header=None).to_numpy(),
            )
        )

    # List with all original numeric identifiers and modalities
    all_id = [i.split(".")[0] for i in guide[:, 0]]  # All ID
    all_mod = [i[-11:-7] for i in guide[:, 1]]  # All modalities

    image_ori, image_nunet = [], []
    # Cases per case check if there is a mask for training if not move to testing
    for i, (im, c_id, c_mod) in enumerate(zip(images, all_id, all_mod)):
        mask = [m for m in masks if basename(m)[:-7] in im]

        new_name = task_name + "_" + str(i).zfill(3)
        new_id += [new_name]

        print("\n Case " + str(i))
        img_n = new_name + "_" + c_mod + ".nii.gz"
        msk_n = new_name + ".nii.gz"

        image_nunet += [new_name]
        image_ori += [c_id]

        # If image has a GT mask use to train
        if len(mask) == 1:
            print(mask[0])
            print(im)
            copyfile(mask[0], join(path_tr_la, msk_n))
            copyfile(im, join(path_tr_im, img_n))

        # Otherwise use for testing
        elif len(mask) == 0:
            print(im)
            copyfile(im, join(path_te_im, img_n))
        else:
            raise ValueError
    guide_n = pd.DataFrame({"Original": image_ori, "nnUNet": image_nunet})
    guide_n.to_csv(join(path_out, "case_guide.txt"), header=None, index=None, sep=" ", mode="a")


# %% nnUNet utils


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, "w") as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix=".nii.gz", join=False)])
    return uniques


def _just_filename(folder_path: str, filename: str) -> str:
    return filename


def subfiles(
    folder: str,
    join: bool = True,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    sort: bool = True,
) -> List[str]:
    """
    Get list of files in a folder with optional filtering and path joining.

    Args:
        folder: Path to the folder to list files from
        join: Whether to join folder path with filenames
        prefix: Optional prefix filter for filenames
        suffix: Optional suffix filter for filenames
        sort: Whether to sort the results

    Returns:
        List of filenames or file paths
    """
    if join:
        path_func: Callable[[str, str], str] = os.path.join
    else:
        path_func = _just_filename
        # path_func = lambda folder_path, filename: filename

    res = [
        path_func(folder, filename)
        for filename in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, filename))
        and (prefix is None or filename.startswith(prefix))
        and (suffix is None or filename.endswith(suffix))
    ]

    if sort:
        res.sort()

    return res


def generate_dataset_json(
    output_folder: str,
    task_name: str,
    overwrite_image_reader_writer: str = None,
    **kwargs,
):
    description = ""
    channel_names = {"0": "M"}
    labels = {"background": 0, "la": 1}
    num_training_cases = len(glob(join(output_folder, task_name, "imagesTr", "*.nii.gz")))
    file_ending = ".nii.gz"
    license = "hands off!"
    release = "0.0"
    reference = ""
    regions_class_order = None

    has_regions: bool = any([isinstance(i, (tuple, list)) and len(i) > 1 for i in labels.values()])
    if has_regions:
        assert (
            regions_class_order is not None
        ), f"You have defined regions but regions_class_order is not set. You need that. Got: {regions_class_order}"

    keys = list(channel_names.keys())
    for k in keys:
        if not isinstance(k, str):
            channel_names[str(k)] = channel_names[k]
            del channel_names[k]

    # labels need ints as values
    for lab in labels.keys():
        value = labels[lab]
        if isinstance(value, (tuple, list)):
            value = tuple([int(i) for i in value])
            labels[lab] = value
        else:
            labels[lab] = int(labels[lab])

    dataset_json = {
        "channel_names": channel_names,
        "labels": labels,
        "numTraining": num_training_cases,
        "file_ending": file_ending,
    }

    if task_name is not None:
        dataset_json["name"] = task_name
    if reference is not None:
        dataset_json["reference"] = reference
    if release is not None:
        dataset_json["release"] = release
    if license is not None:
        dataset_json["licence"] = license
    if description is not None:
        dataset_json["description"] = description
    if overwrite_image_reader_writer is not None:
        dataset_json["overwrite_image_reader_writer"] = overwrite_image_reader_writer
    if regions_class_order is not None:
        dataset_json["regions_class_order"] = regions_class_order

    dataset_json.update(kwargs)

    save_json(dataset_json, join(output_folder, task_name, "dataset.json"), sort_keys=False)


# Check if case has a mask otherwise copy to to_predict
def copy_testing_images(path_images, path_masks, path_inference, structure):
    images = glob(join(path_images, "*.nii.gz"))
    masks = glob(join(path_masks, structure, "*.nii.gz"))

    Path(path_inference, structure).mkdir(parents=True, exist_ok=True)

    for im in images:
        case_id = basename(im).split(".")[0]
        ma = [i for i in masks if case_id in basename(i).split(".")[0]]

        if len(ma) == 0:
            copyfile(
                im,
                join(
                    path_inference,
                    structure,
                    basename(im).replace(".nii.gz", "_0000.nii.gz"),
                ),
            )


# %% Training and testing functions
def training(task_name):
    bash_path = "./src/nnunet/training.sh"

    # Ensure the script has execute permissions
    os.chmod(bash_path, 0o775)

    command = f"{bash_path} {task_name}"

    # Execute the model training
    subprocess.run(command, check=True, shell=True)


def prediction(task_name, path_in, path_out):
    # Ensure the script has execute permissions
    bash_path = "./src/nnunet/predict.sh"
    os.chmod(bash_path, 0o775)

    command = f"{bash_path} {task_name} {path_in} {path_out}"

    # Execute the model training
    subprocess.run(command, check=True, shell=True)
