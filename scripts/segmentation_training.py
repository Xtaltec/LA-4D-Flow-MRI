import logging
import os
from os.path import join

from dotenv import load_dotenv

from src.utils.misc import join_create
from src.utils.parsing import parse_train
from src.utils.seg import generate_dataset_json, nifti_to_nnUnet, nnUNet_dataset_generator, training

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH", "./")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_cohort_datasets(datasets, structure, path_images, path_masks, path_datasets):
    """Generate cohort datasets in nnUNet format."""
    for cohort in datasets:
        for struct in structure:
            nifti_to_nnUnet(path_images, path_masks, path_datasets, cohort, struct)


def mix_nnunet_datasets(datasets, structure, task_num, task_base, path_datasets, path_tasks):
    """Mix nnUNet datasets and generate dataset JSON."""
    for i, struct in enumerate(structure):
        task_name = f"Dataset{task_num + i}_{task_base}" + ("_" + struct if struct != "" else "")
        nnUNet_dataset_generator(tuple(datasets), task_name, path_datasets, path_tasks, struct)
        generate_dataset_json(path_tasks, task_name)
    return task_name


def main():
    args = parse_train()
    datasets = args.datasets
    structure = args.structure
    task_num = args.task_num
    task_base = args.task_base

    path_data = join(BASE_PATH, "data")
    path_images = join_create(path_data, "processed", "segmentation", "pcmra")
    path_masks = join_create(path_data, "processed", "segmentation", "masks")
    path_datasets = join_create(
        path_data, "processed", "segmentation", "nnunet", "training", "datasets"
    )
    path_tasks = join_create(path_data, "processed", "segmentation", "nnunet", "training", "tasks")

    generate_cohort_datasets(datasets, structure, path_images, path_masks, path_datasets)
    task_name = mix_nnunet_datasets(
        datasets, structure, task_num, task_base, path_datasets, path_tasks
    )

    training(task_name)


if __name__ == "__main__":
    main()
