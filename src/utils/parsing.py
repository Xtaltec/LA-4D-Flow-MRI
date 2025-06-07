import argparse


def parse_pcmra():
    parser = argparse.ArgumentParser(
        description="Process 3D PC-MRA data and save it as .nii.gz "
        "for segmentation and visualization"
    )

    parser.add_argument(
        "--not_all",
        nargs="*",
        default=[],
        help="List of IDs of the only cases to process",
    )
    parser.add_argument(
        "--exclude", nargs="*", default=[], help="List of IDs of the cases to exclude"
    )
    parser.add_argument("--gamma", type=float, default=0.4, help="Gamma correction of the image")

    args, unknown = parser.parse_known_args()

    return args


def parse_velocity():
    parser = argparse.ArgumentParser(
        description="Process 4D flow MRI velocity data and save it as .vti for Paraview processing and visualization"
    )

    parser.add_argument(
        "--not_all",
        nargs="*",
        default=[],
        help="List of IDs of the only cases to process",
    )
    parser.add_argument(
        "--exclude", nargs="*", default=[], help="List of IDs of the cases to exclude"
    )
    parser.add_argument(
        "--structure",
        nargs="*",
        default=["la", "lv"],
        help="Define the structures to process",
    )
    parser.add_argument(
        "--crop",
        type=int,
        choices=[0, 1],
        default=1,
        help="Crop the non ROI to reduce memory requirements (0 or 1)",
    )
    parser.add_argument(
        "--rewrite_stl",
        type=int,
        choices=[0, 1],
        default=1,
        help="Rewrite the STL files (0 or 1)",
    )
    parser.add_argument(
        "--rewrite_vtk",
        type=int,
        choices=[0, 1],
        default=1,
        help="Rewrite the VTK files (0 or 1)",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.0035,
        help="Dynamic viscosity coefficient (default: 0.0035 Pa·s)",
    )

    args, unknown = parser.parse_known_args()

    args.crop = bool(args.crop)
    args.rewrite_stl = bool(args.rewrite_stl)
    args.rewrite_vtk = bool(args.rewrite_vtk)

    return args


def parse_train():
    parser = argparse.ArgumentParser(
        description="Process PC-MRA datasets and save results in the suitable format for segmentation with the nnUNet framework"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[],
        help="List of datasets to process",
    )
    parser.add_argument(
        "--structure",
        nargs="+",
        default=[],
        help="List of structures to include",
    )
    parser.add_argument(
        "--task_num",
        type=int,
        default=000,
        help="Task number required for nnUNet model training and inference",
    )
    parser.add_argument(
        "--task_base", type=str, default="MiaTest", help="Base name for the nnUNet training dataset"
    )

    args, unknown = parser.parse_known_args()

    return args


def parse_predict():
    parser = argparse.ArgumentParser(
        description="Define the model configuration for the nnUNet framework and the path to the data to be used for prediction"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Dataset100_ClinicalABI_la",
        help="Name of the trained model to be used for prediction",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Define the maximum number of retries to predict all the images",
    )
    parser.add_argument(
        "--path_to_predict",
        type=str,
        default="/project/data/processed/segmentation/pcmra",
        help="Path to the data to be used for prediction",
    )
    parser.add_argument(
        "--path_output",
        type=str,
        default="/project/data/processed/segmentation/nnunet/inference/predicted",
        help="Path where the output will be saved",
    )

    args, unknown = parser.parse_known_args()

    return args


def parse_states():
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Process 4D Flow MRI data and create sample volumes for PseudoDoppler analysis"
    )
    parser.add_argument(
        "--vti_structure",
        type=str,
        default="lv",
        help="Structure of the VTI files to analyze (default: 'lv')",
    )

    parser.add_argument(
        "--stl_structure",
        type=list,
        default=["la", "lv"],
        help="Structure/s  f the STL files to visualize on top of the VTI (default: ['la', 'lv'])",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=["RS", "RI", "LS", "LI", "MV", "LVOT"],
        help="Structures to process (default: ['RS', 'RI', 'LS', 'LI', 'MV', 'LVOT'])",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.005,
        help="Radius of the sample volume in meters (default: 0.005)",
    )
    parser.add_argument(
        "--sample_seeds",
        nargs=3,
        type=int,
        default=[20, 20, 20],
        help="Number of seeds in the sample volume (default: [20, 20, 20])",
    )
    parser.add_argument("--not_all", nargs="*", default=[], help="Specific cases to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing state files")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        nargs="?",
        help="Output directory for the Paraview state files",
    )

    return parser.parse_args()


def parse_analysis():
    parser = argparse.ArgumentParser(
        description="Parse arguments for time-resolved hemodynamic analysis from 4D flow MRI data"
    )

    parser.add_argument(
        "--id_column",
        type=str,
        default="name",
        help="Column name in the clinical data that contains subject IDs",
    )

    parser.add_argument(
        "--hr_column",
        type=str,
        default="HR_4D",
        help="Column name in the clinical data that contains heart rate",
    )

    parser.add_argument(
        "--clinical_columns",
        nargs="+",
        default=[],
        help="List of clinical data columns to include in the analysis",
    )

    parser.add_argument(
        "--not_all",
        nargs="+",
        default=[],
        help="List of subject IDs to include in the analysis (default: all subjects)",
    )

    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="List of subject IDs to exclude from analysis",
    )

    parser.add_argument(
        "--interp_num",
        type=int,
        default=30,
        help="Number of time steps to interpolate all cases to",
    )

    parser.add_argument(
        "--structure",
        nargs="+",
        default=["la"],
        help="List of anatomical structures to analyze (e.g., ['la', 'lv'])",
    )

    args, unknown = parser.parse_known_args()

    return args
