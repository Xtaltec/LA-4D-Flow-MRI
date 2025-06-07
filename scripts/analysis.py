import logging
import os
from os.path import join

from dotenv import load_dotenv

from src.utils.misc import join_create
from src.utils.parsing import parse_analysis
from src.utils.wall import compute_wall_parameters
from utils.analysis import compute_hemodynamic_parameters, flow_rate_to_R


def setup_analysis_paths(base_path):
    """Set up and return all required analysis and clinical data paths."""
    analysis_path = join(base_path, "data", "analysis")

    paths = {
        "vti_path": join(base_path, "data", "processed", "flow"),
        "analysis_path": join(base_path, "data", "analysis"),
        "fr_path": join(base_path, "data", "processed", "flow_rate"),
        "clinical_data": join(base_path, "data", "clinical", "clinical_data.xlsx"),
        "flow_metrics": join_create(analysis_path, "flow_metrics"),
        "feather_output": join_create(analysis_path, "feather"),
    }

    return paths


def main():
    load_dotenv()
    BASE_PATH = os.getenv("BASE_PATH", "./")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    args = parse_analysis()

    paths = setup_analysis_paths(BASE_PATH)

    # Compute flow rate and volume data and save to feather files for R processing
    flow_rate_to_R(paths, args)

    # Compute advanced hemodynamic parameters and save to feather files
    compute_hemodynamic_parameters(paths, args)

    # Compute wall-based parameters and save as .vtk files
    compute_wall_parameters(paths, args)


if __name__ == "__main__":
    main()
