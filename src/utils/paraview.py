import os
import re
import subprocess
from ntpath import basename
from os.path import join
from xml.etree import ElementTree as ET

from dotenv import load_dotenv


def update_states_data_path(BASE_PATH, DATA_PATH):
    """Update the DATA_PATH in Paraview state files."""
    states_folder = join(BASE_PATH, "src", "ParaviewStates")
    DATA_PATH = os.getenv("DATA_PATH")

    if not DATA_PATH:
        raise ValueError("Environment variable DATA_PATH is not set")

    pattern = re.compile(r"(DATA_PATH\s*=\s*)['\"].*?['\"]")

    for root, _, files in os.walk(states_folder):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, "r") as f:
                    content = f.read()
                new_content, n = pattern.subn(rf"\1'{DATA_PATH}'", content)
                if n > 0:
                    with open(filepath, "w") as f:
                        f.write(new_content)
                    print(f"Updated {n} assignment(s) in: {filepath}")


def pvsm_to_local_paths(pvsm_file: str, reverse: bool = False, output_dir: str = None) -> str:
    """
    Modifies paths in a ParaView .pvsm state file by substituting between a fixed
    container path ("/project/data/") and a local path defined in the DATA_PATH
    environment variable stored in a .env file.

    Parameters:
    ----------
    pvsm_file : str
        Path to the input .pvsm file to be processed.

    reverse : bool, default=False
        If False (default), replaces all occurrences of "/project/data/" with
        the value of DATA_PATH from the .env file.
        If True, performs the reverse substitution (i.e., replaces local paths
        with the fixed remote path).

    output_dir : str, optional
        Directory where the modified .pvsm file will be saved. If not provided,
        it defaults to a predefined output folder based on BASE_PATH.

    Returns:
    -------
    str
        Path to the updated .pvsm file.

    Raises:
    ------
    FileNotFoundError
        If the input .pvsm file does not exist.

    ValueError
        If DATA_PATH is not defined in the .env file.
    """

    # Validate input file exists
    if not os.path.isfile(pvsm_file):
        raise FileNotFoundError(f"Input file not found: {pvsm_file}")

    # Load environment variables
    load_dotenv()
    DATA_PATH = os.getenv("DATA_PATH")
    if not DATA_PATH:
        raise ValueError("DATA_PATH environment variable not found. Please check your .env file.")

    # Normalize paths with trailing slashes
    DATA_PATH = DATA_PATH.rstrip("/").rstrip("\\") + "/"
    fixed_path = "/project/data/"

    # Determine direction of substitution
    from_path = DATA_PATH if reverse else fixed_path
    to_path = fixed_path if reverse else DATA_PATH

    # Parse the .pvsm XML
    tree = ET.parse(pvsm_file)
    root = tree.getroot()

    # Replace paths in text, tail, and attributes
    for elem in root.iter():
        if elem.text and from_path in elem.text:
            elem.text = elem.text.replace(from_path, to_path)
        if elem.tail and from_path in elem.tail:
            elem.tail = elem.tail.replace(from_path, to_path)
        for key in elem.attrib:
            if from_path in elem.attrib[key]:
                elem.attrib[key] = elem.attrib[key].replace(from_path, to_path)

    # Save modified XML to new file
    if output_dir is None:
        output_dir = join(
            os.getenv("BASE_PATH", "./"), "data", "processed", "ParaviewStates", "initial_states"
        )

    if os.path.isdir(output_dir) is False:
        os.makedirs(output_dir, exist_ok=True)

    output_file = join(output_dir, basename(pvsm_file))
    tree.write(output_file)
    print(f"Updated file written to: {output_file}")
    return output_file


def create_pv_state(
    case_id: str,
    conda_path: str = "/opt/miniconda/bin/conda",
    conda_env: str = "paraview",
    output_dir: str = None,
) -> None:
    """
    Create a ParaView state file for the given case ID.
    This function runs a subprocess to execute a Python script that generates
    the state file using ParaView's Python environment.
    Args:
        case_id (str): The ID of the case for which to create the state file.
    """

    BASE_PATH = os.getenv("BASE_PATH", "./")
    if output_dir is None:
        output_dir = join(BASE_PATH, "data", "processed", "ParaviewStates", "container")
    result = subprocess.run(
        [
            conda_path,
            "run",
            "-n",
            conda_env,
            "python",
            "/project/src/ParaviewStates/GenerateStates.py",
            "--not_all",
            case_id,
            "--output_dir",
            output_dir,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    pvsm_file = join(output_dir, f"{case_id}.pvsm")
    if result.returncode != 0:
        print(f"Error creating state file for {case_id}: {result.stderr}")
    else:
        print(f"State file created for {case_id}: {pvsm_file}")
        pvsm_to_local_paths(pvsm_file, reverse=False)
