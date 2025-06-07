"""
Script to save the Paraview state after positioning the sphere in each structure from the initial Paraview states.

"""

from pathlib import Path

from paraview.simple import FindSource, SaveState

# %% Define the paths
DATA_PATH = ""

if not DATA_PATH:
    raise ValueError(
        "DATA_PATH is not defined. Please set it manually in the macro before running the script."
    )

# Define the output path
states_path = Path(DATA_PATH, "processed", "ParaviewStates", "sphere")
states_path.mkdir(parents=True, exist_ok=True)

# Get image source
vti = FindSource("vti_lv")
if not vti:
    print("No source matching 'vti_*' found in the pipeline.")
    raise RuntimeError("No source found in the pipeline.")

# Get the case identifier
case_id = Path(vti.FileName[0]).parent.name

# Save the state
output_file = states_path / f"{case_id}.pvsm"
SaveState(str(output_file))
