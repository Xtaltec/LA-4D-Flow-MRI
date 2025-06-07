import logging
import os
from glob import glob
from ntpath import basename
from os.path import join

from dotenv import load_dotenv
from natsort import natsorted

from src.utils.matrixops import visc_energy_loss
from src.utils.mesh import masking, meshing3D, nii_to_stl, read_vti, save_masked_evtk
from src.utils.misc import not_all_exclude
from src.utils.paraview import create_pv_state, update_states_data_path
from src.utils.parsing import parse_velocity

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH", "./")
DATA_PATH = os.getenv("DATA_PATH")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    """Main function to process 4D Flow MRI data."""
    args = parse_velocity()

    not_all = args.not_all
    exclude = args.exclude
    structures = args.structure
    crop = args.crop
    rewrite_stl = args.rewrite_stl
    rewrite_vtk = args.rewrite_vtk
    mu = args.mu

    processed_path = join(BASE_PATH, "data", "processed")
    flow_path = join(processed_path, "flow", "all")

    update_states_data_path(BASE_PATH, DATA_PATH)

    for structure in structures:
        mask_path, stl_path, vtk_path = setup_paths(processed_path, structure)
        cases = get_cases(mask_path, not_all, exclude)

        for case in cases:
            case_id = basename(case).split(".")[0]

            logging.info(f"Processing case: {case_id}")
            try:
                process_case(
                    case_id,
                    flow_path,
                    mask_path,
                    stl_path,
                    vtk_path,
                    crop,
                    rewrite_stl,
                    rewrite_vtk,
                    structure,
                    mu,
                )
            except Exception as e:
                logging.error(f"Error processing case {case_id}: {e}")
                continue


def setup_paths(processed_path, structure):
    """Set up and return all necessary paths."""
    mask_path = join(processed_path, "segmentation", "masks", structure)
    stl_path = join(processed_path, "segmentation", "stl", structure)
    vtk_path = join(processed_path, "segmentation", "vtk", structure)
    return mask_path, stl_path, vtk_path


def get_cases(mask_path, not_all, exclude):
    """Get the list of cases to process."""
    cases = natsorted(glob(join(mask_path, "*.nii.gz")))
    return not_all_exclude(cases, not_all, exclude)


def process_case(
    case_id,
    flow_path,
    mask_path,
    stl_path,
    vtk_path,
    crop,
    rewrite_stl,
    rewrite_vtk,
    structure,
    mu,
):
    """Process a single case."""
    # Read velocity data
    vel, spacing, points = read_vti(join(flow_path, case_id))

    # Read the mask and apply masking
    vel_masked, points_masked = masking(join(mask_path, case_id), vel, points, crop)

    # Pixel volume
    vol = spacing[0] * spacing[1] * spacing[2]

    # Compute the viscous energy loss
    VEL, _, _ = visc_energy_loss(
        vel_masked, spacing[:3], mu=mu, L=vol  # Use mu from the input argument
    )

    # Save the masked velocity
    save_masked_evtk(
        flow_path,
        case_id,
        vel_masked,
        points_masked,
        VEL,
        spacing,
        structure,
    )

    # Save the surface .stl
    nii_to_stl(mask_path, stl_path, case_id, rewrite=rewrite_stl)

    # Create the volumetric mesh
    meshing3D(stl_path, vtk_path, case_id, rewrite=rewrite_vtk)

    # Create ParaView state file
    if structure == "lv":
        create_pv_state(case_id)


if __name__ == "__main__":
    main()
