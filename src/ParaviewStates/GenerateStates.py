"""
This script automatically processes 4D Flow MRI data to load the processed VTI, STL and VTK files,
and saves results as Paraview state files for further analysis. It generates sample volumes with
spheres positioned at defined locations that should be manually locate at the center of the
structures of interest.

"""

import logging
import os
from glob import glob
from ntpath import basename
from os.path import join
from pathlib import Path

import paraview
from dotenv import load_dotenv
from paraview.simple import *

from src.utils.parsing import parse_states

# Set up offscreen rendering
paraview.simple._DisableFirstRenderCameraReset()
os.environ["DISPLAY"] = ":99.0"  # Virtual display

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH", "./")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def setup_paths(base_path):
    """Set up and return all required paths."""
    processed_path = join(base_path, "data", "processed")
    paths = {
        "flow": join(processed_path, "flow"),
        "stl": join(processed_path, "segmentation", "stl"),
        "states": join(processed_path, "ParaviewStates", "container"),
    }
    return paths


def get_cases(args, paths):
    """Filter cases based on existing state files and user-specified conditions."""
    cases = glob(join(paths["flow"], args.vti_structure, "*"))
    if args.not_all:
        cases = [case for case in cases if basename(case) in args.not_all]
    if not args.overwrite:
        cases = [
            case
            for case in cases
            if not os.path.isfile(join(paths["states"], "pv_states", basename(case) + ".pvsm"))
        ]
    return cases


def delete_all():
    """Delete all existing Paraview sources."""
    sources = GetSources()
    while sources:
        try:
            Delete(list(sources.values())[0])
        except:
            pass
        sources = GetSources()


def reset_session():
    """Reset the current Paraview session."""
    pxm = servermanager.ProxyManager()
    pxm.UnRegisterProxies()
    del pxm
    Disconnect()
    Connect()


def create_sample_volumes(case_path, args, paths):
    """Create sample volumes for the specified case and save the state file."""

    case_id = basename(case_path)
    logging.info(f"Processing case: {case_id}")
    try:
        renderView = GetActiveViewOrCreate("RenderView")

        vti_structure = args.vti_structure
        stl_structure = args.stl_structure
        samples = args.samples
        radius = args.radius
        sample_seeds = args.sample_seeds

        # Load VTI data
        vti_list = glob(join(case_path, "*.vti"))
        globals()["vti_" + vti_structure] = XMLImageDataReader(
            registrationName="vti_" + vti_structure, FileName=vti_list
        )
        globals()["tempInt_" + vti_structure] = TemporalInterpolator(
            registrationName="tempInt_" + vti_structure, Input=globals()["vti_" + vti_structure]
        )
        globals()["tempInt_" + vti_structure].DiscreteTimeStepInterval = 1

        # Render VTI data
        LoadPalette(paletteName="WhiteBackground")
        vti_display = Show(
            globals()["tempInt_" + vti_structure], renderView, "UniformGridRepresentation"
        )
        vti_display.SetRepresentationType("Volume")

        ColorBy(vti_display, ("POINTS", "velocity", "Magnitude"))
        GetColorTransferFunction("velocity").ApplyPreset("Jet", True)
        GetColorTransferFunction("velocity").RescaleTransferFunction(0.0, 0.5)
        GetOpacityTransferFunction("velocity").RescaleTransferFunction(0.0, 0.5)
        GetTransferFunction2D("velocity").RescaleTransferFunction(0.0, 0.5, 0.0, 1.0)

        # Determine reference structure for sample positions
        ref_loc = "la" if "la" in stl_structure else vti_structure

        # Load STL files
        for st in stl_structure:
            stl = glob(join(paths["stl"], st, f"*{case_id}*"))
            globals()["stl_" + st] = STLReader(registrationName="stl_" + st, FileNames=stl)

            stl_display = Show(globals()["stl_" + st], renderView, "GeometryRepresentation")
            stl_display.Representation = "Surface"
            ColorBy(stl_display, None)
            stl_display.Opacity = 0.5

            if st == ref_loc:
                # Use the reference structure to define sample positions
                x_min, x_max, y_min, y_max, z_min, z_max = (
                    globals()["stl_" + st].GetDataInformation().GetBounds()
                )
                pos = [
                    [x_max, y_max, z_max],
                    [x_max, y_min, z_max - (z_max - z_min) / 2],
                    [x_min, y_max, z_max],
                    [x_min, y_min, z_max - (z_max - z_min) / 2],
                    [x_min, y_max - (y_max - y_min) / 2, z_min],
                    [x_max - x_min / 2, y_max, z_min],
                ]

        # Generate sample volumes
        for num, struct in enumerate(samples):
            sample_clip = Clip(registrationName=struct, Input=globals()["tempInt_" + vti_structure])
            sample_clip.ClipType = "Sphere"
            sample_clip.ClipType.Radius = radius
            sample_clip.ClipType.Center = pos[num]

            resample = ResampleToImage(registrationName=f"resample_{struct}", Input=sample_clip)
            resample.SamplingDimensions = sample_seeds

            threshold = Threshold(registrationName=f"threshold_{struct}", Input=resample)
            threshold.Scalars = ["POINTS", "velocity_magnitude"]
            threshold.LowerThreshold, threshold.UpperThreshold = (1e-6, 1000.0)

            calculator = Calculator(registrationName=f"calc_{struct}", Input=threshold)
            calculator.ResultArrayName = "normal"
            calculator.Function = "dot(velocity,1*iHat+1*jHat+1*kHat)"

        # Save Paraview state
        SaveState(join(paths["states"], case_id + ".pvsm"))
        logging.info(f"Saved state file for case: {case_id}")

    except Exception as e:
        logging.error(f"Error processing case {case_id}: {e}")


def main():
    args = parse_states()
    paths = setup_paths(BASE_PATH)
    cases = get_cases(args, paths)

    if args.output_dir:
        paths["states"] = args.output_dir

    Path(paths["states"]).mkdir(parents=True, exist_ok=True)

    for case_path in cases:
        create_sample_volumes(case_path, args, paths)

        delete_all()
        reset_session()


if __name__ == "__main__":
    main()
