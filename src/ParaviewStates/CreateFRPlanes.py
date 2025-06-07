"""
Interpolates creates 5 planes perpendicular to max flow direction based on the sphere position in each structure,
computes flow rate in each plane, and saves results in a CSV. Always check the resulting paraview state
to ensure the planes are correctly positioned and then save using SaveFRState.py.

Author: Xabier Morales Ferez
"""

import csv
import os
import tempfile
from glob import glob
from os.path import basename, join
from pathlib import Path

import numpy as np
from paraview.simple import *

DATA_PATH = ""
not_all = []  # List of patients to process; if empty, all patients will be processed
rewrite = True  # If True, will overwrite existing CSV files
structure = [
    "RS",
    "RI",
    "LS",
    "LI",
    "MV",
    "LVOT",
]  # List of structures to process; can be modified as needed

if not DATA_PATH:
    raise ValueError(
        "DATA_PATH is not defined. Please set it manually in the macro before running the script."
    )

# Set up offscreen rendering
paraview.simple._DisableFirstRenderCameraReset()
os.environ["DISPLAY"] = ":99.0"  # Virtual display

paths = {
    "analysis": join(DATA_PATH, "analysis"),
    "processed": join(DATA_PATH, "processed"),
    "vtk": join(DATA_PATH, "processed", "segmentation", "vtk", "lv"),
    "states_input": join(DATA_PATH, "processed", "ParaviewStates", "sphere"),
    "states_output": join(DATA_PATH, "processed", "ParaviewStates", "cross_section"),
}

Path(paths["analysis"]).mkdir(parents=True, exist_ok=True)
Path(paths["states_output"]).mkdir(parents=True, exist_ok=True)

states_list = glob(join(paths["states_input"], "*.pvsm"))
if rewrite == False:
    states_list = [
        state
        for state in states_list
        if not os.path.isfile(join(paths["states_output"], basename(state)))
    ]


# %% Utility Functions
def delete_all():
    """Deletes all sources in the ParaView session."""
    while GetSources():
        try:
            Delete(next(iter(GetSources().values())))
        except:
            pass


def reset_session():
    """Resets the ParaView session."""
    pxm = servermanager.ProxyManager()
    pxm.UnRegisterProxies()
    Disconnect()
    Connect()


def compute_normal(source):
    # Create a temporary directory to store the data
    temp_dir = tempfile.TemporaryDirectory()
    temp = join(temp_dir.name, "temp.txt")

    # Compute the mean flow direction
    descriptiveStatistics1 = DescriptiveStatistics(Input=source, ModelInput=None)
    descriptiveStatistics1.AttributeMode = "Point Data"
    descriptiveStatistics1.VariablesofInterest = ["velocity"]
    descriptiveStatistics1.Task = "Detailed model of input data"
    descriptiveStatistics1.TrainingFraction = 0.1
    descriptiveStatistics1.Deviationsshouldbe = "Unsigned"

    # Save the data to a temporary file
    SaveData(
        temp,
        proxy=descriptiveStatistics1,
        WriteTimeSteps=1,
        WriteTimeStepsSeparately=1,
        Filenamesuffix="_%.3d",
        ChooseArraysToWrite=1,
        PointDataArrays=[],
        CellDataArrays=[],
        FieldDataArrays=[],
        VertexDataArrays=[],
        EdgeDataArrays=[],
        RowDataArrays=["Mean"],
        Precision=5,
        UseScientificNotation=0,
        FieldAssociation="Row Data",
        AddMetaData=0,
        AddTimeStep=0,
        AddTime=0,
    )

    # Get all the temporary files
    txt = glob(join(temp_dir.name, "*"))
    normal = []

    # Read the data from the temporary files
    for t in txt:
        with open(t) as fp:
            reader = csv.reader(fp, delimiter=",", quotechar='"')
            data_read = [row for row in reader]

        os.remove(t)
        normal += [[float(data_read[1][0]), float(data_read[2][0]), float(data_read[3][0])]]

    # Compute velocity magnitude for each value of normal
    mag = [np.sqrt(i[0] ** 2 + i[1] ** 2 + i[2] ** 2) for i in normal]

    # Get the the top 5 values
    norm_max_ind = sorted(range(len(mag)), key=lambda i: mag[i])[-5:]
    normal = [normal[i] for i in norm_max_ind]

    # Compute the mean and the unit normal vector of the top 5 values
    normal = np.mean(np.array(normal), axis=0)
    normal = normal / (np.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2))

    # Cleanup
    Delete(descriptiveStatistics1)
    del descriptiveStatistics1
    temp_dir.cleanup()

    return normal


def create_and_process_plane(resample, center, normal, name):
    """Creates a slice plane and measures flow rate."""
    # Create slice
    slice_plane = Slice(Input=resample, registrationName=f"Slice_{name}")
    slice_plane.SliceType = "Plane"
    slice_plane.SliceType.Origin = center
    slice_plane.SliceType.Normal = normal

    # Extract closest point region
    connectivity = Connectivity(Input=slice_plane)
    connectivity.ExtractionMode = "Extract Closest Point Region"
    connectivity.ClosestPoint = center
    connectivity.UpdatePipeline()

    # get active view
    renderView = GetActiveViewOrCreate("RenderView")
    connectivityDisplay = Show(connectivity, renderView, "GeometryRepresentation")
    connectivityDisplay.Representation = "Surface"
    ColorBy(connectivityDisplay, ("POINTS", "velocity", "Magnitude"))

    velocityLUT = GetColorTransferFunction("velocity").RescaleTransferFunction(0.0, 0.5)
    velocityPWF = GetOpacityTransferFunction("velocity").RescaleTransferFunction(0.0, 0.5)
    velocityTF2D = GetTransferFunction2D("velocity")

    # Measure flow rate
    surface_flow = SurfaceFlow(Input=connectivity)
    surface_flow.SelectInputVectors = ["POINTS", "velocity"]

    return connectivity, surface_flow


def save_state_and_cleanup(case_id, output_path):
    """Saves the current state and cleans up the session."""
    SaveState(join(output_path, f"{case_id}.pvsm"))
    delete_all()
    reset_session()


failed_cases = []
delete_all()

# %% Processing Loop
for state_file in states_list:
    try:
        LoadState(state_file)
        render_view = GetActiveViewOrCreate("RenderView")

        # Get volumetric mesh and associated data
        vti_lv = FindSource("vti_lv")
        vti_lv.UpdatePipeline()
        temp_interpolator = FindSource("tempInt_lv")
        case_id = basename(os.path.dirname(vti_lv.FileName[0]))

        vtk_mesh = LegacyVTKReader(FileNames=[join(paths["vtk"], f"{case_id}.vtk")])
        resample = ResampleWithDataset(SourceDataArrays=temp_interpolator, DestinationMesh=vtk_mesh)
        resample.UpdatePipeline()

        for structure_name in structure:
            source = FindSource(structure_name)
            center = source.ClipType.Center
            normal = compute_normal(source)

            connectivity, surface_flow = create_and_process_plane(
                resample, center, normal, structure_name
            )

            # Save flow rate data
            plot_data = PlotDataOverTime(Input=surface_flow)
            SaveData(
                join(paths["analysis"], f"{case_id}_{structure_name}.csv"),
                proxy=plot_data,
                RowDataArrays=["avg(Surface Flow)", "max(Surface Flow)"],
                FieldAssociation="Row Data",
            )

        Hide(temp_interpolator, render_view)
        save_state_and_cleanup(case_id, paths["states_output"])

    except Exception as e:
        delete_all()
        reset_session()
        failed_cases.append(case_id)
        print(f"Failed for case {case_id}: {str(e)}")

print("Failed cases:", failed_cases)
