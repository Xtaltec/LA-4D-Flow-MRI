import os
import re
from glob import glob
from ntpath import basename
from os.path import join
from pathlib import Path

import pandas as pd
from paraview.simple import *

DATA_PATH = ""
not_all = []  # List of patients to process; if empty, all patients will be processed
structure = "lv"

if not DATA_PATH:
    raise ValueError(
        "DATA_PATH is not defined. Please set it manually in the macro before running the script."
    )


def delete_all():
    sources = GetSources()

    while len(GetSources()) >= 1:
        try:
            Delete(list(sources.values())[0])
            del list(sources.values())[0]
        except:
            pass
        sources = GetSources()


def update_pipeline():
    sources = GetSources().values()

    for s in sources:
        try:
            UpdatePipeline(None, s)
        except:
            pass


states_path = Path(DATA_PATH, "processed", "ParaviewStates")
fr_path = join(states_path, "flow_rate")
anim_path = join(states_path, "animations")

Path(anim_path).mkdir(parents=True, exist_ok=True)

# Load clinical data safely
clinical_data = None
clinical_data_path = join(DATA_PATH, "clinical_data", "clinical_data.csv")
if os.path.exists(clinical_data_path):
    try:
        clinical_data = pd.read_csv(clinical_data_path)
    except Exception as e:
        print(f"Error loading clinical data: {e}")
        clinical_data = None

# List of all patients
patients = glob(join(fr_path, "*.pvsm*"))
not_all = []

if not_all:
    patients = [i for i in patients if basename(i).split(".")[0] in not_all]

for p in patients:
    case_id = basename(p).split(".")[0]
    LoadState(p)
    UpdateScalarBars()

    # Delete LV source
    lv = FindSource(f"vti_{structure}")
    timesteps = len(lv.FileName)
    renderView1 = GetActiveViewOrCreate("RenderView")

    # Default heart rate
    HR = 60

    # If clinical data is available, attempt to get HR
    if clinical_data is not None:
        hr_row = clinical_data[clinical_data["name"] == case_id]
        if not hr_row.empty and not pd.isna(hr_row["HR_4D"].values[0]):
            HR = hr_row["HR_4D"].values[0]

    shift = 60 / HR * 1 / (timesteps - 1)

    # get the time-keeper and update the animation scene
    timeKeeper1 = GetTimeKeeper()
    animationScene1 = GetAnimationScene()
    animationScene1.AnimationTime = 0.0
    animationScene1.UpdateAnimationUsingDataTimeSteps()

    # create a new 'Temporal Shift Scale'
    temporalShiftScale1 = TemporalShiftScale(registrationName="TemporalShiftScale1", Input=lv)

    # Properties modified on temporalShiftScale1
    temporalShiftScale1.Scale = shift
    temporalShiftScale1.Periodic = 1
    temporalShiftScale1.PeriodicEndCorrection = 0
    temporalShiftScale1.MaximumNumberOfPeriods = 4

    temporalShiftScale1Display = Show(temporalShiftScale1, renderView1, "UniformGridRepresentation")

    # Properties modified on tempInt
    tempInt = FindSource(f"tempInt_{structure}")
    tempInt.Input = temporalShiftScale1
    tempInt.DiscreteTimeStepInterval = 0
    tempInt.ResampleFactor = 3 if timesteps == 20 else 2

    # update the view to ensure updated data information
    renderView1.Update()

    sources = []

    names = GetSources()

    color = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]

    num_con = sum([1 for i, s in list(names.keys()) if "Connectivity" in i])
    sources = [FindSource("Connectivity" + str(i + 1)) for i in range(num_con)]

    # Arrego temporal
    sources = [x for x in sources if x is not None]
    num_con = len(sources)

    # Replace by extract selection or clip if found as lower branch
    for i in range(1, num_con + 1):
        if FindSource("ExtractSelection" + str(i)):
            pos = re.sub(
                "[^0-9]",
                "",
                list(names.keys())[
                    list(names.values()).index(FindSource("ExtractSelection" + str(i)).Input)
                ][0],
            )
            sources[int(pos) - 1] = FindSource("ExtractSelection" + str(i))
        elif FindSource("Clip" + str(i)):
            pos = re.sub(
                "[^0-9]",
                "",
                list(names.keys())[list(names.values()).index(FindSource("Clip" + str(i)).Input)][
                    0
                ],
            )
            sources[int(pos) - 1] = FindSource("Clip" + str(i))

    for j, s in enumerate(sources[0:4]):
        w = str(j + 1)
        print(i)
        globals()["particleTracer" + w] = ParticleTracer(
            registrationName="particleTracer" + w, Input=tempInt, SeedSource=s
        )
        globals()["particleTracer" + w].SelectInputVectors = ["POINTS", "velocity"]
        globals()["particleTracer" + w].ForceReinjectionEveryNSteps = 1
        globals()["particleTracer" + w + "Display"] = Show(
            globals()["particleTracer" + w], renderView1, "GeometryRepresentation"
        )

        # create a new 'Temporal Particles To Pathlines'
        globals()["pathlines" + w] = TemporalParticlesToPathlines(
            registrationName="pathlines" + w,
            Input=globals()["particleTracer" + w],
            Selection=None,
        )

        # Properties modified on temporalParticlesToPathlines1
        globals()["pathlines" + w].MaskPoints = 1
        globals()["pathlines" + w].MaxTrackLength = 6
        globals()["pathlines" + w].IdChannelArray = "ParticleId"
        globals()["pathlines" + w + "Display"] = Show(
            globals()["pathlines" + w], renderView1, "GeometryRepresentation"
        )
        globals()["pathlines" + w + "Display"].SetScalarBarVisibility(renderView1, True)
        globals()["pathlines" + w + "Display_1"] = Show(
            OutputPort(globals()["pathlines" + w], 1),
            renderView1,
            "GeometryRepresentation",
        )
        # Change color
        ColorBy(globals()["pathlines" + w + "Display"], None)
        globals()["pathlines" + w + "Display"].AmbientColor = color[j]
        globals()["pathlines" + w + "Display"].DiffuseColor = color[j]
        ColorBy(globals()["pathlines" + w + "Display"], ("POINTS", "velocity", "Magnitude"))

        # rescale color and/or opacity maps used to include current data range
        globals()["pathlines" + w + "Display"].RescaleTransferFunctionToDataRange(True, False)

        # get opacity transfer function/opacity map for 'velocity'
        velocityLUT = GetColorTransferFunction("velocity")
        velocityLUT.ApplyPreset("Jet", True)
        velocityLUT.RescaleTransferFunction(0.0, 0.5)
        velocityPWF = GetOpacityTransferFunction("velocity")
        velocityPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.5, 0.3478260934352875, 0.5, 0.0]
        velocityPWF.RescaleTransferFunction(0.0, 0.5)

        globals()["pathlines" + w + "Display"].Opacity = 0.5
        globals()["pathlines" + w + "Display"] = Show(
            globals()["pathlines" + w], renderView1, "GeometryRepresentation"
        )
        globals()["pathlines" + w + "Display"].SetScalarBarVisibility(renderView1, True)

    # HideAll
    HideAll(renderView1)

    # update the view to ensure updated data information
    renderView1.Update()

    for k, s in enumerate(sources[0:4]):
        globals()["pathlines" + str(k + 1) + "Display"] = Show(
            globals()["pathlines" + str(k + 1)], renderView1, "GeometryRepresentation"
        )

    # reset view to fit data
    renderView1.ResetCamera(False)

    # Show the LA
    lA_Final = FindSource("stl_la")
    lA_FinalDisplay = Show(lA_Final, renderView1, "GeometryRepresentation")
    lA_FinalDisplay.Opacity = 0.2

    # # get animation scene
    animationScene1 = GetAnimationScene()
    animationScene1.GoToFirst()

    # update the view to ensure updated data information
    # renderView1.ResetActiveCameraToNegativeY()
    renderView1: ResetCamera()

    # Update scalar bars
    update_pipeline()
    UpdateScalarBars()

    # Zoom slightly
    camera = GetActiveCamera()
    camera.Dolly(1.5)
    Render()
    SaveState(join(anim_path, basename(p)))

    delete_all()
