"""
This script exports the flow rate data from all Paraview states stored under the `flow_rate` directory. for further analysis.

@author: Xabier Morales Ferez
"""

import csv
import itertools
import re
import tempfile
from glob import glob
from os.path import basename, join
from pathlib import Path

import numpy as np
from paraview.simple import *

DATA_PATH = ""

structure = [
    "RS",
    "RI",
    "LS",
    "LI",
    "MV",
    "LVOT",
]  # List of structures to process; can be modified as needed
not_all = []  # List of patients to process; if empty, all patients will be processed
exclude = []  # List of patients to exclude from processing
rewrite = True  # If True, will overwrite existing CSV files

if not DATA_PATH:
    raise ValueError(
        "DATA_PATH is not defined. Please set it manually in the macro before running the script."
    )

states_path = join(DATA_PATH, "processed", "ParaviewStates", "flow_rate")
output_path = join(DATA_PATH, "processed", "flow_rate")

Path(output_path).mkdir(parents=True, exist_ok=True)

# List of all patients
patients = glob(join(states_path, "*.pvsm"))

# Filter patients based on criteria
if not_all:
    patients = [p for p in patients if basename(p).split(".")[0] in not_all]

if exclude:
    patients = [p for p in patients if basename(p).split(".")[0] not in exclude]

if not rewrite:
    flow_pat = np.unique(
        [
            "_".join(basename(p).split(".")[0].split("_")[:2])
            for p in glob(join(output_path, "*.csv"))
        ]
    )
    patients = [p for p in patients if basename(p).split(".")[0] not in flow_pat]

failed = []


# Helper functions
def delete_all():
    while len(GetSources()) >= 1:
        try:
            Delete(list(GetSources().values())[0])
        except:
            pass


def reset_session():
    pxm = servermanager.ProxyManager()
    pxm.UnRegisterProxies()
    del pxm
    Disconnect()
    Connect()


# Process each patient
for patient in patients:
    case_id = basename(patient).split(".")[0]
    print(f"Processing case: {case_id}...")

    try:
        LoadState(patient)
        vti_lv = FindSource("vti_lv")

        # Process each structure
        for i, struct in enumerate(structure):
            plot_data = FindSource(f"PlotDataOverTime{i + 1}")
            plot_data.UpdatePipeline()

            SaveData(
                join(output_path, f"{case_id}_{struct}.csv"),
                proxy=plot_data,
                ChooseArraysToWrite=1,
                RowDataArrays=["avg(Surface Flow)", "max(Surface Flow)"],
                FieldAssociation="Row Data",
                AddMetaData=0,
            )

        # Process connectivity filters
        names = GetSources()
        num_con = sum(1 for key, _ in names if "Connectivity" in key)
        sources = [FindSource(f"Connectivity{i + 1}") for i in range(num_con)]
        sources = [s for s in sources if s is not None]

        for i in range(1, num_con + 1):
            source = FindSource(f"ExtractSelection{i}") or FindSource(f"Clip{i}")
            if source:
                pos = re.sub(
                    r"[^0-9]", "", list(names.keys())[list(names.values()).index(source.Input)][0]
                )
                sources[int(pos) - 1] = source

        # Compute and save area data
        area = []
        for source in sources:
            integrate = IntegrateVariables(Input=source)
            integrate.UpdatePipeline()

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = join(temp_dir, "temp.csv")

                SaveData(
                    temp_file,
                    proxy=integrate,
                    ChooseArraysToWrite=1,
                    CellDataArrays=["Area"],
                    FieldAssociation="Cell Data",
                    AddMetaData=0,
                )

                with open(temp_file, "r") as fp:
                    reader = csv.reader(fp)
                    area.append(float(next(itertools.islice(reader, 1, None))[0]))

            Delete(integrate)

        np.savetxt(
            join(output_path, f"{case_id}_Area.csv"),
            np.vstack((structure, area)),
            fmt="%s",
            delimiter=",",
        )

    except Exception as e:
        failed.append(case_id)
        print(f"Failed for case {case_id}: {e}")

    finally:
        delete_all()
        reset_session()

if failed:
    print(f"Failed cases: {', '.join(failed)}")
