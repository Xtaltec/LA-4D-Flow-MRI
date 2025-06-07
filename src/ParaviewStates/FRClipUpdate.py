"""
Script to update SurfaceFlow filters in ParaView based on ExtractSelection or Clip sources after correcting the positioning of vessel cross-sections.
Afterward, save the resulting state using SaveFRState.py.

@author: Xabier Morales Ferez
"""

import getpass
import os
import re
from os.path import basename, join
from pathlib import Path

from paraview.simple import *

DATA_PATH = ""

if not DATA_PATH:
    raise ValueError(
        "DATA_PATH is not defined. Please set it manually in the macro before running the script."
    )

states_path = join(DATA_PATH, "processed", "ParaviewStates", "flow_rate")
Path(states_path).mkdir(parents=True, exist_ok=True)

# Retrieve sources and count connectivity filters
names = GetSources()
num_con = sum(1 for key, _ in names if "Connectivity" in key)

# Get active render view
render_view = GetActiveViewOrCreate("RenderView")

# Update SurfaceFlow filters based on ExtractSelection or Clip sources
for i in range(1, num_con + 1):
    extract_selection = FindSource(f"ExtractSelection{i}")
    clip = FindSource(f"Clip{i}")
    if extract_selection or clip:
        source = extract_selection if extract_selection else clip
        pos = re.sub(r"[^0-9]", "", list(names.keys())[list(names.values()).index(source.Input)][0])
        FindSource("SurfaceFlow" + str(pos)).Input = FindSource("Clip" + str(i))
        surface_flow = FindSource(f"SurfaceFlow{pos}")
        surface_flow.Input = source
        Show(surface_flow, render_view, "UnstructuredGridRepresentation")

# Retrieve case identifier
vti_lv = FindSource("vti_lv")
case_id = basename(os.path.dirname(vti_lv.FileName[0]))

# Save the updated state
SaveState(join(states_path, f"{case_id}.pvsm"))
