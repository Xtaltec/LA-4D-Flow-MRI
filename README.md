# A Computational Pipeline for Advanced Analysis of 4D Flow MRI in the Left Atrium

[![arXiv Preprint](https://img.shields.io/badge/arXiv-2505.09746-b31b1b.svg)](https://arxiv.org/abs/2505.09746)

This repository contains the implementation of the computational pipeline described in the preprint: "[A Computational Pipeline for Advanced Analysis of 4D Flow MRI in the Left Atrium](https://arxiv.org/abs/2505.09746)".

## Table of Contents
- [Overview](#overview)
- [Pipeline Workflow](#pipeline-workflow)
- [System Requirements](#system-requirements)
- [Installation and Setup](#installation-and-setup)
- [Configuration](#configuration)
- [Data Structure Requirements](#data-structure-requirements)
- [Usage](#usage)
  - [1. Data Conversion (`generate_pcmra.py`)](#1-data-conversion-generate_pcmrapy)
  - [2. Segmentation (`segmentation_*.py`)](#2-segmentation-segmentation_py)
  - [3. Velocity Masking (`velocity_masking.py`)](#3-velocity-masking-velocity_maskingpy)
  - [4. ParaView Analysis (Manual Steps)](#4-paraview-analysis-manual-steps)
  - [5. Final Analysis (`analysis.py`)](#5-final-analysis-analysispy)
- [Project Structure](#project-structure)
- [Known Issues](#known-issues)
- [Ongoing features](#ongoing-features)
- [References](#references)

## Overview

This pipeline provides a comprehensive framework for the analysis of 4D Flow MRI data in the left atrium (LA). It leverages a combination of Python and R scripts and ParaView for the processing, visualization, and post-processing of 4D Flow MRI data.

### Key Features

-   **End-to-End Processing:** From raw DICOM data to publication-ready statistical analysis and plots.
-   **Automated Segmentation:** Utilizes the state-of-the-art **nnU-Net** framework for segmenting anatomical structures from Phase Contrast MR Angiography (PC-MRA).
-   **Advanced Hemodynamic Analysis:** Includes scripts for calculating metrics like Kinetic Energy (KE), Vorticity,  Wall Shear Stress (WSS), Viscous Energy Loss and more.
-   **Reproducibility:** The entire pipeline is containerized using Docker, ensuring a consistent and cross-platform environment.
-   **Customizable and Extensible:** Provides clear guidelines and template functions for adapting the pipeline to data from different medical centers and scanners.


## Pipeline Workflow

The pipeline is organized into five main stages:

1.  **Data Conversion**: Raw 4D Flow MRI data (e.g., DICOM) is converted into the NIfTI format. Time-averaged PC-MRA images are generated for segmentation.
2.  **Segmentation**: The generated PC-MRA images are used to train an nnU-Net model or for inference with a pre-trained model to segment anatomical structures.
3.  **Velocity Masking**: The resulting segmentation masks are applied to the time-resolved velocity fields to isolate the regions of interest.
4.  **ParaView Analysis**: A semi-automated workflow in ParaView is used to place measurement planes and extract initial flow rate data.
5.  **Final Analysis**: Python scripts process the exported data to calculate advanced hemodynamic parameters and generate plots for analysis.


## System Requirements

-   **Operating System**: Linux, macOS, or Windows with WSL2.
-   **Hardware**: A CUDA-enabled NVIDIA GPU is required for training the segmentation model and is highly recommended for inference.
-   **Software**: Docker and Docker Compose must be installed.
    -   Install Docker [here](https://docs.docker.com/get-docker/).
    -   Install Docker Compose [here](https://docs.docker.com/compose/install/).

### Software Requirements & Recommended Tools

-   **ParaView**: Required for visualization and flow extraction. Tested with versions **5.11.1** and **5.12.1**. Download [here](https://www.paraview.org/download/).
-   **dcm2niix**: Used to convert DICOM images to the NIfTI format. It is installed automatically within the Docker container.
-   **Visual Studio Code (Recommended)**: The recommended editor for interacting with the development container via the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

## Installation and Setup

### Prerequisites
Ensure Docker is installed on your system before proceeding.

### Running the Application

We recommend using the integrated Docker support in VS Code.

#### Option 1: Visual Studio Code (Recommended)
1.  Open the project folder in VS Code.
2.  Right-click on the `docker-compose.yaml` file.
3.  Select **"Compose Up"**.
4.  Choose the `4d_flow_gpu` service for GPU-accelerated training/inference. If you do not have a GPU or already have segmentations, choose the `4d_flow_cpu` service.
5.  Once the container is running, attach VS Code to it using the Dev Containers extension.

#### Option 2: Command Line
1.  Navigate to the project's root directory in your terminal.
2.  Run the following command to build and start the GPU-enabled service:
    ```bash
    docker compose up --build 4d_flow_gpu
    ```
3.  Connect to the running container in a new terminal window:
    ```bash
    docker exec -it <container_name> /bin/bash
    ```

## Configuration

Before running the pipeline, create a `.env` file in the project root with the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_PATH` | Local path where raw data is stored (will be mounted in the container) | Required |
| `BASE_PATH` | Base working directory for project code and data | `/project/` |

## Data Structure Requirements

The pipeline expects a specific naming convention for patient identification:
- Format: `STUDY_ID_PATIENT_ID`
- Study ID: Character-based identifier
- Patient ID: 3-digit numeric identifier (can repeat across studies)
- Separator: Underscore (`_`)

### Expected Directory Structure

The project expects the following directory structure for data:

```
BASE_PATH/data/
├── raw/ # Raw data directory created by the user at DATA_PATH
│   ├── magnitude/
│   │   ├── STUDY1_001/
│   │   ├── STUDY1_002/
│   │   ├── STUDY2_001/
│   │   └── ...
│   └── velocity/
│       ├── STUDY1_001/
│       └── ...
├── processed/ # Generated by the pipeline
│   ├── anatomy/
│   ├── flow/
│   ├── pcmra/
│   ├── segmentation/ # Segmentation masks and nnU-Net dataset structure
│   └── ...
├── clinical/
│   └── clinical_data.xlsx # Clinical data file with patient IDs and clinical parameters
└── analysis/
    ├── flow_metrics/
    ├── feather/
    └── plots/
```

## Data Preprocessing Requirements

**Important**: The 4D Flow MRI data from the reference medical centers was preprocessed using commercial software for:
- Phase unwrapping
- Eddy current correction
- Concomitant gradient field correction

You may need to perform similar preprocessing on your data before using this pipeline.

## Usage

The pipeline consists of 5 main scripts, designed to be run in sequence.


### 1.Data Conversion (`generate_pcmra.py`)

This script generates time-averaged PC-MRA from 4D Flow MRI data.

**Input**: Raw phase images and anatomical/magnitude images (optional) from 4D Flow MRI  
**Output**: PC-MRA suitable for region of interest segmentation

**Important:** Due to the high variability of DICOM formats, you will likely need to customize the data reading functions. We provide helper utilities and reference implementations in `src/utils/`. 

- `src/utils/data_readers.py`:  Contains reference functions (`magnitude_ABI`, `velocity_Clinic`) for reading data and converting it to the required format.
    - `magnitude_*`: Convert magnitude DICOM files into RAS-oriented NIfTI format.
       - **Parameters**:
            - `path_in`: Folder containing subdirectories with magnitude DICOMs (typically named with "M")
            - `path_out`: Destination folder for output file
            - `case`: String identifier for naming the output file
        - **Output**: Single RAS-oriented NIfTI image to `DATA_PATH/anatomy/case_id.nii.gz`

    - `velocity_*`: Process velocity data and save as VTI files containing 3D velocity vector fields [X, Y, Z] in mm/s, oriented to RAS.
        - **Parameters**:
            - `path_in`: Folder containing case directory with VTI files or DICOM phase images
            - `path_out`: Output folder
            - `case`: Case identifier
        - **Output**: RAS-oriented VTI files saved to `DATA_PATH/flow/case_id/case_id_***.vti` with velocities in mm/s for each timestep (identified by 3-digit numbers)    

- `src/utils/data_readers.py`: Contains the `dcmtonii` function to convert DICOM files to NIfTI format.
- `src/utils/matrixops.py`: Contains the `orient_to_RAS` function to orient images in RAS (Right-Anterior-Superior) orientation.

### 2. Segmentation (`segmentation_training.py` and `segmentation_inference.py`)

These scripts handle the training and inference of nnU-Net [1] for segmenting PC-MRA images to identify anatomical structures of interest. Trained models for segmenting the left atrium, left ventricle, and whole left heart with ventricle outflow tract are provided in this Zenodo repository: [Zenodo: 4D Flow MRI nU-Net Pretrained Models](https://zenodo.org/records/15615378).

#### `segmentation_training.py`

Automates the data preparation and training workflow for nnUNet using 4D Flow MRI Phase Contrast Magnetic Resonance Angiography (PC-MRA) datasets. The script processes input images and masks, formats them into the nnUNet dataset structure, generates task-specific configurations, and launches training.

**Command-line Arguments**:

| Argument | Description | Example |
|----------|-------------|---------|
| `--datasets` | List of dataset names to include in training | `--datasets Study1 Study2` |
| `--structure` | List of anatomical structures to train on | `--structure aorta pulm_artery` |
| `--task_num` | Integer base ID to generate nnUNet task numbers | `--task_num 100` |
| `--task_base` | Task name suffix for generating descriptive task names | `--task_base PCMRAFlow` |

**Example Usage**:
```bash
python ./scripts/segmentation_training.py --datasets Study1 Study2 --structure la --task_num 200 --task_base PCMRAFlow
```

**Expected Input Directory Structure**:
```
BASE_PATH/
└── data/
    └── processed/
        └── segmentation/
            ├── pcmra/              # NIfTI input images
            ├── masks/              # NIfTI ground truth masks by structure
            │   ├── la/             # Left Atrium masks
            │   └── lv/             # Left Ventricle masks
            └── nnunet/
                └── training/
                    ├── datasets/   # Temporary per-cohort datasets
                    └── tasks/      # Combined dataset for each structure formatted for nnUNet training
```
#### `segmentation_inference.py`

Performs inference using trained nnU-Net models to segment PC-MRA images. The script handles batch processing with retry mechanisms and automatically organizes predicted masks by anatomical structure.

**Command-line Arguments**:

| Argument | Description | Example |
|----------|-------------|---------|
| `--model_name` | Name of the trained nnU-Net model (structure extracted from suffix) | `--model_name Dataset200_PCMRAFlow_la` |
| `--path_to_predict` | Directory containing input NIfTI images for prediction | `--path_to_predict /data/pcmra/` |
| `--path_output` | Output directory for predicted segmentation masks | `--path_output /data/predicted/` |
| `--max_retries` | Maximum number of retry attempts for failed predictions | `--max_retries 3` |

**Example Usage**:
```bash
python ./scripts/segmentation_inference.py --model_name Dataset200_PCMRAFlow_la --path_to_predict /data/pcmra/ --path_output /data/predicted/ --max_retries 3
```

## 3. Velocity Masking (`velocity_masking.py`)

Processes 4D Flow MRI velocity data to isolate anatomical structures of interest using segmentation masks. The script applies masks to velocity fields, computes viscous energy loss, and generates visualization files for ParaView analysis.

**Command-line Arguments**:

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--not_all` | List of specific case IDs to process (overrides processing all cases) | `[]` | `--not_all STUDY1_001 STUDY1_002` |
| `--exclude` | List of case IDs to exclude from processing | `[]` | `--exclude STUDY2_003 STUDY2_004` |
| `--structure` | List of anatomical structures to process | `["la", "lv"]` | `--structure la lv aorta` |
| `--crop` | Crop non-ROI regions to reduce memory requirements | `1` | `--crop 1` |
| `--rewrite_stl` | Force rewrite of existing STL surface files | `1` | `--rewrite_stl 0` |
| `--rewrite_vtk` | Force rewrite of existing VTK volumetric mesh files | `1` | `--rewrite_vtk 0` |
| `--mu` | Dynamic viscosity coefficient (Pa·s) | `0.0035` | `--mu 0.004` |

**Example Usage**:
```bash
python ./scripts/velocity_masking.py --structure la lv --crop 1 --mu 0.0035 --exclude STUDY1_003
```

## 4. ParaView Analysis (Manual Steps)

This stage requires manual interaction in ParaView to ensure accurate placement of measurement planes. The required state files and scripts are located in the `src/ParaviewStates` directory.

### Step 1: Initial Setup
Add all the Paraview States except 'GenerateStates.py' in your Paraview GUI as Macros. Then load individually the initial ParaView states located in the `src/ParaviewStates/initial_states` directory. These contain the base 4D flow data and segmentation meshes for each patient case.

### Step 2: Sphere Positioning
1. **Manually position spheres** in ParaView to mark the center of each anatomical structure (RS, RI, LS, LI, MV, LVOT)
2. **Run SaveSphereState.py** - Saves the ParaView state after positioning spheres in each structure

### Step 3: Automated Cross-Section Generation
**Run CreateFRPlanes.py** - Automatically generates cross-sectional planes perpendicular to the flow direction at each sphere position. This script:
- Computes the dominant flow direction for each structure
- Creates perpendicular planes as vesel cross-section for flow rate measurement
- Saves preliminary results in CSV format

### Step 4: Manual Cross-Section Correction
1. **Open the generated state** in ParaView to review the automatically positioned cross-sections
2. **Manually adjust** any incorrectly positioned planes to ensure they properly intersect the vessel lumens
3. **Clip if necessary** to remove any unwanted regions outside the vessel lumen. Select the Slice source and use the "Clip" filter in the "Sphere" mode to refine the cross-section. Typically, it needs to be done always for the MV and LVOT structures but it may also be necessary for the RS, RI, LS, and LI structures.
3. **Run FRClipUpdate.py** - Updates SurfaceFlow filters based on ExtractSelection or Clip sources after correcting cross-section positioning

### Step 5: Final State Saving
**Run SaveFRState.py** - Saves the corrected flow rate measurement state after manual positioning corrections

### Step 6: Final Data Export
**Run ExportFRData.py** - Extracts all flow rate measurements and cross-sectional areas from the corrected ParaView states for statistical analysis. The final processed state files are also saved in the `src/ParaviewStates/flow_rate` directory.

**Important**: Always visually inspect the automatically generated cross-sections in ParaView before proceeding to ensure accurate flow measurements. The manual correction step is critical for reliable results.

### Helper Scripts
DeleteAll.py script can be used to delete all sources in the ParaView GUI and resets the current session.

### Optional: ParticleViz.py
This script can be used to visualize particle traces in the ParaView GUI. It allows you to see the flow patterns and trajectories of particles within the 4D flow data.

## 5. Hemodynamic parameter analysis (`analysis.py`)

The flow analysis module provides comprehensive computational fluid dynamics analysis for 4D flow MRI data, calculating hemodynamic parameters, flow metrics, and wall shear stress values and derivatives (OSI, TAWSS, ECAP). It also exports results in a format suitable for statistical analysis and visualization in R using `ggplot2` and generates several plots.

### Key Components

**Core Analysis Functions (`utils/analysis.py`)**
- `flow_rate_to_R()` - Processes patient-level clinical data and temporal flow rate data, calculating volume metrics and stroke volumes
- `compute_hemodynamic_parameters()` - Computes kinetic energy, viscous energy loss, vorticity, and Q-criterion for each cardiac frame
- `calculate_viscous_energy_loss()` - Calculates viscous dissipation based on strain rate tensor analysis
- `create_ggplot2_plots()` - Generates ggplot2 plots for visualizing flow metrics and hemodynamic parameters grouped by patient ID or clinical parameters

**Wall Parameter Analysis (`src/utils/wall.py`)**
- `wss_calculator()` - Calculates wall shear stress from velocity field data using surface mesh interpolation
- `compute_wall_parameters()` - Computes time-averaged wall shear stress (TAWSS), oscillatory shear index (OSI), and endothelial cell activation potential (ECAP) based on wall shear stress data

### Requirements

- **Clinical data file** (Excel/CSV format) located at `data/clinical/clinical_data.xlsx` containing:
  - Patient ID column matching the identifiers used throughout the pipeline
  - Heart rate column (HR_4D or specified via `--hr_column`)
  - Additional clinical parameters as specified in `--clinical_columns`
- Processed flow rate CSV files from the ParaView workflow, located in `data/flow_rate/` with the naming convention:
  - `case_id_flow_rate.csv` (e.g., `STUDY1_001_flow_rate.csv`)
- VTI velocity field files for hemodynamic parameter calculation
- VTK surface meshes for wall shear stress computation

### Usage

```bash
python ./scripts/analysis.py --structure la lv --clinical_columns Age BSA EF --interp_num 30
```

### Command Line Arguments

- `--structure` - Anatomical structures to analyze (e.g., LV, LA, LVOT, MV)
- `--clinical_columns` - Clinical parameters to include in output
- `--interp_num` - Number of time points for temporal interpolation in case there are datasets with different time resolutions (default: 30)
- `--structure` - Anatomical structures to analyze (default: ["la"])
- `--id_column` - Patient identifier column name (default: "name")
- `--hr_column` - Heart rate column name (default: "HR_4D")
- `--not_all` - Specific patient IDs to include
- `--exclude` - Patient IDs to exclude from analysis

### Output Files

**Flow Metrics**
- `flow_rate.feather/xlsx` - Time-resolved flow rate data with clinical parameters
- `volume_tot.feather/xlsx` - Calculated volume metrics and stroke volumes
- `hemodynamic_parameters.feather` - Advanced hemodynamic parameters (kinetic energy, Q-criterion, vorticity)
- ggplot2 plots in `plots/` directory for visualizing flow metrics and hemodynamic parameters

**Wall Parameters**
- `wss/{structure}/{case_id}_{frame}.vtk` - Individual frame wall shear stress data
- `ecap/{structure}/{case_id}.vtk` - Time-averaged hemodynamic parameters (TAWSS, OSI, ECAP)

### Data Processing Pipeline

1. **Clinical Data Integration** - Merges 4D flow measurements with patient demographics and clinical parameters
2. **Flow Rate Analysis** - Processes surface flow measurements from multiple cardiac structures
3. **Volume Calculations** - Computes stroke volume, net flow, and cumulative volume changes
4. **Hemodynamic Parameter Computation** - Calculates energy-based and vorticity-based flow metrics
5. **Wall Shear Stress Analysis** - Computes surface-based hemodynamic parameters using velocity gradients

## Known Issues

- **Gmsh on macOS**: There's currently no working distribution of Gmsh for aarch64 (Apple Silicon) CPUs on macOS.
- **Docker Desktop on Windows**: When running Docker Desktop on Windows, the virtual disk `.vhdx` file used by WSL2 to store its filesystems can grow in size, potentially causing you to run out of disk space, especially on smaller SSDs. Below are some links to possible solutions:

  - [WSL Issue #4699](https://github.com/microsoft/WSL/issues/4699)
  - [StackOverflow: Docker Desktop WSL ext4.vhdx too large](https://stackoverflow.com/questions/70946140/docker-desktop-wsl-ext4-vhdx-too-large)

  In our case, converting the virtual disk image to sparse resolved the issue:

  ```sh
  wsl --shutdown # Necessary to stop all WSL processes
  wsl --manage docker-desktop --set-sparse true
  ```
## Ongoing features
- Integration of R and `ggplot2` for generating visualizations from the ParaView analysis.

## Citation

If you find this repository useful in your research, please cite our paper:

Morales X, Elsayed A, Zhao D, Loncaric F, Aguado A, Masias M, Quill G, Ramos M, Doltra A, García-Alvarez A, et al.  
**Automated 4D flow MRI pipeline for the quantification of advanced hemodynamic parameters in the left atrium.**  
*Scientific Reports* (2026).  
https://doi.org/10.1038/s41598-025-34972-7

```bibtex
@article{morales2026automated,
  title={Automated 4D flow MRI pipeline for the quantification of advanced hemodynamic parameters in the left atrium},
  author={Morales, Xabier and Elsayed, Ayah and Zhao, Debbie and Loncaric, Filip and Aguado, Ainhoa and Masias, Mireia and Quill, Gina and Ramos, Marc and Doltra, Adelina and Garc{\'\i}a-Alvarez, Ana and others},
  journal={Scientific Reports},
  year={2026},
  publisher={Nature Publishing Group UK London},
  doi={10.1038/s41598-025-34972-7}
}

## References
[1] Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat Methods 18, 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z
