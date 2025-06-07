from glob import glob
from ntpath import basename
from os.path import join, splitext
from typing import Dict

import numpy as np
import pandas as pd
import pyvista as pv
from natsort import natsorted
from scipy import interpolate


def cubic_interp(y, n_new):
    """Interpolate a list to a new number of elements using cubic interpolation"""
    n_old = len(y)
    f = interpolate.interp1d(np.linspace(0, 1, n_old), y, kind="cubic")
    return f(np.linspace(0, 1, n_new))


def load_clinical_data(paths: Dict[str, str]):
    clinical_data_path = paths["clinical_data"]
    file_ext = splitext(clinical_data_path)[1].lower()

    # Check if the clinical data file exists
    if not glob(clinical_data_path):
        raise FileNotFoundError(f"Clinical data file not found: {clinical_data_path}")

    try:
        if file_ext == ".xlsx":
            clinical_data = pd.read_excel(clinical_data_path)
        elif file_ext == ".csv":
            clinical_data = pd.read_csv(clinical_data_path)
        else:
            raise ValueError(
                f"Unsupported clinical data file format: {file_ext}. Please use .xlsx or .csv."
            )
    except Exception as e:
        raise Exception(f"Error reading clinical data file '{clinical_data_path}': {e}")

    return clinical_data


def calculate_viscous_energy_loss(velocity, divergence, grid_spacing, viscosity=0.0035):
    """
    Calculate the viscous energy loss in a fluid system.

    Args:
        velocity (np.ndarray): Velocity field with shape (nx, ny, nz, 3, nt)
                             where nx, ny, nz are spatial dimensions,
                             3 represents velocity components (vx, vy, vz),
                             and nt is the number of time steps
        divergence (np.ndarray): Velocity divergence field
        grid_spacing (tuple): Grid spacing (dx, dy, dz) in each direction
        viscosity (float, optional): Fluid dynamic viscosity. Defaults to 0.0035

    Returns:
        np.ndarray: Viscous energy loss per voxel and time step

    Notes:
        The function calculates viscous dissipation based on the strain rate tensor:
        Φᵥ = μ * (∂ᵢvⱼ + ∂ⱼvᵢ - (2/3)δᵢⱼ∇·v)
    """
    # Validate input shapes
    nx, ny, nz, _, nt = velocity.shape
    if velocity.shape[3] != 3:
        raise ValueError("Velocity field must have 3 components (vx, vy, vz)")

    # Initialize dissipation tensor
    dissipation = np.zeros((nx, ny, nz, nt))

    # Calculate volume element
    volume_element = np.prod(grid_spacing)

    # Calculate strain rate tensor components and accumulate dissipation
    for i in range(3):
        for j in range(3):
            # Calculate velocity gradients
            grad_i = np.gradient(velocity[..., j, :], axis=i)
            grad_j = np.gradient(velocity[..., i, :], axis=j)

            # Add Kronecker delta term for diagonal elements
            kronecker_term = (2 / 3) * divergence if i == j else 0

            # Accumulate strain rate tensor contribution
            strain_rate = grad_i + grad_j - kronecker_term
            dissipation += strain_rate

    # Complete the calculation
    dissipation *= 0.5
    energy_loss = viscosity * dissipation * volume_element

    return energy_loss


def flow_rate_to_R(
    paths,
    args,
):
    """
    Processes patient-level clinical data and temporal flow rate data for visualization in R.

    This function reads clinical and demographic data from an Excel or CSV file,
    reads time-series flow rate data from 4D Flow MRI CSV files, merges these
    datasets, and transforms the flow rate data into a long-format DataFrame
    suitable for plotting. It also calculates various volume metrics.

    Args:
        paths (dict): A dictionary containing paths to input and output directories/files:
                      - 'clinical_data': Path to the clinical and demographic data file (Excel or CSV).
                      - 'fr_path': Directory containing flow rate CSV files.
                      - 'fr_feather': Directory to save the processed flow rate feather file.
                      - 'flow_metrics': Directory to save flow metrics Excel files.
                      - 'energy_feather': Directory to save the processed volume feather file.

        args (object): An object containing arguments:
                       - id_column (str): The name of the column in clinical_data that uniquely
                         identifies patients (e.g., "id").
                       - clinical_columns (list): A list of strings specifying clinical
                         parameters (column names) to include in the output DataFrames.
                       - not_all (list, optional): A list of patient ids to include if
                         `args.not_all` is True. Defaults to an empty list.
                       - exclude (list, optional): A list of patient ids to exclude.
                         Defaults to an empty list.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
               - fr_df (pd.DataFrame): Long-format DataFrame with flow rate data and merged
                                    clinical parameters specified in `args.clinical_columns`.
               - vol_df (pd.DataFrame): Long-format DataFrame with calculated volume metrics and
                                     merged clinical parameters specified in `args.clinical_columns`.

    Raises:
        FileNotFoundError: If clinical data file or flow rate directory is not found.
        ValueError: If essential columns are missing from clinical data (e.g., `id_column`),
                    or if no valid flow rate files remain after filtering.
        Exception: For other unexpected errors during processing.
    """
    print("\nComputing flow rate...")

    # --- 1. Load Clinical Data ---
    clinical_data = load_clinical_data(paths)

    # Validate essential args and columns
    if not hasattr(args, "id_column") or not isinstance(args.id_column, str):
        raise ValueError("Argument 'args.id_column' must be provided and be a string.")

    if args.id_column not in clinical_data.columns:
        raise ValueError(f"The specified id column '{args.id_column}' not found in clinical data.")

    # Ensure hr_column and SV columns exist, as they are used internally for calculations
    # and SV is updated. Initialize with NaN if they don't exist.
    if args.hr_column not in clinical_data.columns:
        clinical_data[args.hr_column] = np.nan
        print(
            "Warning: Heart Rate column not found in clinical data. Defaulting to 60 for HR calculations."
        )
    if "SV" not in clinical_data.columns:
        clinical_data["SV"] = np.nan
        print("Warning: 'SV' column not found in clinical data. It will be added and updated.")

    # Pre-set id column as index for faster lookups using the user-defined id_column
    clinical_lookup = clinical_data.set_index(args.id_column)

    # --- 2. Gather Flow Rate Files ---
    fr_path_dir = paths["fr_path"]
    # Check if the flow rate directory exists and contains CSVs
    if not glob(join(fr_path_dir, "*.csv")):
        raise FileNotFoundError(
            f"No CSV files found in flow rate directory: {fr_path_dir}. Please check the path."
        )

    csv_list = glob(join(fr_path_dir, "*.csv"))
    csv_list = [c for c in csv_list if "Area" not in c]  # Exclude Area.csv files

    # Apply filtering based on args.not_all and args.exclude
    if hasattr(args, "not_all") and isinstance(args.not_all, list) and args.not_all:
        csv_list = [
            c
            for c in csv_list
            if f"{basename(c).split('_')[0]}_{basename(c).split('_')[1]}" in args.not_all
        ]
    elif hasattr(args, "not_all") and args.not_all:
        print(
            "Warning: 'args.not_all' was provided but is not a valid non-empty list. Skipping filtering."
        )

    if hasattr(args, "exclude") and isinstance(args.exclude, list) and args.exclude:
        csv_list = [
            c
            for c in csv_list
            if f"{basename(c).split('_')[0]}_{basename(c).split('_')[1]}" not in args.exclude
        ]
    elif hasattr(args, "exclude") and args.exclude:
        print(
            "Warning: 'args.exclude' was provided but is not a valid non-empty list. Skipping filtering."
        )

    if not csv_list:
        raise ValueError(
            "No flow rate CSV files remaining after filtering. Please check your 'not_all' or 'exclude' arguments."
        )

    # --- 3. Process Each Flow Rate File ---
    processed_data_list = []

    # Ensure clinical_columns is a list, even if empty or not provided
    user_clinical_cols = (
        args.clinical_columns
        if hasattr(args, "clinical_columns") and isinstance(args.clinical_columns, list)
        else []
    )

    for c in csv_list:
        try:
            # Extract metadata from filename (e.g., "id_study_structure.csv")
            file_parts = basename(c).split("_", 2)
            if len(file_parts) < 3:
                print(
                    f"Warning: Skipping malformed filename: {basename(c)}. Expected format like id_study_structure.csv"
                )
                continue

            case_id = f"{file_parts[0]}_{file_parts[1]}"
            structure = splitext(file_parts[2])[0]  # e.g., "MV", "LVOT", "LS"

            # Read the flow rate data
            fr_df_temp = pd.read_csv(c)
            if "avg(Surface Flow)" not in fr_df_temp.columns:
                print(f"Warning: Skipping {basename(c)} as 'avg(Surface Flow)' column not found.")
                continue
            f_temp = fr_df_temp["avg(Surface Flow)"].values

            # Flip flow rate if predominantly negative
            if np.all(f_temp < 0):
                f_temp *= -1
            elif np.any(f_temp < 0) and np.any(
                f_temp > 0
            ):  # Only check quantiles if both positive and negative values exist
                pos_vals = f_temp[f_temp > 0]
                neg_vals = f_temp[f_temp < 0]
                if len(pos_vals) > 0 and len(neg_vals) > 0:
                    if np.quantile(np.abs(pos_vals), 0.9) < np.quantile(np.abs(neg_vals), 0.9):
                        f_temp *= -1

            # Convert flow rate units to ml/s
            f_temp_ml_s = f_temp * 10**6

            # Retrieve patient-specific clinical data using the `id_column` from args
            if case_id not in clinical_lookup.index:
                print(
                    f"Warning: Clinical data not found for id: {case_id} (using '{args.id_column}' as identifier). Skipping flow rate file: {basename(c)}"
                )
                continue
            patient_clinical_row = clinical_lookup.loc[case_id]

            # Extract heart rate, default to 60 if NaN
            hr_value = patient_clinical_row.get(args.hr_column, 60)
            if pd.isna(hr_value):
                hr_value = 60

            num_time_points = len(f_temp)

            # Prepare data dictionary for this single flow rate file
            file_record = {
                "id": case_id,
                "study": file_parts[0],  # Assuming study is the first part of id_study
                "structure": structure,
                "flow_rate": f_temp_ml_s,
                "time": np.arange(num_time_points),
                "frame_idx": np.arange(num_time_points),
            }

            # Add user-defined clinical columns from args.clinical_columns
            for col in user_clinical_cols:
                file_record[col] = patient_clinical_row.get(col, "N/A")

            # Create a DataFrame for this file and append to list
            fr_df_for_file = pd.DataFrame(
                {
                    k: np.repeat(v, num_time_points) if np.isscalar(v) else v
                    for k, v in file_record.items()
                }
            )
            processed_data_list.append(fr_df_for_file)

        except Exception as e:
            print(f"Error processing file {basename(c)}: {e}")
            continue  # Continue to the next file

    if not processed_data_list:
        raise ValueError(
            "No flow rate data was successfully processed. Check input files and processing logic."
        )

    fr_df = pd.concat(processed_data_list, ignore_index=True)

    # Make all negative MV flow rates zero (contamination from LVOT)
    fr_df.loc[(fr_df["structure"] == "MV") & (fr_df["flow_rate"] < 0), "flow_rate"] = 0

    feather_output_path_fr_df = join(paths["feather_output"], "flow_rate.feather")
    fr_df.to_feather(feather_output_path_fr_df)
    print(f"Flow rate data saved to {feather_output_path_fr_df}")

    # --- 4. Calculate Volume Metrics for vol_df ---
    volume_data_list = []

    # Group the main DataFrame by id to process each patient's data
    for patient_id, patient_fr_df_group in fr_df.groupby("id"):
        if patient_id not in clinical_lookup.index:
            print(
                f"Warning: Clinical data not found for patient id '{patient_id}' during volume calculation. Skipping."
            )
            continue  # Skip this patient if clinical data is missing

        patient_clinical_row = clinical_lookup.loc[patient_id]

        hr_value = patient_clinical_row.get(args.hr_column, 60)
        if pd.isna(hr_value):
            hr_value = 60

        num_time_points = len(patient_fr_df_group["time"].unique())

        # Calculate time step (dt)
        dt = 60 / ((num_time_points - 1) * hr_value) if num_time_points > 1 else 0

        pv_flow_rates = {}
        for pv_type in ["MV", "RS", "RI", "LS", "LI", "LVOT"]:
            pv_data = patient_fr_df_group[patient_fr_df_group["structure"] == pv_type][
                "flow_rate"
            ].values
            if len(pv_data) != num_time_points:
                print(
                    f"Warning: Flow data for structure '{pv_type}' for patient {patient_id} is incomplete ({len(pv_data)} points, expected {num_time_points}). Padding with zeros."
                )
                pv_flow_rates[pv_type] = np.zeros(num_time_points)
            else:
                pv_flow_rates[pv_type] = pv_data

        vol_t_temp = (
            -pv_flow_rates["MV"]
            + pv_flow_rates["RS"]
            + pv_flow_rates["RI"]
            + pv_flow_rates["LS"]
            + pv_flow_rates["LI"]
        )
        vol_pv_temp = (
            pv_flow_rates["LS"] + pv_flow_rates["LI"] + pv_flow_rates["RS"] + pv_flow_rates["RI"]
        )
        vol_mv_temp = pv_flow_rates["MV"]
        vol_lvot_temp = pv_flow_rates["LVOT"]
        vol_lv_temp = pv_flow_rates["LVOT"] - pv_flow_rates["MV"]

        # Calculate volumes and cumulative volumes
        current_volumes = {
            "Volume_tot": vol_t_temp * dt,  # Total volume: MV flow - PV flow
            "Volume_mv": vol_mv_temp * dt,
            "Volume_pv": vol_pv_temp * dt,
            "Volume_lvot": vol_lvot_temp * dt,
            "Volume_lv": vol_lv_temp * dt,
            "Volume_tot_cm": np.cumsum(vol_t_temp * dt),
            "Volume_mv_cm": np.cumsum(vol_mv_temp * dt),
            "Volume_pv_cm": np.cumsum(vol_pv_temp * dt),
            "Volume_lvot_cm": np.cumsum(vol_lvot_temp * dt),
            "Volume_lv_cm": np.cumsum(vol_lv_temp * dt),
        }

        # Calculate Stroke Volume and update in the main clinical_data DataFrame
        stroke_vol = (
            np.quantile(current_volumes["Volume_lvot_cm"], 0.99)
            if len(current_volumes["Volume_lvot_cm"]) > 0
            else 0
        )

        # Prepare data for vol_df for this patient
        patient_volume_record = {
            "id": patient_id,
            "study": patient_fr_df_group["study"].iloc[0],  # Get first study value for the group
            "time": np.arange(num_time_points),
            "stroke_volume": stroke_vol,
        }
        patient_volume_record.update(current_volumes)  # Add the calculated volume metrics

        # Add user-defined clinical columns from args.clinical_columns
        for col in user_clinical_cols:
            patient_volume_record[col] = patient_clinical_row.get(col, "N/A")

        vol_df_for_patient = pd.DataFrame(
            {
                k: np.repeat(v, num_time_points) if np.isscalar(v) else v
                for k, v in patient_volume_record.items()
            }
        )
        volume_data_list.append(vol_df_for_patient)

    if not volume_data_list:
        raise ValueError(
            "No volume data could be calculated. Check processed flow rate data and clinical data consistency."
        )

    vol_df = pd.concat(volume_data_list, ignore_index=True)

    # Save vol_df to feather format
    feather_output_path_vol_df = join(paths["feather_output"], "volume_tot.feather")
    vol_df.to_feather(feather_output_path_vol_df)
    print(f"Volume data saved to {feather_output_path_vol_df}")

    # Save fr_df and vol_df to Excel format as well
    excel_output_path_fr_df = join(paths["flow_metrics"], "flow_rate.xlsx")
    fr_df.to_excel(excel_output_path_fr_df, index=False)
    print(f"Flow rate data saved to {excel_output_path_fr_df}")

    excel_output_path_vol_df = join(paths["flow_metrics"], "Volume.xlsx")
    vol_df.to_excel(excel_output_path_vol_df, index=False)
    print(f"Volume data saved to {excel_output_path_vol_df}")

    # Save the updated clinical data back to its original file path and format
    clinical_data_output_path = paths["clinical_data"]
    file_ext = splitext(clinical_data_output_path)[1].lower()
    if file_ext == ".csv":
        clinical_data.to_csv(clinical_data_output_path, index=False)
        print(f"Updated clinical data saved to {clinical_data_output_path} (CSV format)")
    else:
        clinical_data.to_excel(clinical_data_output_path, index=False)
        print(f"Updated clinical data saved to {clinical_data_output_path} (Excel format)")

    return fr_df, vol_df


def compute_hemodynamic_parameters(
    paths,
    args,
):
    """
    Computes kinetic energy, viscous energy loss, vorticity, and q-criterion for each frame,
    merges with clinical data, and saves to a feather file.

    Args:
        paths (dict): A dictionary containing paths to input and output directories/files:
                      - 'clinical_data': Path to the clinical and demographic data file (Excel or CSV).
                      - 'fr_path': Directory containing flow rate CSV files (for timing calculation).
                      - 'base_path': Base directory for processed flow data containing .vti files.
                      - 'feather_output': Directory to save the processed q-criterion feather file.
        args (object): An object containing arguments:
                       - id_column (str): The name of the column in clinical_data that uniquely
                         identifies patients (e.g., "ID").
                       - clinical_columns (list): A list of strings specifying clinical
                         parameters (column names) to include in the output DataFrame.
                       - not_all (list, optional): A list of patient IDs to include.
                         Defaults to an empty list.
                       - exclude (list, optional): A list of patient IDs to exclude.
                         Defaults to an empty list.
                       - structure (list): A list of anatomical structures (strings) to process (e.g., "LV", "LA").
                       - interp_num (int): The number of time points to interpolate the data to.

    Returns:
        pd.DataFrame: The processed DataFrame containing q-criterion data and merged clinical parameters.

    Raises:
        FileNotFoundError: If clinical data file or flow data directory is not found.
        ValueError: If essential columns are missing from clinical data,
                    or if no valid patient directories/VTI files are found after filtering.
        Exception: For other unexpected errors during processing.
    """
    print("\nComputing q-criterion...")

    # --- 1. Load Clinical Data ---
    clinical_data = load_clinical_data(paths)

    # Validate essential args and ID column
    if not hasattr(args, "id_column") or not isinstance(args.id_column, str):
        raise ValueError("Argument 'args.id_column' must be provided and be a string.")

    if args.id_column not in clinical_data.columns:
        raise ValueError(f"The specified ID column '{args.id_column}' not found in clinical data.")

    if args.hr_column not in clinical_data.columns:
        clinical_data[args.hr_column] = np.nan
        print(
            f"Warning: '{args.hr_column}' column not found in clinical data. Defaulting to 60 for HR calculations."
        )

    clinical_lookup = clinical_data.set_index(args.id_column)
    user_clinical_cols = (
        args.clinical_columns
        if hasattr(args, "clinical_columns") and isinstance(args.clinical_columns, list)
        else []
    )

    # --- 2. Process Data Per Structure and Patient ---
    all_processed_data = []  # List to collect dictionaries for final DataFrame

    for st in args.structure:
        print(f"\nProcessing structure: {st}")

        patients = natsorted(glob(join(paths["vti_path"], st, "*")))

        if not patients:
            print(
                f"Warning: No patient directories found for structure '{st}' in '{paths['vti_path']}'. Skipping."
            )
            continue

        filtered_patients = []
        for p in patients:
            case_id = basename(p)
            if hasattr(args, "not_all") and isinstance(args.not_all, list) and args.not_all:
                if case_id not in args.not_all:
                    continue
            if hasattr(args, "exclude") and isinstance(args.exclude, list) and args.exclude:
                if case_id in args.exclude:
                    continue
            filtered_patients.append(p)
        patients = filtered_patients

        if not patients:
            print(f"Warning: No patients remaining for structure '{st}' after filtering. Skipping.")
            continue

        # Loop through the patients
        for p_dir in patients:
            case_id = basename(p_dir)
            print(f"Processing patient: {case_id} for structure: {st}")

            try:
                # Retrieve patient-specific clinical data
                if case_id not in clinical_lookup.index:
                    print(
                        f"Warning: Clinical data not found for ID: {case_id}. Skipping patient for Q-criterion calculation."
                    )
                    continue
                patient_clinical_row = clinical_lookup.loc[case_id]

                hr_value = patient_clinical_row.get("HR_4D", 60)
                if pd.isna(hr_value):
                    hr_value = 60

                # Get velocity arrays (.vti files)
                vti_files = natsorted(glob(join(p_dir, "*.vti")))
                if not vti_files:
                    print(
                        f"Warning: No .vti files found for patient '{case_id}' in '{p_dir}'. Skipping."
                    )
                    continue

                VEL, KE, DIV, VOR, QCR = [], [], [], [], []
                for vti_file in vti_files:
                    try:
                        tmp = pv.read(vti_file)

                        VEL.append(
                            np.reshape(tmp.point_data["velocity"], tmp.dimensions + (3,))
                        )  # Shape: (dim_x, dim_y, dim_z, 3)
                        KE.append(np.reshape(tmp.point_data["kinetic_energy"], tmp.dimensions))
                        DIV.append(np.reshape(tmp.point_data["divergence"], tmp.dimensions))
                        VOR.append(
                            np.reshape(
                                np.linalg.norm(tmp.point_data["vorticity"], axis=1), tmp.dimensions
                            )
                        )
                        QCR.append(np.reshape(tmp.point_data["qcriterion"], tmp.dimensions))

                    except Exception as e:
                        print(
                            f"Error reading VTI file {basename(vti_file)}: {e}. Skipping this file."
                        )
                        continue

                VEL_array = np.moveaxis(
                    np.array(VEL), 0, -1
                )  # VEL_array shape: (dim_x, dim_y, dim_z, 3, num_frames)
                KE_array = np.moveaxis(
                    np.array(KE), 0, -1
                )  # KE_array shape: (dim_x, dim_y, dim_z, num_frames)
                DIV_array = np.moveaxis(np.array(DIV), 0, -1)
                VOR_array = np.moveaxis(np.array(VOR), 0, -1)
                QCR_array = np.moveaxis(np.array(QCR), 0, -1)

                EL_array = calculate_viscous_energy_loss(
                    VEL_array, DIV_array, tmp.spacing, viscosity=0.0035  # Default viscosity value
                )

                nt = QCR_array.shape[-1]  # Number of time points
                dt = (
                    (60 / hr_value) / args.interp_num if args.interp_num > 0 else 0
                )  # Avoid division by zero

                # Voxel volume
                voxel_V = np.prod(tmp.spacing)  # Volume of a single voxel

                KE_t = cubic_interp(np.sum(KE_array, axis=(0, 1, 2)), args.interp_num)
                EL_t = cubic_interp(np.sum(EL_array, axis=(0, 1, 2)), args.interp_num)
                VOR_t = cubic_interp(np.sum(VOR_array, axis=(0, 1, 2)), args.interp_num) * voxel_V
                QCR_t = cubic_interp(np.sum(QCR_array, axis=(0, 1, 2)), args.interp_num) * voxel_V

                # Calculate QPos and QNeg counts for each time point
                QPos_per_frame = np.sum(QCR_array > 0, axis=(0, 1, 2))
                QNeg_per_frame = np.sum(QCR_array < 0, axis=(0, 1, 2))

                # Interpolate QPos and QNeg
                QPos_t = cubic_interp(QPos_per_frame, args.interp_num) * voxel_V
                QNeg_t = cubic_interp(QNeg_per_frame, args.interp_num) * voxel_V

                # Prepare dictionary for QPosX00 values
                QPos_threshold_interp = {}
                for i in range(1, 11):
                    threshold = i * 100
                    # Sum counts where qcriterion is >= threshold for each frame
                    Q_sum_at_threshold_per_frame = np.sum(QCR_array >= threshold, axis=(0, 1, 2))
                    QPos_threshold_interp[f"Qcriterion_pos{threshold}"] = (
                        cubic_interp(Q_sum_at_threshold_per_frame, args.interp_num) * voxel_V
                    )

                # Generate time and normalized time arrays
                time_points = np.linspace(0, nt * dt, args.interp_num)
                time_norm_points = np.linspace(0, 1, args.interp_num)
                frame_indices = np.arange(0, args.interp_num)

                # Collect data for each interpolated time point
                for i in range(args.interp_num):
                    record = {
                        "id": case_id,
                        "study": case_id.split("_")[
                            0
                        ],  # Assuming study is the first part of ID_Study
                        "structure": st,
                        "time": time_points[i],
                        "time_norm": time_norm_points[i],
                        "frame_idx": frame_indices[i],
                        "kinetic_energy": KE_t[i],
                        "viscous_energy_loss": EL_t[i],
                        "vorticity": VOR_t[i],
                        "qcriterion": QCR_t[i],
                        "qcriterion_pos": QPos_t[i],
                        "qcriterion_neg": QNeg_t[i],
                    }

                    # Add QPosX00 values
                    for k, v in QPos_threshold_interp.items():
                        record[k] = v[i]

                    # Add user-defined clinical columns
                    for col in user_clinical_cols:
                        record[col] = patient_clinical_row.get(col, "N/A")

                    all_processed_data.append(record)

            except Exception as e:
                print(
                    f"Error processing patient {case_id} for structure {st}: {e}. Skipping patient."
                )
                continue

    if not all_processed_data:
        raise ValueError(
            "No hemodynamic parameters could be computed. Check input data and processing logic."
        )

    # Create final DataFrame
    qcrit_df = pd.DataFrame(all_processed_data)

    # Save the dataframe to feather
    feather_output_path = join(paths["feather_output"], "hemodynamic_parameters.feather")
    qcrit_df.to_feather(feather_output_path)
    print(f"Hemodynamic parameters saved to {feather_output_path}")

    return qcrit_df
