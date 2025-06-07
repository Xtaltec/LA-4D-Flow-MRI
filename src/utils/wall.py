import logging
import os
from glob import glob
from ntpath import basename
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from natsort import natsorted

from src.utils.misc import not_all_exclude
from utils.analysis import load_clinical_data

logging.basicConfig()
logger = logging.getLogger("wss_utils")


def extract_vectors(polydata):
    """
    Extract vector from separate columns on polydata and stack them to (n,3) vector
    """
    u = polydata["u"]
    v = polydata["v"]
    w = polydata["w"]
    vector = np.stack((u, v, w), axis=-1)
    return vector


def get_orthogonal_vectors(vectors, point_normals):
    logger.debug("Get orthogonal vectors")
    logger.debug("Actual vector {}".format(vectors[0:2]))

    # Calculate the scalar v.n
    c = vectors * point_normals
    c = np.sum(c, axis=1)

    # Get normal and tangent vector
    normal_vectors = c[:, np.newaxis] * point_normals
    tangent_vectors = vectors - normal_vectors

    logger.debug("Normal vector {}".format(normal_vectors[0:2]))
    logger.debug("Tangent vector {}".format(tangent_vectors[0:2]))

    return normal_vectors, tangent_vectors


def get_vector_magnitude(vectors):
    """
    Calculate vector magnitude |v| from an array of (n,3)
    """
    c = vectors * vectors
    c = np.sum(c, axis=1)
    c = c**0.5
    return c


def _calculate_gradient_with_values(
    pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag, inward_distance, use_parabolic
):
    """
    Fitting polynomial for multiple rows of set of points, then calculate the gradient
    Based on: https://stackoverflow.com/questions/20202710/numpy-polyfit-and-polyval-in-multiple-dimensions
    """
    logger.info("Calculating gradient for {} points".format(pc1_tangent_mag.shape))
    # Prepare to calculate the slopes for each points
    x = np.array([0, 1, 2])  # We have 3 points to evaluate
    x = x * inward_distance  # Get the correct distance scaling

    # Stack them so it has (n x (v0, v1, v2))
    y = np.stack((pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag), axis=1)

    # Calculate an n-1 polynomial
    y = np.transpose(y)
    z = np.polynomial.polynomial.polyfit(x, y, len(x) - 1)

    # Create new X with more points to obtain a smooth curve
    if use_parabolic:
        x_new = np.linspace(x[0], x[-1], len(x) * 5)
    else:
        x_new = x

    # Evaluate the fitted function with an evenly distributed x
    y_new = np.polynomial.polynomial.polyval(x_new, z)

    # Get all the gradient at once
    gg = np.gradient(y_new, x_new, axis=1)

    # Only return the gradient at the wall
    return gg[:, 0], x_new, y_new


def calculate_gradient(
    pc0_tangent_mag,
    pc1_tangent_mag,
    pc2_tangent_mag,
    inward_distance,
    use_parabolic=True,
):
    gradients, x_, y_ = _calculate_gradient_with_values(
        pc0_tangent_mag,
        pc1_tangent_mag,
        pc2_tangent_mag,
        inward_distance,
        use_parabolic,
    )
    # Only return the gradients, x_ and y_ for test purpose ONLY
    return gradients


def wss_calculator(
    vtk_path: str,
    vti_path: str,
    output_path: str,
    parabolic: int = 0,
    no_slip: int = 1,
    viscosity: float = 0.0035,
) -> pv.PolyData:
    """
    Computes the Wall Shear Stress (WSS) from velocity field data by interpolating velocity vectors
    onto a surface mesh and calculating tangential shear gradients.

    Parameters:
    ----------
    vtk_path : str
        Path to the input VTK file containing the surface mesh.
    vti_path : str
        Path to the input VTI file containing the velocity field data.
    parabolic : int, optional, default=0
        Flag to determine whether to use parabolic interpolation in gradient computation.
    no_slip : int, optional, default=1
        If set to 1, applies the no-slip boundary condition at the wall.
    viscosity : float, optional, default=0.0035
        Dynamic viscosity of the fluid (in appropriate units).

    Returns:
    -------
    pyvista.PolyData
        The computed surface mesh with added WSS data.

    Notes:
    ------
    - The function interpolates velocity vectors from the volume data to the surface mesh.
    - Wall shear stress is computed based on velocity gradients at the boundary.
    - The processed output is saved in a structured directory under "data/processed/wss".
    """

    # Extract structure and case identifiers
    case_id = basename(vtk_path).split(".")[0]
    it = basename(vti_path).split("_")[-1].split(".")[0]

    # Read the VTK and VTI files
    vtk_mesh = pv.read(vtk_path)
    vti_data = pv.read(vti_path)

    # Compute inward distance based on voxel size
    inward_distance = np.mean(vti_data.spacing)

    # Interpolate velocity vectors to the mesh
    vtk_mesh = vtk_mesh.sample(vti_data, tolerance=inward_distance * 2)

    # Convert point data to cell data and extract velocity components
    mesh = vtk_mesh.point_data_to_cell_data()
    mesh.cell_data["u"] = mesh.cell_data["velocity"][:, 0].flatten(order="F")
    mesh.cell_data["v"] = mesh.cell_data["velocity"][:, 1].flatten(order="F")
    mesh.cell_data["w"] = mesh.cell_data["velocity"][:, 2].flatten(order="F")

    # Store velocity magnitude and set it as the active scalar
    mesh.cell_data["Velocity"] = mesh.cell_data["velocity_magnitude"].flatten(order="F")
    mesh.set_active_scalars("Velocity")

    # Extract surface and compute normals
    surf = vtk_mesh.extract_surface()
    surf = surf.compute_normals(
        point_normals=True, cell_normals=True, inplace=True, flip_normals=True
    )

    # Define surface points and their inward normal displacement
    pc0, pc1, pc2 = [pv.PolyData(surf.points) for _ in range(3)]
    pc1.points = pc0.points + (inward_distance * surf.point_normals)
    pc2.points = pc0.points + (2 * inward_distance * surf.point_normals)

    # Convert cell data back to point data
    mesh = mesh.cell_data_to_point_data()

    # Sample velocity data at displaced points
    pc0, pc1, pc2 = pc0.interpolate(mesh), pc1.interpolate(mesh), pc2.interpolate(mesh)
    pc0.set_active_scalars("Velocity")
    pc1.set_active_scalars("Velocity")
    pc2.set_active_scalars("Velocity")

    # Compute tangential velocity components
    if no_slip:
        pc0_tangent_mag = np.zeros(len(pc0.points))
    else:
        pc0_vectors = extract_vectors(pc0)
        _, pc0_tangent = get_orthogonal_vectors(pc0_vectors, surf.point_normals)
        pc0_tangent_mag = get_vector_magnitude(pc0_tangent)

    pc1_vectors = extract_vectors(pc1)
    _, pc1_tangent = get_orthogonal_vectors(pc1_vectors, surf.point_normals)
    pc1_tangent_mag = get_vector_magnitude(pc1_tangent)
    pc1["vectors"] = pc1_tangent

    pc2_vectors = extract_vectors(pc2)
    _, pc2_tangent = get_orthogonal_vectors(pc2_vectors, surf.point_normals)
    pc2_tangent_mag = get_vector_magnitude(pc2_tangent)
    pc2["vectors"] = pc2_tangent

    # Adjust magnitude of pc2_tangent to maintain directional consistency
    dot_product = np.sum(pc1_tangent * pc2_tangent, axis=1).clip(min=-1, max=1)
    pc2_tangent_mag *= dot_product

    # Compute shear gradients
    gradients = calculate_gradient(
        pc0_tangent_mag, pc1_tangent_mag, pc2_tangent_mag, inward_distance, use_parabolic=parabolic
    )

    # Compute and store wall shear stress data
    surf["wss"] = gradients * viscosity
    surf["wss_vectors"] = pc1_tangent

    # Convert point data to cell data and save results
    surf = surf.point_data_to_cell_data()
    surf.save(join(output_path, f"{case_id}_{it}.vtk"))

    return surf


def compute_wall_parameters(paths, args):
    BASE_PATH = os.getenv("BASE_PATH", "./")
    Path(join(BASE_PATH, "data", "processed", "wss")).mkdir(parents=True, exist_ok=True)

    volume_path = join(BASE_PATH, "data", "processed", "segmentation", "vtk")
    flow_path = join(BASE_PATH, "data", "processed", "flow")

    clinical_data = load_clinical_data(paths)
    clinical_lookup = clinical_data.set_index(args.id_column)

    for s in args.structure:
        wss_path = join(BASE_PATH, "data", "processed", "wss", s)
        ecap_path = join(BASE_PATH, "data", "processed", "ecap", s)

        Path(join(wss_path)).mkdir(parents=True, exist_ok=True)
        Path(join(ecap_path)).mkdir(parents=True, exist_ok=True)

        patients = natsorted(glob(join(volume_path, s, "*")))
        patients = not_all_exclude(patients, args.not_all, args.exclude)

        for vtk in patients:
            try:
                case_id = basename(vtk).split(".")[0]
                print(case_id)

                # Get volume list
                vti_list = natsorted(glob(join(flow_path, s, case_id, "*.vti")))

                # Get the heart rate
                if case_id not in clinical_lookup.index:
                    print(
                        f"Warning: Clinical data not found for id: {case_id} (using '{args.id_column}' as identifier). Using default heart rate of 60 bpm."
                    )

                patient_clinical_row = clinical_lookup.loc[case_id]
                hr_value = patient_clinical_row.get(args.hr_column, 60)
                if pd.isna(hr_value):
                    hr_value = 60

                beatD = 60 / hr_value

                wssm = []
                wssv = []

                for vti in vti_list:
                    print(f"Processing {vti}")
                    temp = wss_calculator(vtk, vti, wss_path).cell_data_to_point_data()
                    wssm.append(temp.point_data["wss"])
                    wssv.append(temp.point_data["wss_vectors"])

                normals = -temp.point_data["Normals"]
                wssM = np.stack(wssm, axis=1)
                wssV = np.stack(wssv, axis=1)

                # %% Hemodynamic Parameters
                time = np.linspace(0, beatD, num=len(vti_list))
                dt = time[1] - time[0]

                # Calculation of the 3 components of tawss
                tawssX = np.trapezoid(wssV[:, :, 0], dx=dt) / beatD
                tawssY = np.trapezoid(wssV[:, :, 1], dx=dt) / beatD
                tawssZ = np.trapezoid(wssV[:, :, 2], dx=dt) / beatD

                # Tawss magnitude
                tawssT = np.linalg.norm(np.transpose(np.vstack((tawssX, tawssY, tawssZ))), axis=1)

                # Compute the 3 hemodynamic parameters
                TAWSS = np.trapezoid(wssM, dx=dt) / beatD
                OSI = 1 / 2 * (1 - np.divide(tawssT, TAWSS))
                ECAP = OSI / TAWSS

                surf = pv.read(vtk).extract_surface()

                surf.point_data["ECAP"] = np.asfortranarray(ECAP)
                surf.point_data["TAWSS"] = np.asfortranarray(TAWSS)
                surf.point_data["OSI"] = np.asfortranarray(OSI)
                surf.point_data["Normals"] = np.asfortranarray(normals)
                surf.save(join(ecap_path, case_id + ".vtk"))
            except Exception:
                print("Error in {}".format(case_id))
                continue
