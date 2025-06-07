from glob import glob
from ntpath import basename
from os.path import isfile, join
from pathlib import Path

import gmsh
import nibabel as nib
import numpy as np
import pyevtk
import pymeshfix
import pyvista as pv
import SimpleITK as sitk
import vtk
from natsort import natsorted, ns
from scipy.ndimage import median_filter

from src.utils.matrixops import sp_to_m
from src.utils.misc import join_create


def read_vti(path_in):
    """Reads the .vti files from the path_in folder and returns the velocity array"""
    vti_files = natsorted(glob(join(path_in, "*.vti")), alg=ns.IGNORECASE)
    v = []
    for vti_file in vti_files:
        vti = pv.read(vti_file)
        v += [vti["velocity"].reshape(vti.dimensions + (3,), order="F")]
    vel = np.stack(v, axis=-1)
    points = vti.points.reshape(vti.dimensions + (3,), order="F")
    spacing = vti.spacing

    return vel, spacing, points


def masking(path_in, vel, points, crop=1):
    """Reads the mask and masks the velocity array"""
    # Read the mask
    mask_file = glob(path_in + "*.nii.gz")[0]
    mask = nib.load(mask_file).get_fdata()

    # Mask the velocity
    vel_masked = vel * mask[..., np.newaxis, np.newaxis]

    # Crop all the the smallest cube containing the ROI
    if crop:
        # Get the indices of the ROI
        idx = np.where(mask)
        # Get the min and max indices
        min_idx = np.min(idx, axis=1)
        max_idx = np.max(idx, axis=1)
        # Crop the velocity
        vel_masked = vel_masked[
            min_idx[0] : max_idx[0],
            min_idx[1] : max_idx[1],
            min_idx[2] : max_idx[2],
            :,
            :,
        ]
        # Crop the points
        points = points[
            min_idx[0] : max_idx[0], min_idx[1] : max_idx[1], min_idx[2] : max_idx[2], :
        ]

    return vel_masked, points


def save_masked(path_in, case_id, vel_masked, points_masked, VEL, spacing, structure):
    path_out = join_create(path_in, "..", structure, str(case_id))

    voxel_mass = spacing[0] * spacing[1] * spacing[2] * 1060

    for i in range(vel_masked.shape[-1]):
        grid = pv.UniformGrid()
        grid.dimensions = vel_masked.shape[:-2]
        grid.origin = points_masked[0, 0, 0]
        grid.spacing = spacing
        grid.point_data["velocity"] = vel_masked[..., i].reshape(-1, 3, order="F")
        grid.point_data["velocity_magnitude"] = np.linalg.norm(vel_masked[..., i], axis=-1).reshape(
            -1, order="F"
        )
        grid = grid.compute_derivative(
            scalars="velocity", divergence=True, vorticity=True, qcriterion=True
        )
        grid["kinetic_energy"] = 0.5 * grid["velocity_magnitude"] ** 2 * voxel_mass
        grid["viscous_energy_loss"] = VEL[..., i].reshape(-1, order="F")
        grid.save(join(path_out, case_id + "_" + str(i).zfill(3) + ".vti"))


def save_vti(vel, spacing, origin, path_out, case):
    """Save the velocity matrix as .vti in path_out"""

    join_create(path_out, case)
    for i in range(vel.shape[-1]):
        pyevtk.hl.imageToVTK(
            join(path_out, case, case + "_" + str(i).zfill(3)),
            spacing=sp_to_m(spacing),
            origin=origin,
            pointData={
                "velocity": (
                    np.asfortranarray(vel[:, :, :, 0, i]),
                    np.asfortranarray(vel[:, :, :, 1, i]),
                    np.asfortranarray(vel[:, :, :, 2, i]),
                ),
                "velocity_magnitude": np.asfortranarray(
                    np.sqrt(
                        vel[:, :, :, 0, i] ** 2 + vel[:, :, :, 1, i] ** 2 + vel[:, :, :, 2, i] ** 2
                    )
                ),
            },
        )


def save_masked_evtk(path_in, case_id, vel_masked, points_masked, VEL, spacing, structure):
    path_out = join_create(path_in, "..", structure, str(case_id))

    voxel_mass = spacing[0] * spacing[1] * spacing[2] * 1060

    for i in range(vel_masked.shape[-1]):
        grid = pv.ImageData()
        grid.dimensions = vel_masked.shape[:-2]
        grid.origin = tuple(float(p) for p in points_masked[0, 0, 0])
        grid.spacing = spacing
        grid.point_data["velocity"] = vel_masked[..., i].reshape(-1, 3, order="F")
        grid = grid.compute_derivative(
            scalars="velocity", divergence=True, vorticity=True, qcriterion=True
        )
        vorticity = grid["vorticity"].reshape(grid.dimensions + (3,), order="F")

        pyevtk.hl.imageToVTK(
            join(path_out, case_id + "_" + str(i).zfill(3)),
            origin=tuple(float(p) for p in points_masked[0, 0, 0]),
            spacing=spacing,
            pointData={
                "velocity": (
                    np.asfortranarray(vel_masked[:, :, :, 0, i]),
                    np.asfortranarray(vel_masked[:, :, :, 1, i]),
                    np.asfortranarray(vel_masked[:, :, :, 2, i]),
                ),
                "velocity_magnitude": np.asfortranarray(
                    np.linalg.norm(vel_masked[:, :, :, :, i], axis=3)
                ),
                "kinetic_energy": np.asfortranarray(
                    0.5 * np.linalg.norm(vel_masked[:, :, :, :, i], axis=3) ** 2 * voxel_mass
                ),
                "viscous_energy_loss": np.asfortranarray(VEL[:, :, :, i]),
                "qcriterion": np.asfortranarray(
                    grid["qcriterion"].reshape(grid.dimensions, order="F")
                ),
                "qcriterion_median": np.asfortranarray(
                    median_filter(
                        grid["qcriterion"].reshape(grid.dimensions, order="F"),
                        size=5,
                        mode="constant",
                        cval=0.0,
                    )
                ),
                "divergence": np.asfortranarray(
                    grid["divergence"].reshape(grid.dimensions, order="F")
                ),
                "vorticity": (
                    np.asfortranarray(vorticity[:, :, :, 0]),
                    np.asfortranarray(vorticity[:, :, :, 1]),
                    np.asfortranarray(vorticity[:, :, :, 2]),
                ),
            },
        )


def nii_to_stl(path_in, path_out, case_id, smoothing=100, rewrite=False):
    """Convert nii to smoothed stl file,
    based on https://github.com/DavidTu21/Multilabel_Niftis2Stl"""

    Path(path_out).mkdir(parents=True, exist_ok=True)
    file_path = glob(join(path_in, case_id + "*.nii.gz"))[0]
    stl_path = join(path_out, case_id + ".stl")

    if not rewrite and isfile(stl_path):
        print("File already exists, skipping")
    else:
        # read all the labels present in the file
        multi_label_image = sitk.ReadImage(file_path)
        img_npy = sitk.GetArrayFromImage(multi_label_image)
        labels = np.unique(img_npy)

        # read the file
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(file_path)
        reader.Update()

        # for all labels presented in the segmented file
        for label in labels:
            if int(label) != 0:
                # apply marching cube surface generation
                surf = vtk.vtkDiscreteMarchingCubes()
                surf.SetInputConnection(reader.GetOutputPort())
                surf.SetValue(0, int(label))
                surf.Update()

                # Change the coordinates to meters
                transform = vtk.vtkTransform()
                transform.Scale(0.001, 0.001, 0.001)
                transformFilter = vtk.vtkTransformPolyDataFilter()
                transformFilter.SetInputConnection(surf.GetOutputPort())
                transformFilter.SetTransform(transform)
                transformFilter.Update()

                # smoothing the mesh
                smoother = vtk.vtkWindowedSincPolyDataFilter()
                if vtk.VTK_MAJOR_VERSION <= 5:
                    smoother.SetInput(transformFilter.GetOutput())
                else:
                    smoother.SetInputConnection(transformFilter.GetOutputPort())

                # increase this integer set number of iterations if smoother surface wanted
                smoother.SetNumberOfIterations(smoothing)
                smoother.NonManifoldSmoothingOn()
                smoother.NormalizeCoordinatesOn()
                smoother.GenerateErrorScalarsOn()
                smoother.Update()

                # save the output
                writer = vtk.vtkSTLWriter()
                writer.SetInputConnection(smoother.GetOutputPort())
                writer.SetFileTypeToASCII()

                # Set output folder
                writer.SetFileName(stl_path)
                writer.Write()

        # Open mesh and repair with pymeshfix
        pymeshfix.clean_from_file(stl_path, stl_path)


def meshing3D(path_in, path_out, case_id, rewrite=True, extension=".vtk"):
    """Meshes 3D STL files using Gmsh"""

    output_file = join(path_out, case_id + extension)

    if isfile(output_file) and not rewrite:
        print("\n", basename(output_file) + " already exists. Skipping...")
    else:
        # Mesh the 3D STL file
        file_path = glob(join(path_in, case_id + "*.stl"))[0]
        Path(path_out).mkdir(parents=True, exist_ok=True)
        try:
            # Gmsh 3D meshing
            gmsh.initialize()
            gmsh.merge(file_path)
            surf = gmsh.model.getEntities(2)
            loop = gmsh.model.geo.addSurfaceLoop([surf[i][1] for i in range(len(surf))])
            gmsh.model.geo.addVolume([loop])
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(3)
            gmsh.write(output_file)
            gmsh.finalize()
        except Exception as e:
            gmsh.finalize()
            print(f"Case {case_id} failed: {e}")
