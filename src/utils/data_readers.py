import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from glob import glob
from ntpath import basename
from os.path import isdir, isfile, join
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
import pydicom
import pyvista as pv
from natsort import natsorted, ns
from nipype.interfaces.dcm2nii import Dcm2niix

from src.utils.matrixops import orient_to_RAS, phase_to_velocity, vel_to_ms
from src.utils.mesh import save_vti
from src.utils.misc import join_create


def dcmtonii(inp_dir, out_dir):
    """Convert the DICOM files to NIfTI using dcm2niix."""
    # Initialize the Dcm2niix converter
    converter = Dcm2niix()

    # Set the input and output directories
    converter.inputs.source_dir = inp_dir
    converter.inputs.output_dir = out_dir

    # Set additional options
    converter.inputs.compression = 5  # Compression level (0-9)
    converter.inputs.merge_imgs = True  # Merge DICOMs into one 3D file

    # Build the command line
    cmd = converter.cmdline

    # Execute the command line in a subprocess
    subprocess.call(cmd, shell=True)


def read_magnitude(case, path_in, path_out):
    """Call the center specific function to read the magnitude image"""

    path_raw = join(path_in, case)
    file_out = join(path_out, case + ".nii.gz")

    if isfile(file_out):
        mag, spacing, _ = orient_to_RAS(file_out)

    elif isdir(path_raw):
        if case.split("_")[0] in ["cardiohance", "tof"]:
            magnitude_ABI(path_raw, path_out, case)
        elif case.split("_")[0] in ["laao", "laaop", "paf", "hcm"]:
            magnitude_Clinic(path_raw, path_out, case)
        mag, spacing, _ = orient_to_RAS(file_out)
    else:
        mag, spacing = None, None
    return mag, spacing


def read_velocity(case, path_in, path_out, mag_spacing=None):
    """Call the center specific function to read the velocity image"""

    if not glob(join(path_out, case, "*vti")):
        if case.split("_")[0] in ["cardiohance", "tof"]:
            velocity_ABI(path_in, path_out, case)
        elif case.split("_")[0] in ["laao", "laaop", "paf", "hcm"]:
            velocity_Clinic(path_in, path_out, case, mag_spacing)

    vti_files = natsorted(glob(join(path_out, case, "*.vti")), alg=ns.IGNORECASE)
    v = []
    for vti_file in vti_files:
        vti = pv.read(vti_file)
        v += [vti["velocity"].reshape(vti.dimensions + (3,), order="F")]
    vel = np.stack(v, axis=-1)
    spacing = vti.spacing
    origin = vti.origin

    return vel, spacing, origin


def magnitude_Clinic(path_in, path_out, case):
    """Read the magnitude dicom files, orient as RAS and
    return the image and the transformation matrix"""

    # Find the files in the folder
    f = glob(join(path_in, "*.zip"))
    f = f[0] if len(f) > 0 else f
    print(f)

    path_temp = None
    path_temp2 = None

    try:
        t = []
        # If DICOM is zip, uncompress in temporal directory
        if ".zip" in f:
            while ".zip" in f:
                # path_temp = join_create(path_in,'temp'+str(count))
                path_temp = tempfile.mkdtemp()
                print(path_temp)
                t += [path_temp]
                with zipfile.ZipFile(f, "r") as compr:
                    compr.extractall(path_temp)
                f = glob(join(path_temp, "*"))[0]
            [shutil.rmtree(i) for i in t[:-1]]
            path_temp2 = tempfile.mkdtemp()
            print(path_temp2)
            dcmtonii(path_temp, path_temp2)
            shutil.rmtree(path_temp)
            path_temp = None  # Mark as cleaned up
        else:
            path_temp2 = join_create(path_in, "temp2")
            dcmtonii(path_in, path_temp2)

        # Find the 4 biggest files and keep the first one based on natural order
        four = natsorted(
            natsorted(glob(join(path_temp2, "*.nii.gz")), key=lambda y: os.path.getsize(y))[-4:]
        )
        for fo in four:
            dim = nib.load(fo).shape[0:3]
            if all(i > 100 for i in dim):
                mag = fo
                print(fo)
                break

        # Define new name and copy to output folder
        mag_name = join(path_out, case + ".nii.gz")
        shutil.copyfile(mag, mag_name)
        shutil.rmtree(path_temp2)

    except (OSError, IOError, zipfile.BadZipFile, IndexError) as e:
        print(f"Error processing files: {e}")
        # Clean up any remaining temporary directories
        try:
            if path_temp and os.path.exists(path_temp):
                shutil.rmtree(path_temp)
        except OSError as cleanup_error:
            print(f"Warning: Could not clean up {path_temp}: {cleanup_error}")

        try:
            if path_temp2 and os.path.exists(path_temp2):
                shutil.rmtree(path_temp2)
        except OSError as cleanup_error:
            print(f"Warning: Could not clean up {path_temp2}: {cleanup_error}")

        raise


def magnitude_ABI(path_in, path_out, case):
    """Read the magnitude dicom files, orient as RAS and
    return the image and the transformation matrix"""

    # Get the folder contents
    dicom_folders = glob(join(path_in, "*"))

    # Get the first folder with M in the name
    mag_folder = [i for i in dicom_folders if "M" in basename(i)][0]

    # Create a temporal folder
    path_temp = tempfile.mkdtemp()

    # Convert the dicom to nifti
    dcmtonii(mag_folder, path_temp)

    # Read the nifti file rename and delete the json file
    file_nii = glob(join(path_temp, "*.nii.gz"))[0]
    shutil.copyfile(file_nii, join(path_out, case + ".nii.gz"))
    shutil.rmtree(path_temp)


def velocity_ABI(path_in, path_out, case):
    """Read the velocity dicom files and return the
    velocity matrix, spacing and origin oriented as RAS"""

    # Check if vti in path_out, dont overwrite previous results
    if len(glob(join(path_out, case, "*.vti"))) == 0:
        if isdir(join(path_in, case)):
            vti_files = natsorted(glob(join(path_in, case, "*.vti")), alg=ns.IGNORECASE)
            v = []
            for vti_file in vti_files:
                vti = pv.read(vti_file)
                v += [
                    np.stack((vti["U"], vti["W"], vti["V"]), axis=-1).reshape(
                        vti.dimensions + (3,), order="F"
                    )
                ]

            vel = np.stack(v, axis=-1)
            spacing = list(vti.spacing)

            # Set the correct axis direction
            vel = -np.flip(np.flip(np.flip(np.moveaxis(vel, 1, 2), axis=0), axis=1), axis=2)
            spacing[0], spacing[2] = spacing[2], spacing[0]

            origin = (0, 0, 0)

        else:
            path_mag = path_in.replace("velocity", join("magnitude", case))
            files = natsorted(glob(join(path_mag, "*")), alg=ns.IGNORECASE)
            phase_folders = natsorted([i for i in files if "_P_" in basename(i)], alg=ns.IGNORECASE)

            v = []
            with TemporaryDirectory() as temp_dir:
                for i, phase_folder in enumerate(phase_folders):
                    # Get the VENC from the a reference file
                    refDs = pydicom.read_file(glob(join(phase_folder, "**", "*"))[0])
                    venc = int(re.findall(r"\d+", refDs.SequenceName)[-1])
                    # Get the velocity for each direction on space
                    with TemporaryDirectory() as temp_dir2:
                        dcmtonii(phase_folder, temp_dir2)
                        shutil.copyfile(
                            glob(join(temp_dir2, "*.nii.gz"))[0],
                            join(temp_dir, case + "_P_" + str(i) + ".nii.gz"),
                        )
                        ph, spacing, _ = orient_to_RAS(
                            join(temp_dir, case + "_P_" + str(i) + ".nii.gz")
                        )
                        v += [phase_to_velocity(ph.get_fdata(), venc)]

            vel = np.stack(v, axis=3)
            # spacing =tuple(ph.header['pixdim'][1:4])
            origin = (0, 0, 0)

        vel = vel_to_ms(vel)

        save_vti(vel, spacing, origin, path_out, case)


def velocity_Clinic(path_in, path_out, case, mag_spacing=None, origin=(0, 0, 0)):
    """Read the velocity dicom files and return the velocity matrix,
    spacing and origin orient as RAS"""

    if mag_spacing:
        spacing = mag_spacing
    else:
        spacing = (1.4063, 1.4063, 1.1)

    npy = glob(join(path_in, case, "*.npy"))[0]

    # Orient the velocity to RAS
    vel = np.load(npy)
    vel = np.flip(np.flip(np.transpose(vel), axis=0), axis=1)
    vel[:, :, :, 0] = -vel[:, :, :, 0]
    vel[:, :, :, 1] = -vel[:, :, :, 1]

    vel = vel_to_ms(vel)

    save_vti(vel, spacing, origin, path_out, case)
