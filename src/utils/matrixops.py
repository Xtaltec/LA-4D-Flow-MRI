import nibabel as nib
import numpy as np


def phase_to_velocity(phase_image, venc):
    """Phase image range: 0-4096, with 2048 as 0 velocity (in m/s)"""
    return (phase_image - 2048.0) / 2048.0 * venc / 100.0


def vel_to_ms(vel):
    """Convert the velocity from mm/s to m/s and ensure the spacing is in mm"""
    if np.log10(np.mean(np.abs(vel))) > 1:
        vel = vel / 1000

    return vel


def sp_to_mm(spacing):
    """Convert the spacing from m to mm"""
    if np.log10(np.mean(spacing)) < -1:
        spacing = tuple([i * 1000 for i in spacing])
    elif np.log10(np.mean(spacing)) > 1:
        spacing = tuple([i / 1000 for i in spacing])
    return spacing


def sp_to_m(spacing):
    """Convert the spacing from mm to m"""
    mean_log = np.log10(np.mean(spacing))
    if mean_log < -4:
        spacing = [i * 1000 for i in spacing]
    elif mean_log > -2:
        spacing = [i / 1000 for i in spacing]
    return tuple(float(s) for s in spacing)


def orient_to_RAS(path):
    """Read the nifti file orient as RAS and return the image and the transformation matrix"""

    # Read the nifti file and orient to RAS
    img = nib.load(path)
    img_or = nib.as_closest_canonical(img)

    # Get the pixel spacing
    spacing = tuple(img_or.header["pixdim"][1:4])

    # Get the transformation matrix
    T = img_or.affine.dot(np.linalg.inv(img.affine))

    # Save the nifti file
    nib.save(img_or, path)

    # Return the image and the transformation matrix
    return img_or, spacing, T


def divergence(f, sp):
    """Computes divergence of vector field
    f: array -> vector field components [Fx,Fy,Fz,...]
    sp: array -> spacing between points in respecitve directions [spx, spy,spz,...]
    https://stackoverflow.com/questions/67970477/compute-divergence-with-python/67971515#67971515
    """
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], sp[i], axis=i) for i in range(num_dims)])


def visc_energy_loss(v, spacing, mu=0.0035, L=2.1869100000000003e-09):
    # Initialize the energy loss
    phi_v = np.zeros((v.shape[0], v.shape[1], v.shape[2], v.shape[4]))

    div = divergence(np.moveaxis(v, 3, 0), spacing)

    # Viscous dissipation per voxel
    for i in range(3):
        for j in range(3):
            # Calculate the terms of the equation
            term1 = np.gradient(v[:, :, :, j, :], axis=i) + np.gradient(v[:, :, :, i, :], axis=j)
            term2 = (2 / 3) * div if i == j else 0

            # Calculate the result for this point
            phi_v += (term1 - term2) ** 2

    phi_v = 0.5 * phi_v

    # Viscous dissipation per voxel and time step
    E = mu * phi_v * L

    # Volumetric rate of viscous energy loss per time instance
    E_t = np.apply_over_axes(np.sum, E, [0, 1, 2])

    # Total volumetric rate of viscous energy loss
    E_tot = np.sum(E_t)

    return E, E_t.squeeze(), E_tot
