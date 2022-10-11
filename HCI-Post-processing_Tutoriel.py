#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sun Sep 27 14:44:55 2020: Creation

@author: echoquet
"""

import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from copy import deepcopy
from time import time
plt.rcParams['image.origin'] = 'lower'


#% FUNCTIONS


def create_mask(rIn, shape, cent=None, polar_system=False):
    """ Creates a boolean frame with the pixels outside rIn at True.
    
    Parameters
    ----------
    rIn : the radius of the mask in pixels.
    shape : the shape of the output array (pix, pix).
    cent : (optional) the coordinates of the mask relative to the center. Default is the frame center.
    system : (optional) 'cartesian' or 'polar' system for the center coordinates. Default is cartesian.
        
    Returns
    -------
    mask : 2D Array of booleans, True outside rIn, False inside.
        
    """
    if len(shape) != 2:
        raise TypeError('Shape should be tuple of 2 elements')
    im_center = np.array(shape) / 2 - 0.5
    
    if cent is None:
        center = im_center
    else:
        if polar_system:
            phi = cent[1] *np.pi/180
            center = [cent[0] * np.cos(phi), -cent[0] * np.sin(phi)] + im_center
        else:
            center = cent + im_center
            
    y, x = np.indices(shape)
    rads = np.sqrt((y-center[0])**2+(x-center[1])**2)
    mask = rads >= rIn
    return mask


def create_annulus(rIn, rOut, dim):
    """ Creates a boolean array with the pixels within rIn and rOut at True.

    Parameters
    ----------
    rIn : the inner radius of the annulus in pixels.
    rOut : the outer radius of the annulus in pixels.
    dim : list with the shape of the output array [pix, pix].

    Returns
    -------
    mask : 2D Array of booleans, True within rIn and rOut, False outside.

    """
    if len(dim) != 2:
        raise TypeError('dim should be list of 2 elements')

    cent = (np.array(dim)-1) / 2  # minus 1 because of difference between length and indexation
    x, y = np.indices(dim)
    rads = np.sqrt((x-cent[0])**2+(y-cent[1])**2)
    mask = (rOut > rads) & (rads >= rIn)
    return mask


def radial_mean(array, step=1, rStart=0, rStop=None, width=3, quick_view=False):
    """ Computes the radial average of a 2D image.
    
    Parameters
    ----------
    array : the input image, 2D array.
    step : (optional) the radial step for the output array (pix).
    rStart : (optional) the first radius of the output array (pix).
    rStop : (optional) the last radius of the output array (pix).
    width : (optional) the width of the anulus used to compute the average. 
    quick_view: (optional) flag to show a basic plot of the output
        
    Returns
    -------
    mean_array : 1D array with the radial average values.
    rad_list : 1D array with the correspoding radial positions.
        
    """
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array')
        
    shape = array.shape
    if rStop is None:
        rStop = min(shape)/2.
    
    rad_list = np.arange(rStart, rStop, step)
    mean_array = [np.mean(array[create_annulus(rad-width/2., rad+width/2., shape)]) for rad in rad_list]
    
    if quick_view:
        fig,ax = plt.subplots(1,1,figsize=(12,8))
        ax.plot(rad_list, mean_array)
        ax.set_xlabel('Separation (pix)')
        ax.set_ylabel('Mean value (counts/s)')
        plt.show()
    
    return mean_array, rad_list
        
    
def radial_std(array, step=1, rStart=0, rStop=None, width=3, quick_view=False):
    """ Computes the radial standard deviation of a 2D image.
    
    Parameters
    ----------
    array : the input image, 2D array.
    step : (optional) the radial step for the output array (pix).
    rStart : (optional) the first radius of the output array (pix).
    rStop : (optional) the last radius of the output array (pix).
    width : (optional) the width of the anulus used to compute the std. 
    quick_view: (optional) flag to show a basic plot of the output
        
    Returns
    -------
    std_array : 1D array with the radial std values.
    rad_list : 1D array with the correspoding radial positions.
        
    """
    if array.ndim != 2:
        raise TypeError('Input array is not a frame or 2d array')
        
    shape = array.shape
    if rStop is None:
        rStop = min(shape)/2.
    
    rad_list = np.arange(rStart, rStop, step)
    std_array = [np.std(array[create_annulus(rad-width/2., rad+width/2., shape)]) for rad in rad_list]
    
    if quick_view:
        fig,ax = plt.subplots(1,1,figsize=(12,8))
        ax.plot(rad_list, std_array)
        ax.set_xlabel('Separation (pix)')
        ax.set_ylabel('Std value (counts/s)')
        plt.show()
    
    return std_array, rad_list
        
   

def frame_rotate_interp(array, angle, center=None, mode='constant', cval=0, order=3):
    ''' Rotates a frame or 2D array.

        Parameters
        ----------
        array : Input image, 2d array.
        angle : Rotation angle (deg).
        center : Coordinates X,Y  of the point with respect to which the rotation will be
                    performed. By default the rotation is done with respect to the center
                    of the frame; central pixel if frame has odd size.
        interp_order: Interpolation order for the rotation. See skimage rotate function.
        border_mode : Pixel extrapolation method for handling the borders.
                    See skimage rotate function.

        Returns
        -------
        rotated_array : Resulting frame.

    '''
    dtype = array.dtype
    dims = array.shape
    angle_rad = -np.deg2rad(angle)

    if center is None:
        center = (np.array(dims)-1) / 2  # The minus 1 is because of python indexation at 0

    x, y = np.meshgrid(np.arange(dims[1], dtype=dtype), np.arange(dims[0], dtype=dtype))

    xp = (x-center[1])*np.cos(angle_rad) + (y-center[0])*np.sin(angle_rad) + center[1]
    yp = -(x-center[1])*np.sin(angle_rad) + (y-center[0])*np.cos(angle_rad) + center[0]

    rotated_array = ndimage.map_coordinates(array, [yp, xp], mode=mode, cval=cval, order=order)

    return rotated_array


def derotate_and_combine(cube, angles):
    """ Derotates a cube of images then mean-combine them.
    
    Parameters
    ----------
    cube : the input cube, 3D array.
    angles : the list of parallactic angles corresponding to the cube.
        
    Returns
    -------
    image_out : the mean-combined image
    cube_out : the cube of derotated frames.
    
    """
    if cube.ndim != 3:
        raise TypeError('Input cube is not a cube or 3d array')
    if angles.ndim != 1:
        raise TypeError('Input angles must be a 1D array')
    if len(cube) != len(angles):
        raise TypeError('Input cube and input angle list must have the same length')
        
    shape = cube.shape
    cube_out = np.zeros(shape)
    for im in range(shape[0]):
        cube_out[im] = frame_rotate_interp(cube[im], -angles[im])
    
    image_out = np.nanmean(cube_out, axis=0)
    return image_out, cube_out


def principal_components(cube):
    """ Computes the principal components of a cube of images.
    
    Parameters
    ----------
    cube : the input cube, 3D array.
        
    Returns
    -------
    pc_cube : the cube of 2D principal components
    
    """
    if cube.ndim != 3:
        raise TypeError('Input cube is not a cube or 3d array')
    
    
    # Works with vectorized images (1D) instead of 2D images:
    shape = cube.shape
    matrix = cube.reshape((shape[0], shape[1]*shape[2]))
    
    # Subtracts the mean of each images for the covariance and the PCs:
    matrix_ms = matrix - np.reshape(np.mean(matrix, axis=1),(shape[0],1))
    
    # Computes the covariance matrix and its Eigen values and vectors:
    covariance = np.dot(matrix_ms, matrix_ms.T)
    e_values, e_vectors = np.linalg.eigh(covariance)
    
    # Computes the principal components, and sorts them from the strongest to the weakest:
    pc_tmp = np.dot(e_vectors.T, matrix_ms) / np.sqrt(np.abs(e_values.reshape((shape[0],1))))
    pc_vec = pc_tmp[::-1]
    
    # Reshapes the PCs into 2D images:
    pc_cube = pc_vec.reshape(shape)
    return pc_cube, pc_vec


def project_on_principal_components(sci_cube, pc_cube, trunc):
    """ Projects a cube of images on principal components using the proper shapes and normalizations.
    
    Parameters
    ----------
    sci_cube : the input cube to be fitted, 3D array.
    pc_cube : the PCs cube, 3D array
    trunc : the number of PCs to keep in the projection, int.
        
    Returns
    -------
    pca_models : the cube of 2D models for each input image.
    
    """
    if sci_cube.ndim != 3:
        raise TypeError('Input sci_cube is not a cube or 3d array')
    
    if pc_cube.ndim != 3:
        raise TypeError('Input pc_cube is not a cube or 3d array')
        
    # Works with vectorized images (1D) instead of 2D images:
    shape_pc = pc_cube.shape
    pc = pc_cube[:trunc].reshape(trunc, shape_pc[1]*shape_pc[2])
    
    # Works with vectorized images (1D) instead of 2D images:
    shape_cube = sci_cube.shape
    matrix = sci_cube.reshape(shape_cube[0], shape_cube[1]*shape_cube[2])
    
    # Subtracts the mean of each science image:
    matrix_means = np.mean(matrix, axis=1)
    matrix_ms = matrix - np.reshape(matrix_means,(shape_cube[0],1))
    
    # Projects once one the modes to get the coefficients, 
    # then once again to sum the weighted modes:
    coeffs = np.dot(matrix_ms, pc.T)
    pca_models_vec = np.dot(coeffs, pc) + np.reshape(matrix_means,(shape_cube[0],1))
    
    # Reshapes the models into 2D images:
    pca_models = pca_models_vec.reshape(shape_cube) 
    return pca_models


def create_planet_image(template, separation, position_angle, shape):
    """ Creates an image with a fake at a specific position.
    
    Parameters
    ----------
    template : the template psf used to simulate a planet, 2D array.
    separation : separation to the image center, in pixels.
    position_angle : Position angle from the vertical vounter-clockwise, in degrees.
    shape : shape of the output image, tuple of int.
        
    Returns
    -------
    pca_models : the cube of 2D models for each input image.
    
    """    
    if template.ndim != 2:
        raise TypeError('Input template is not an image or 2d array')
        
    if len(shape) != 2:
        raise TypeError('Input template is not an image or 2d array')
    
    temp_shape = template.shape
    diff0 = (shape[0]-temp_shape[0])
    diff1 = (shape[1]-temp_shape[1])
    
    temp_large = np.pad(template, ((int((diff0-diff0%2)/2),int((diff0+diff0%2)/2)),(int((diff1-diff1%2)/2),int((diff1+diff1%2)/2))))
    
    phi = position_angle *np.pi/180
    dy = separation * np.cos(phi)
    dx = -separation * np.sin(phi)
    planet_image = np.roll(temp_large, (int(round(dy)),int(round(dx))), axis = (0,1))
    return planet_image

def inject_planet_in_cube(cube, cube_angles, template, contrast, separation, position_angle):
    """ Inject a fake planet in a cube at a fixed position in the sky, knowing the parallactic angles of the cube.
    
    Parameters
    ----------
    cube : the cube in which we want to inject a planet
    cuble_angles : the parallactic angles of the cube
    template : the template psf used to simulate a planet, 2D array.
    contrast : the contrast of the simulated planet compared to the template
    separation : separation to the image center, in pixels.
    position_angle : Position angle from the vertical vounter-clockwise, in degrees.
        
    Returns
    -------
    new_cube : the cube of with the planet injected.
    
    """        
    if cube.ndim != 3:
        raise TypeError('Input cube is not a cube or 3d array')
    
    if cube_angles.ndim != 1:
        raise TypeError('Input angles must be a 1D array')
    
    if len(cube) != len(cube_angles):
        raise TypeError('Input cube and input angle list must have the same length')
    
    if template.ndim != 2:
        raise TypeError('Input template is not an image or 2d array')
    
    image_shape = (cube[0]).shape
    
    new_cube = deepcopy(cube)
    for k, angle in enumerate(cube_angles):
        new_cube[k] += (create_planet_image(template, separation, position_angle-angle, image_shape)*contrast)
    
    return new_cube
    

#%% Imports the data


path_data = '/Users/echoquet/Documents/Research/Code_Workspace/Python_workspace/2020-HRA_School_draft/Dataset_2'
target_name = 'Target-2'


if not os.path.exists(path_data):
    print('ERROR! Folder path_data does not exist.')

if not os.path.exists(os.path.join(path_data, target_name + '_science_cube.fits')):
    print('ERROR! target_name does not exist.')
   
# Imports the science data cube, the associated parallactic angles, and the unobscured PSF.
# The data have a 4th dimension because the instrument has two cameras working in parallel (with different filters).
# We focus on a single camera in this tutorial (cam=0) for sake of simplicity.
# The unobscured PSF is used to calibrate the contrast to the star, and is used as template PSF to simulate fake planets.
# The unit of the images (data cube and PSF) is count/s.
# cam = 0
# sci_cube = fits.getdata(os.path.join(path_data, target_name + '_science_cube.fits'))[cam]
# unobscured_psf = np.mean(fits.getdata(os.path.join(path_data, target_name + '_psf.fits'))[cam],axis=0)

# # sci_cube = fits.getdata(os.path.join(path_data, target_name + '_science_cube.fits'))
# # print(sci_cube.shape)
# # dimout=300
# # dimin=sci_cube.shape[2]
# # trunc=int((dimin-dimout)/2)
# # sci_cube = sci_cube[:,trunc:-trunc,trunc:-trunc]
# # print(sci_cube.shape)

# hdu0 = fits.PrimaryHDU(sci_cube)
# hdu0.writeto(os.path.join(path_data, target_name + '_science_cube.fits'), overwrite=True)

# # # psf = np.mean(fits.getdata(os.path.join(path_data, target_name + '_psf_cube.fits'))[cam],axis=0)
# plt.imshow(unobscured_psf)
# print(unobscured_psf.shape)

# hdu00 = fits.PrimaryHDU(unobscured_psf)
# hdu00.writeto(os.path.join(path_data, target_name + '_psf.fits'), overwrite=True)

sci_cube = fits.getdata(os.path.join(path_data, target_name + '_science_cube.fits'))
sci_angles = fits.getdata(os.path.join(path_data, target_name + '_science_derot.fits'))
unobscured_psf =  fits.getdata(os.path.join(path_data, target_name + '_psf.fits'))
sci_shape = sci_cube.shape

print('Number of images in sci_cube: {}'.format(sci_shape[0]))
print('Image size in sci_cube: {}x{} pix'.format(sci_shape[1],sci_shape[2]))


# Some useful values:
pixel_size = 0.01225  # arcsec/pix
stellar_max_value = np.max(unobscured_psf) #count/s


# Inject a fake planet or not in the raw data:
inject_Planet_Q = False
if inject_Planet_Q:
    separation = 1.2  # in arcsec
    pos_angle = 120.  # in deg.
    contr = 1e-4
    sci_cube = inject_planet_in_cube(sci_cube, sci_angles, unobscured_psf, contr, separation/pixel_size, pos_angle)



# Here we derotate all the raw images to be North up, and mean-combine them without PSF subtraction
raw_im, _ = derotate_and_combine(sci_cube, sci_angles)


# The following shows the results without starlight subtraction:
# The mean-conbined image, the radial mean profile, and the radial STD profile.
width = 5
mean_raw, rad_list = radial_mean(raw_im, width=width)
std_raw, rad_list = radial_std(raw_im, width=width)

vmax = 10
fig, ax = plt.subplots(1,1,figsize=(4,4), dpi=600)
ax.imshow(raw_im, vmin=-0.5*vmax, vmax=vmax)
ax.set_title('Raw data combined')

fig,ax = plt.subplots(1,1,figsize=(6,4), dpi=600)
ax.semilogy(rad_list*pixel_size, mean_raw/stellar_max_value, label='Raw image')
ax.set_xlabel('Separation (arcsec)')
ax.set_ylabel('Radial average profile')
# ax.set_ylim([5e-8,4e-3])
ax.set_title('Raw contrast profile')

fig,ax = plt.subplots(1,1,figsize=(6,4), dpi=600)
ax.semilogy(rad_list*pixel_size, std_raw/stellar_max_value, label='Raw image')
ax.set_xlabel('Separation (arcsec)')
ax.set_ylabel('Radial STD profile')
ax.set_ylim([5e-8,4e-3])
ax.set_title('Radial variations')

plt.show()




#%% Observing strategy

# For ADI, the PSF library is simply the science dataset:
ref_cube = deepcopy(sci_cube)

# For RDI you have to import another dataset
# path_data_ref = '/Users/echoquet/Documents/Research/Code_Workspace/Python_workspace/2020-HRA_School_draft/Dataset_2'
# ref_name = 'Target-2'
# ref_cube = fits.getdata(os.path.join(path_data_ref, ref_name + '_science_cube.fits'))#[cam]
# print(ref_cube.shape)

#%% Classical subtraction

mean_model = np.mean(ref_cube,axis=0)
red1_cube = sci_cube - mean_model
red1_im, _ = derotate_and_combine(red1_cube, sci_angles)



vmax = 2
fig,ax = plt.subplots(1,1,figsize=(8,8))
ax.imshow(red1_im, vmin=-0.5*vmax, vmax=vmax)
ax.set_title('Classical subtraction combined')
plt.show()

# hdu1 = fits.PrimaryHDU(red1_cube)
# hdu1.writeto(os.path.join(path_data, target_name + '_red1_cube.fits'), overwrite=True)

# hdu2 = fits.PrimaryHDU(red1_im)
# hdu2.writeto(os.path.join(path_data, target_name + '_red1_image.fits'), overwrite=True)

# hdu3 = fits.PrimaryHDU(mean_model)
# hdu3.writeto(os.path.join(path_data, target_name + '_red1_model.fits'), overwrite=True)


width = 5
std_raw, rad_list = radial_std(raw_im, width=width)
std_red1, rad_list = radial_std(red1_im, width=width)

fig,ax = plt.subplots(1,1,figsize=(6,4),dpi=800)
ax.semilogy(rad_list*pixel_size, std_raw/stellar_max_value, label='Raw image')
ax.semilogy(rad_list*pixel_size, std_red1/stellar_max_value, label='Classical subtraction')
ax.set_xlabel('Separation (arcsec)')
ax.set_ylabel('Std deviation contrast')
ax.set_ylim([5e-8,4e-3])
ax.legend(loc='best')
plt.show()


#%% PCA


pc_cube, pc_vec = principal_components(ref_cube)
print(np.dot(pc_vec[0],pc_vec[10]))
print(np.dot(pc_vec[10],pc_vec[10]))


#See some PCs
# comp = [1, 20]
# vmax = 0.5
# fig,axes = plt.subplots(1,2,figsize=(8,4))
# axes[0].imshow(pc_cube[comp[0]], vmin=-0.5*vmax, vmax=vmax)
# axes[1].imshow(pc_cube[comp[1]], vmin=-0.5*vmax, vmax=vmax)
# axes[0].set_title('PC {}'.format(comp[0]))
# axes[1].set_title('PC {}'.format(comp[1]))

# hdu4 = fits.PrimaryHDU(pc_cube)
# hdu4.writeto(os.path.join(path_data, target_name + '_pca_modes.fits'), overwrite=True)



trunc_list = [1,3,5,10,15,20]
# trunc_list = [10,30,50,70,150,200]
red2_im_list = np.zeros((len(trunc_list), sci_shape[1], sci_shape[2]))
t1 = time()
for k,trunc in enumerate(trunc_list):
    pca_models = project_on_principal_components(sci_cube, pc_cube, trunc)
    red2_cube = sci_cube - pca_models
    red2_im_list[k], _ = derotate_and_combine(red2_cube, sci_angles)
t2= time()
print('Computation time: {} s'.format(t2-t1))

# hdu5 = fits.PrimaryHDU(red2_im_list)
# hdu5.writeto(os.path.join(path_data, target_name + '_pca_images.fits'), overwrite=True)

k=3
vmax=0.5
fig,ax = plt.subplots(1,1,figsize=(8,8))
ax.imshow(red2_im_list[k], vmin=-0.5*vmax, vmax=vmax)
ax.set_title('PCA subtraction combined, trunc={}'.format(trunc_list[k]))
plt.show()


# Throughput
im_shape = sci_cube[0].shape
paList = np.arange(0,90,15)   # degrees
sepList = np.arange(int(im_shape[0]/2))   # pixels
# throughput_list_5 = np.zeros((len(sepList),len(paList)))
throughput_list = np.zeros((len(sepList),len(paList)))
for i,sep in enumerate(sepList):
    for kk, pa in enumerate(paList):
        planet = create_planet_image(unobscured_psf,sep, pa, im_shape)
        aperture = ~create_mask(10, im_shape, cent=[sep, pa], polar_system=True)
        planet_flux = np.sum(planet[aperture])
    
        planet_projected = project_on_principal_components(np.reshape(planet,(1,im_shape[0],im_shape[1])), pc_cube, trunc_list[k])[0]
        planet_processed = planet - planet_projected
        throughput_list[i,kk] = np.sum(planet_processed[aperture])/planet_flux
throughput_mean = np.mean(throughput_list, axis=1)

fig,ax = plt.subplots(1,1,figsize=(6,4),dpi=800)
ax.plot(sepList*pixel_size, throughput_mean, label='trunc = {} PCs'.format(trunc_list[k]))
ax.set_xlabel('Separation (arcsec)')
ax.set_ylabel('PCA planet Throughput')
ax.set_title('PCA planet oversubtraction')
ax.legend(loc='best')


#%% Radial contrast curves

# k=3

width = 5
std_raw, rad_list = radial_std(raw_im, width=width)
std_red1, rad_list = radial_std(red1_im, width=width)
std_pca, rad_list = radial_std(red2_im_list[k], width=width)

fig,ax = plt.subplots(1,1,figsize=(6,4),dpi=800)
ax.semilogy(rad_list*pixel_size, std_raw/stellar_max_value, label='Raw image')
ax.semilogy(rad_list*pixel_size, std_red1/stellar_max_value, label='Classical subtraction')
ax.semilogy(rad_list*pixel_size, std_pca/stellar_max_value, label='PCA trunc={}'.format(trunc_list[k]))
ax.set_xlabel('Separation (arcsec)')
ax.set_ylabel('Std deviation contrast')
ax.set_ylim([5e-8,4e-3])
ax.legend(loc='best')
plt.show()