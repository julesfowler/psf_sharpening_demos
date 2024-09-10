### -- is this just an eye doctor knockoff? 

from astropy.io import fits
import io
import matplotlib.pyplot as plt
import numpy as np
import requests 

import ktl

## -- FUNCTIONS 



def apply_zernike(mode, amplitude, ao_running=False):

    zernike = build_zernike(mode, amplitude)
    apply_command(zernike, ao_running)


def build_zernike(mode, amplitude):
    
    #TODO build zernikes 
    # TODO put zernike file on orkid
    zernike_file = f'zernike_modes_wv=940_k2dm.fits'
    zernike_data = fits.getdata(zernike_file, mode)
    zernike_out = zernike_data * amplitude 

    return zernike_out


def apply_command(zernike, ao_running):
    # think about something iterative? 
    # I think this can be a later problem tho ...
    
    if ao_running:
        ktl.write('ao2', 'DMORVEC', zernike)
    else:
        # TODO put file on orkid 
        interaction_matrix = np.load(some_file)
        slopes = np.dot(interaction_matrix, zernike) 
        ktl.write('ao2', 'WSCNOR', slopes)


def background_subtract(image, bkg_reference):
    # given some region where the PSF isn't
    # take the median of the region and subtract 
    bkg_sub_image = image - np.median(bkg_reference)
    bkg_sub_image[bkg_sub_image < 0] = 0 

    return bkg_sub_image


def set_camera_parameters(subarray, exposure_time):
    # adapted from Shui's camera class 
    
    url = "SEKRET/setFrameParams"
    params = {'x':, 'y':, 'width':, 'height':, 'expTimeMS':}

    params_str = '&'.join([f'{key}={value}' for key, value in params.items()])
    response = requests.request("POST", url, data=params_str)


def take_exposure():
    
    url = "SEKRET/takeFitsExposure"
    # TODO does anything need to be here? 
    params = {}
    response = requests.request("POST", url, data=params)
    response.raise_for_status()

    return fits.getdata(io.BytesIO(response.content))


def build_image():

    stack = np.zeros((n_exps, subarray[2], subarray[2]))
    for i in range(n_exps):
        stack[i] = take_exposure()
    
    median_image = np.median(stack, axis=0)
    
    bkg_reference = median_image[:, -30:]
    reduced_image = background_subtract(median_image, bkg_reference) 
    
    return reduced_image

def run_psf_tuning(modes=11, amplitudes=np.arange(-3, 3, 0.5)):
   
    # if we are gonna update params it should probably be once right here 
    #set_camera_parameters()
    # I think we can set this from the camera gui and make it a later problem 

    sweep = {}
    
    for z in range(modes):
        
        sweep[z] = {}
        max_pix_array = []

        for amp in amplitudes:
            
            run = {}    
            run['zernike'] = z
            run['amplitude'] = amp
            
            apply_zernike(mode, amplitude)
            image = build_image()
            
            max_pix = np.max(image)
            max_pix_array.append(max_pix)
            run['image'] = image
            run['max_pix'] = max_pix
            if max_pix == np.max(max_pix_array):
                best_amp = amp
            
            sweep[z][amp] = run

        sweep[z]['optimal_amp'] = best_amp
        plt.plot(amplitudes, max_pix_array)
        plt.xlable('zernike amplitude [waves]')
        plt.ylable('max pixel value [photons]')
        plt.savefig(f'{z}_bright_pixel_optimal.png')
        plt.clf()
        
    return sweep

