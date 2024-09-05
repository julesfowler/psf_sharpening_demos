## -- IMPORTS
from astropy.io import fits
import glob
import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import resize 
from skimage.restoration import unwrap_phase

from image_sharpening import FocusDiversePhaseRetrieval, ft_rev, mft_rev
from image_sharpening import InstrumentConfiguration

from prysm.polynomials import (
    noll_to_nm,
    zernike_nm_sequence,
    lstsq
)
from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.geometry import circle,spider

## -- functions

def build_median_image(files, size):
    # Assuming we take odd # of multiple frames and want to median combine
    # them to use as our images -- this will do just that... 
    stack = np.zeros((len(files), size[0], size[1]))
    for index, data_file in enumerate(files):
        stack[index] = np.array(fits.getdata(data_file).reshape(size[0],
            size[1]), dtype=float)
    median_image = np.median(stack, axis=0)
    return median_image


def background_subtract(image, bkg_reference):
    # given some region where the PSF isn't
    # take the median of the region and subtract 
    bkg_sub_image = image - np.median(bkg_reference)
    bkg_sub_image[bkg_sub_image < 0] = 0 

    return bkg_sub_image


def recenter(image, size, center='bright'):
    # Recenter the image 
    # Ideally we want to do this to the brightest pixel
    # But was finding that the defocus messes up the PSF enough to change where
    # the center would be
    # So you can either recent on brightest pixel OR pass it a center 
    # this way we can find the center from the more in focus image and use that
    # for the defocused ones 
    # note this is rough logic -- assumes that we can crop a region (dictated by
    # 'size') around the new center of the PSF -- this won't work well if the 
    # PSF isn't totally in frame 
    if center=='bright':
        center = np.where(image == np.max(image))
        center_x, center_y = (center[0][0], center[1][0])
    else:
        center_x, center_y = center
     
    recentered_image = image[center_x - int(size/2):center_x + int(size/2),
            center_y - int(size/2):center_y + int(size/2)]
    return recentered_image, (center_x, center_y)

def smooth(image, gaussian_size):
    # smooths the image in the pupil plane by multiplying the image by a
    # gaussian in the focal plane 
    # the smaller gaussian size is, the tighter the profile of the gaussian is
    # in the focal plane, and the more smoothing you apply in the pupil plane
    # this is *NOT* the same as convolving a gaussian -- it was Jaren's rec
    # specfically as a precursor to the image sharpening logic 
    x_dim, y_dim = np.shape(image)
    x = np.arange(-1*x_dim/2, x_dim/2)
    y = np.arange(-1*x_dim/2, x_dim/2)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2) 
    gauss = np.exp(-(r/gaussian_size)**2)

    return image*gauss

def apply_image_sharpening(psf_list, pixel_dx, distance_list, steps=200, pix_to_um=4.837, wavelength=0.94):
    # aplly Jaren's image sharpening function (phase diversitiy) 
    # takes a list of PSFs (as image/arrays), a distance across the image in
    # pixels, a list of the defocus distances in mm, the number of steps we want
    # the algorithm to run, and the pixel to micron conversion of the detector

    # convert the pixel to physical units in mm
    dx = pixel_dx*pix_to_um
    dx_list = [dx, dx, dx]
    #wavelength=0.94 # set wavelength to 0.94 um, this should be an argument 
   




    epd = 10950 # in mm
    orkid_params = {'image_dx': 12.2, # um 
                    'efl': 34.3 * epd, # ORKID effective focal length, mm
                    'wavelength': 0.94, # FIXME: ORKID center wavelength, microns
                    'pupil_size': epd, # Keck entrance pupil diameter
                    }

    orkid_conf = InstrumentConfiguration(orkid_params)

    # build the mp object from the phase retreival code 
    mp = FocusDiversePhaseRetrieval(psf_list, wavelength, dx_list, distance_list)
    for i in range(200):
        psf_est = mp.step()
    
    pup_phase_est = np.angle(mft_rev(psf_est, orkid_conf))

    # current HACK -- seeing weird edge effects -- this is me cropping them out... 
    # applying a fourier transform to the image, and then using np.angle to
    # scoop out the associated phase 
    #pup_phase_est = np.angle(ft_rev(psf_est[30:90, 30:90]))
    #pup_phase_est = np.angle(ft_rev(psf_est))
    # now do phase unwrapping -- so many options lol
    # phase_unwrap_2d is an algo custom written by Jaren (source code is in
    # here) -- it provides smoother unwrapping but potentially less accurate and
    # repeatable, unwrap_phase is the scikit-image option, it was working well
    # for me in sim but not producing super great results here -- apparently it
    # does not do so hot with real image artifacts 
    # or we could leave it wrapped and hope the DM doesn't get too mad about
    # discontinuities 
    unwrap_phase_est = phase_unwrap_2d(pup_phase_est)
    #unwrap_phase_est = unwrap_phase(pup_phase_est)
    #unwrap_phase_est = pup_phase_est 

    return unwrap_phase_est


def convert_phase_to_dm_cmd(pupil_phase, mask, wv=0.94):
    # once we have the phase in radians, we need to resize it to the DM
    # resolution, convert from radians to microns to volts, and then mask it
    # into the flat DM shape
    resized_img = resize(pupil_phase, (21,21))
    # convert to microns, and then 0.6 um per v 
    cmd = resized_img * wv * 0.6 / (2*np.pi)*mask
    cmd_flat = cmd[mask==1] 

    return cmd, cmd_flat


## stolen from jaren 
def phase_unwrap_2d(phase_wrapped):

    """phase unwrapping routine based on the phaseunwrap2d.go script in IDL and the following proceedings:
    M.D. Pritt; J.S. Shipman, "Least-squares two-dimensional phase unwrapping using FFT's",
    IEEE Transactions on Geoscience and Remote Sensing ( Volume: 32, Issue: 3, May 1994),
    DOI: 10.1109/36.297989

    Uses a finite differences approach to determine the partial derivative of the wrapped phase in x and y,
    then solves the solution in the fourier domain

    TODO: Test this function against the prior in IDL, it doesn't appear to reconstruct phase well

    Parameters
    ----------
    phase_wrapped : numpy.ndarray
        array containing 2D signal to unwrap

    Returns
    -------
    numpy.ndarray
        unwrapped phase
    """

    imsize = phase_wrapped.shape
    M = imsize[0]
    N = imsize[1]

    Nmirror = 2 * (N )
    Mmirror = 2 * (M )

    phmirror = np.ones([Mmirror,Nmirror])

    # Quadrant 3
    phmirror[:M,:N] = phase_wrapped

    # First mirror reflection Quadrant 2
    phmirror[M:,:N] = np.flipud(phase_wrapped)

    # Second mirror reflection Quadrant 4
    phmirror[:M,N:] = np.fliplr(phase_wrapped)

    # Final reflection Quadrant 1
    phmirror[M:,N:] = np.flipud(np.fliplr(phase_wrapped))

    phroll = np.zeros_like(phmirror)
    phroll[:M,:N-1] = phmirror[:M,1:N]
    phroll[:M,N-1] = phmirror[:M,0]
    deltafd = phroll-phmirror

    pluspi = np.pi*np.ones_like(phmirror)
    mask = (deltafd > pluspi).astype(int)

    deltafd = deltafd - mask*2*np.pi
    negpi = -pluspi
    mask = (deltafd < negpi).astype(int)
    deltafd = deltafd + mask * 2 * np.pi
    deltafdx = deltafd

    # compute forward difference
    phroll = np.zeros_like(phmirror)
    phroll[:M-1,:N] = phmirror[1:M,:N]
    phroll[M,:N] = phmirror[0,:N]
    deltafd = phroll - phmirror

    pluspi = np.pi*np.ones_like(phmirror)
    mask = (deltafd > pluspi).astype(int)
    deltafd = deltafd - mask*2*np.pi
    negpi = -pluspi
    mask = (deltafd < negpi).astype(int)
    deltafd = deltafd + mask * 2 * np.pi
    deltafdy = deltafd

    # Solve system of equations formed by min LS -> phi
    D_n = np.fft.fft2(deltafdx)
    D_m = np.fft.fft2(deltafdy)
    inc_n = 2 * np.pi / Nmirror
    inc_m = 2 * np.pi / Mmirror

    nn = np.ones([Mmirror,1]) @ (np.arange(Nmirror))[np.newaxis]
    mm = np.ones([Nmirror,1]) @ (np.arange(Mmirror))[np.newaxis]
    mm = mm.transpose()
    
    i = 1j
    mult_n = np.ones([Mmirror,Nmirror]) - np.exp(-nn * i * inc_n)
    mult_m = np.ones([Mmirror,Nmirror]) - np.exp(-mm * i * inc_m)
    divisor = (np.cos(mm*inc_m) + np.cos(nn*inc_n) - np.ones([Mmirror,Nmirror])*2)*2
    divisor[0,0] = 1
    phi = (D_n*mult_n + D_m*mult_m) / divisor
    phi[0,0] = 0
    phi = np.fft.ifft2(phi)[:M,:N]
    phout = np.real(phi)
    return phout

def unpack_zernikes(phase_img, n_modes, modes_to_exclude):
    npix = phase_img.shape[0]
    x, y = make_xy_grid(npix, diameter=1)
    r, t = cart_to_polar(x, y)

    nzern = 100
    nms = [noll_to_nm(i) for i in range(nzern)]
    basis = np.array(list(zernike_nm_sequence(nms, r, t, norm=True)))
    coefs = lstsq(basis, phase_img)
    fit_polys = np.array([c*b for c,b in zip(coefs, basis)])
    reconstructed_image = np.sum(fit_polys, axis=0)
    return reconstructed_image

## -- big scripty chunk of operations 

if __name__ == "__main__":

    # pull in files
    # our first set had an even number so zap one 
    files_0 = glob.glob('0/*.fits')
    files_2 = glob.glob('2/*.fits')
    files_4 = glob.glob('4/*.fits')
    files_neg = glob.glob('2_neg/*.fits')

    # file prep shenanigans 
    # start with gaussian = 8 for our smoothing
    # 200pix across

    # 1. median combine the images
    # 2. background subtract the images 
    # 3. recenter the images 
    # 4. smooth the images 

    # median combine time
    psf_focused = build_median_image(files_0, (150,150))
    
    #plt.imshow(np.log10(psf_focused))
    #plt.colorbar()
    #plt.savefig('raw_focused.png')
    #plt.clf()

    psf_2 = build_median_image(files_2, (150,150))
    
    #plt.imshow(np.log10(psf_2))
    #plt.colorbar()
    #plt.savefig('raw_2mm.png')
    #plt.clf()

    psf_4 = build_median_image(files_4, (150,150))
    
    #plt.imshow(np.log10(psf_4))
    #plt.colorbar()
    #plt.savefig('raw_4mm.png')

    psf_2_neg = build_median_image(files_neg, (150,150))
    
    #plt.imshow(np.log10(psf_2_neg))
    #plt.colorbar()
    #plt.savefig('raw_-2mm.png')
    
    # background subtraction 
    bkg_reference = psf_focused[:, -30:]
    bkg_sub_focused = background_subtract(psf_focused, bkg_reference)
    bkg_sub_2 = background_subtract(psf_2, bkg_reference)
    bkg_sub_4 = background_subtract(psf_4, bkg_reference)
    bkg_sub_2_neg = background_subtract(psf_2_neg, bkg_reference)
    
    # recentring 
    recenter_focus, center = recenter(bkg_sub_focused, 110, center='bright')
    recenter_2, _ = recenter(bkg_sub_2, 110, center=center)
    recenter_4, _ = recenter(bkg_sub_4, 110, center=center)
    recenter_2_neg, _ = recenter(bkg_sub_2_neg, 110, center=center)
    
    # smoothing
    #gauss = 8
    #smooth_focus = smooth(recenter_focus, gauss)
    
    # and now's a good time to save some outputs 
    #plt.imshow(smooth_focus)
    #plt.colorbar()
    #plt.savefig('smoothed_focus.png')
    #plt.clf()

    #smooth_2 = smooth(recenter_2, gauss)
    #plt.imshow(smooth_2)
    #plt.colorbar()
    #plt.savefig('smoothed_2mm.png')
    #plt.clf()

    #smooth_4 = smooth(recenter_4, gauss)
    #plt.imshow(smooth_4)
    #plt.colorbar()
    #plt.savefig('smoothed_4mm.png')
    #plt.clf()

    #smooth_2_neg = smooth(recenter_2_neg, gauss)
    #plt.imshow(smooth_2_neg)
    #plt.colorbar()
    #plt.savefig('smoothed_-2mm.png')
    #plt.clf()


    # Now we can build up our list and do our psf sharpening
    #psf_list = [recenter_focus[29:81, 29:81], 
    #            recenter_2[29:81, 29:81],
    #            recenter_4[29:81, 29:81], 
    #            recenter_2_neg[29:81, 29:81]]
    psf_list = [recenter_focus, recenter_2, recenter_4, recenter_2_neg]
    distance_list = [-2e3, -4e3, 2e3]
    pixel_dx = 110 # 52 
    phase_est = apply_image_sharpening(psf_list, pixel_dx, distance_list, steps=200, pix_to_um=4.837)
   
    
    plt.imshow(phase_est)
    plt.colorbar()
    plt.savefig('phase_est.png')
    plt.clf()

    unpacked_phase_est = unpack_zernikes(phase_est, 300, 3)
    plt.imshow(unpacked_phase_est)
    plt.colorbar()
    plt.savefig('phase_est_zern.png')
    plt.clf()
    
    # And finally turn it into a command and write it out 
    mask = np.loadtxt('../../../flats/k2_dm_mask.txt')
    plt.imshow(phase_est)
    plt.colorbar()
    plt.plot('full_phase.png')
    plt.clf()

    cmd, cmd_apply = convert_phase_to_dm_cmd(phase_est, mask)
    cmd_zern, cmd_apply_zern = convert_phase_to_dm_cmd(unpacked_phase_est, mask)

    plt.imshow(cmd)
    plt.colorbar()
    plt.savefig('cmd_out.png')
    plt.clf()

    plt.imshow(cmd_zern)
    plt.colorbar()
    plt.savefig('cmd_out_zern.png')
    plt.clf()

    hdulist = fits.HDUList([fits.PrimaryHDU(cmd)])
    hdulist.writeto('cmd_out.fits', overwrite=True)
    hdulist = fits.HDUList([fits.PrimaryHDU(cmd_zern)])
    hdulist.writeto('cmd_out_zern.fits', overwrite=True)
