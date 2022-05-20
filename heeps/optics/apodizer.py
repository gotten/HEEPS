from .circular_apodization import circular_apodization
import heeps.util.img_processing as impro
import proper
import numpy as np
import os.path
from astropy.io import fits 

def apodizer(wf, mode='RAVC', ravc_t=0.8, ravc_r=0.6, ngrid=1024, npupil=285,
        f_app_amp='', f_app_phase='', f_ravc_amp='', f_ravc_phase='',
        apo_misalign=None, onaxis=True, verbose=False, save_ring=False, **conf):

    ''' Create a wavefront object at the entrance pupil plane.
    The pupil is either loaded from a fits file, or created using
    pupil parameters.
    Can also select only one petal and mask the others.

    wf: WaveFront
        PROPER wavefront object
    mode: str
        HCI mode
    ravc_t: float
        RA transmittance
    ravc_r: float
        RA radius
    ngrid: int
        number of pixels of the wavefront array
    npupil: int
        number of pixels of the pupil
    f_app_amp: str
    f_app_phase: str 
        apodizing phase plate files
    f_ravc_amp: str
    f_ravc_phase: str 
        ring apodizer files (optional)
    apo_misalign: list of float
        apodizer misalignment

    '''

    # case 1: Ring Apodizer
    if 'RAVC' in mode and ravc_r > 0:

        # load apodizer from files if provided
        if os.path.isfile(f_ravc_amp) and os.path.isfile(f_ravc_phase):
            if verbose is True:
                print('   apply ring apodizer from files')
            # get amplitude and phase data
            RAVC_amp = fits.getdata(f_ravc_amp)
            RAVC_phase = fits.getdata(f_ravc_phase)
            # resize to npupil
            RAVC_amp = impro.resize_img(RAVC_amp, npupil)
            RAVC_phase = impro.resize_img(RAVC_phase, npupil)
            # pad with zeros to match PROPER gridsize
            RAVC_amp = impro.pad_img(RAVC_amp, ngrid)
            RAVC_phase = impro.pad_img(RAVC_phase, ngrid)
            # build complex apodizer
            ring = RAVC_amp*np.exp(1j*RAVC_phase)

        # else, define the apodizer as a ring (with % misalignments)
        else:
            # RAVC misalignments
            dx, dy = [0, 0] if apo_misalign is None else list(apo_misalign)[0:2]
            # create apodizer
            ring = circular_apodization(wf, ravc_r, 1, ravc_t, xc=dx, 
                yc=dy, NORM=True)
            if save_ring is True:
                fits.writeto('apo_ring_r=%.4f_t=%.4f.fits'%(ravc_r, ravc_t),
                    impro.crop_img(ring, npupil), overwrite=True)
            if verbose is True:
                print('   apply ring apodizer: ravc_t=%s, ravc_r=%s'
                    %(round(ravc_t, 4), round(ravc_r, 4)))

        # multiply the loaded apodizer
        proper.prop_multiply(wf, ring)

    # case 2: Apodizing Phase Plate
    elif 'APP' in mode:
        # get amplitude and phase data
        if os.path.isfile(f_app_amp):
            if verbose is True:
                print('   apply APP stop (amplitude)')
            APP_amp = fits.getdata(f_app_amp)
        else:
            APP_amp = np.ones((npupil, npupil))
        if os.path.isfile(f_app_phase) and onaxis == True:
            if verbose is True:
                print('   apply APP phase')
            APP_phase = fits.getdata(f_app_phase)
        else:
            APP_phase = np.zeros((npupil, npupil))
        # resize to npupil
        APP_amp = impro.resize_img(APP_amp, npupil)
        APP_phase = impro.resize_img(APP_phase, npupil)
        # rotate for negative PSF
        if 'neg' in mode:
            APP_amp = np.rot90(APP_amp, 2)
            APP_phase = np.rot90(APP_phase, 2)
        # pad with zeros to match PROPER ngrid
        APP_amp = impro.pad_img(APP_amp, ngrid, 0)
        APP_phase = impro.pad_img(APP_phase, ngrid, 0)
        
        # multiply the loaded APP
        proper.prop_multiply(wf, APP_amp*np.exp(1j*APP_phase))

    return wf