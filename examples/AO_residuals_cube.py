#!/usr/bin/env python3

# =============================================================================
#       Example file for creating a coronagraphic/non-coronagraphic METIS PSF
# =============================================================================

""" Required libraries """
import matplotlib.pyplot as plt # for plotting simulated PSFs
import numpy as np
from astropy.io import fits 
from heeps.config import conf, download_from_gdrive
from heeps.pupil import pupil
from heeps.abberations import wavefront_abberations
from heeps.coronagraphs import apodization, vortex, lyotstop, lyot
from heeps.detector import detector
import os.path
import copy
import time


""" Default simulation configuration defined in "read_config.py" can be 
overridden here by updating the dictionary """
lams = (3.8e-6, 4.8e-6, 8.7e-6, 11.5e-6)
pscales = (5, 5, 10, 10)
conf['ONAXIS'] = True
conf['STATIC_NCPA'] = False
conf['MODE'] = 'RAVC'
conf['CL_DIAM'] = 4 # classical lyot diam in lam/D

for pscale,lam in zip(pscales,lams):
    
    conf['PIXEL_SCALE'] = pscale
    conf['WAVELENGTH'] = lam
    lamstr = str(round(conf['WAVELENGTH']*1e9))
    
    """ Pupil """
    conf['PUPIL_FILE'] = 'ELT_2048_37m_11m_5mas_nospiders_cut.fits'
    wfo, PUP = pupil(conf, get_pupil='amp')
    
    """ Loading tip-tilt values """
    conf['tip_tilt'] = (0, 0)
    # conf['tip_tilt'] = np.random.randn(conf['TILT_CUBE'], 2)
    
    """ Loading AO residual values, getting multi-cube phase screen from Google Drive """
    if True:
        AO_residuals_cube = fits.getdata(conf['INPUT_DIR'] + conf['ATM_SCREEN_CUBE'])[:15]
    else:
        AO_residuals_cube = np.array([None])
    ncube = AO_residuals_cube.shape[0]
    print('ncube = %s'%ncube)
    
    psfs = None
    t0 = time.time()
    for i in range(ncube):
        print(i)
        wf = copy.copy(wfo)
        wavefront_abberations(wf, AO_residuals=AO_residuals_cube[i], **conf)
        
        """ Coronagraph modes"""
        if conf['MODE'] == 'ELT': # no Lyot stop (1, -0.3, 0)
            conf['LS_PARAMS'] = [1., -0.3, 0.]
            _, PUP = lyotstop(wf, conf, get_pupil='amp')
        elif conf['MODE'] in ('CL', 'CL4', 'CL5'): # classical lyot
            conf['LS_PARAMS'] = [0.8, 0.1, 1.1]
            if conf['ONAXIS'] == True:
                lyot(wf, conf)
#            _, PUP = lyotstop(wf, conf, get_pupil='amp')
            lyotstop(wf, conf)
        elif conf['MODE'] == 'VC':
            if conf['ONAXIS'] == True:
                vortex(wf, conf)
#            _, PUP = lyotstop(wf, conf, get_pupil='amp')
            lyotstop(wf, conf)
        elif conf['MODE'] == 'RAVC':
            RAVC = True
            apodization(wf, conf, RAVC=RAVC)
            if conf['ONAXIS'] == True:
                vortex(wf, conf)
#            _, PUP = lyotstop(wf, conf, RAVC=RAVC, get_pupil='amp')
            lyotstop(wf, conf, RAVC=RAVC)
        elif conf['MODE'] == 'APP': 
            APP = True
            if conf['ONAXIS'] == True:
#                _, PUP = lyotstop(wf, conf, APP=APP, get_pupil='phase')
                lyotstop(wf, conf, APP=APP)
        
        """ Science image """
        psf = detector(wf, conf)
        
        # stack psfs
        if psfs is None:
            psfs = psf[None, ...]
        else:
            psfs = np.concatenate([psfs, psf[None, ...]])
    
    print(time.time() - t0)
    if ncube == 1:
        psfs = psfs[0]
    
    """ write to fits """
    # File name
    conf['PREFIX'] = ''
    filename = '%s%s_lam%s'%(conf['PREFIX'], conf['MODE'], lamstr)
    # save PSF (or PSFs cube)
    if True:
        fits.writeto(os.path.join(conf['OUT_DIR'], 'PSF_' + filename) \
                + '.fits', psfs, overwrite=True)
    # save Lyot-Stop
    if False:
        fits.writeto(os.path.join(conf['OUT_DIR'], 'LS_' + filename) \
                + '.fits', PUP, overwrite=True)
    
    """ Save figures in .png """
    if False:
        plt.figure()
        #plt.imshow(psf**0.05, origin='lower')
        plt.imshow(np.log10(psf/1482.22), origin='lower') # 1482.22 is peak in ELT mode
        plt.colorbar()
        plt.show(block=False)
        plt.savefig(os.path.join(conf['OUT_DIR'], 'PSF_' + filename) \
                + '.png', dpi=300, transparent=True)
        plt.figure()
        plt.imshow(PUP[50:-50,50:-50], origin='lower', cmap='gray', vmin=0, vmax=1)
        #plt.imshow(PUP[50:-50,50:-50], origin='lower')
        plt.colorbar()
        plt.show(block=False)
        plt.savefig(os.path.join(conf['OUT_DIR'], 'LS_' + filename) \
                + '.png', dpi=300, transparent=True)
