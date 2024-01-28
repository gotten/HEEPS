import heeps
import sys
from astropy.io import fits
import vip_hci
import numpy as np
import matplotlib.pyplot as plt
from heeps.util.psf_template import psf_template
import numpy as np

from heeps.util.img_processing import crop_cube, get_radial_profile
from vip_hci.var.shapes import get_annular_wedge
import matplotlib.pyplot as plt

mode="RAVC"
band="HCI_L_long"

mode="CVC"
band="N2"

if mode in ["RAVC","CVC"]:
    wedge=(0,360)
    if band in ["HCI_L_long","CO_ref"]:
        transmission=fits.getdata("/home/gpplotten/heeps_metis/input_files/optics/oat_{0}_{1}.fits".format("L",mode))
    else:
        transmission=fits.getdata("/home/gpplotten/heeps_metis/input_files/optics/oat_{0}_{1}.fits".format("N2",mode)) # 2D array with coronagraphic transmission [1] vs pixels distance [0]

    # this is only the hardware term where a planet close to the coronagraph focal plane mask also gets nulled by the coronagraph
elif mode == "APP":
    wedge=(360-70,360+110) # simple wedge shape to focus on dark zone. We should define a better mask to have less impact close to star because the performance of the APP is impacted by it. Doesnt like negative numbers.
    transmission=None # APP has same transmission close to the star as off-axis

psf_ON=fits.getdata("psf_sum_scopesim_{0}_{1}.fits".format(mode,band)) # noisy science cube (this time without planet) _sum has a planet in it. 120 frames of 0.3 seconds each spread across 1 hour of field rotation.
psf_OFF=fits.getdata("psf_off_scopesim_{0}_{1}.fits".format(mode,band)) # off-axis science image (unsaturated, stack many to get high SNR of core)

#from vip_hci.var.shapes import get_annular_wedge
#masked=get_annular_wedge(psf_OFF,0,100,wedge=(360-70,360+110),mode="mask") # this shows the simple mask still has a lot of crap close to the baseline of the dark hole (for APP)

#plt.imshow(np.log10(masked))
#plt.show()

psf_OFF_crop, fwhm, ap_flux = psf_template(psf_OFF) # we extract a stamp around the core of the off-axis PSF reference, we also estimate FWHM and aperture flux

print(np.where(psf_OFF[:,:] == np.max(psf_OFF[:,:]))) # double check centroid, 201x201 for L band

nplanes=np.shape(psf_ON)[0]

#calculate position angles (this will come from parang header keyword)
hourangles=np.linspace(-0.5/24.*360.,0.5/24.*360.,nplanes)
lat = -24.59            # deg
dec = -51.0665168055                # deg -51 == beta pic
hr = np.deg2rad(hourangles)
dr = np.deg2rad(dec)
lr = np.deg2rad(lat)
# parallactic angle in deg
pa = -np.rad2deg(np.arctan2(-np.sin(hr), np.cos(dr)*np.tan(lr)
     - np.sin(dr)*np.cos(hr)))
pa = (pa + 360)%360 
pa_rad = np.deg2rad(pa)

#pscale=5.47 # pixel scale in mas in L band
if band in ["HCI_L_long","CO_ref"]:
    rim=403//2 # divide and truncate?
    pscale=5.47
else:
    rim=325//2
    pscale=6.79



algo=vip_hci.psfsub.medsub.median_sub # we exclusively use median subtraction as algorithm


cc_pp = vip_hci.metrics.contrast_curve(psf_ON, pa, psf_OFF_crop, \
                fwhm, pscale/1e3, ap_flux, algo=algo, nbranch=1, sigma=5, \
                debug=True, plot=True, verbose=True,wedge=wedge,nproc=1,transmission=transmission)
plt.show() # plots include software throughput, contrast curves

# stacked cubes
cube_out,cube_der,frame=vip_hci.psfsub.medsub.median_sub(psf_ON,pa,full_output=True,nproc=1)
fits.writeto("cube_out_resid_{0}_{1}.fits".format(mode,band),cube_out,overwrite=1) # medsub full cube
fits.writeto("cube_der_resid_{0}_{1}.fits".format(mode,band),cube_der,overwrite=1) # medsub derotated full cube
fits.writeto("cube_stack_{0}_{1}.fits".format(mode,band),frame,overwrite=1) # medsub derotated median image

# raw contrast profile
sep = pscale*1e-3*np.arange(rim)

off = get_radial_profile(psf_OFF, (rim,rim), 1)[:-1]

psf_ON_mean=np.mean(psf_ON,0)

raw = get_radial_profile(psf_ON_mean-np.median(psf_ON_mean[0:25,:]), (rim,rim), 1)[:-1]
raw=raw/np.max(off)
raw2=off/np.max(off)

plt.semilogy(sep,raw)
plt.semilogy(sep,raw2)
plt.show()


