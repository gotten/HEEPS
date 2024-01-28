import numpy as np
import scopesim as sim
from astropy.io import fits
import matplotlib.pyplot as plt
from synphot import SourceSpectrum
import os
import astropy.modeling
psf_ON=np.zeros([800,800])

xx,yy=np.meshgrid(np.arange(800)-400,np.arange(800)-400)

func=astropy.modeling.functional_models.AiryDisk2D(1,0,0,1.22*21/5.2) # airy disk with location of the first zero at 1.22 * lambda/D in mas / pixel scale in mas

#plt.imshow(func(xx,yy))
#plt.show()

psf_ON=func(xx,yy) 
psf_ON=psf_ON/np.sum(psf_ON)*10**(-0.4*3.5) # normalize to 1 and make flux equal to a L=3.5 star


#psf_ON=psf_ON+1000

hdu=fits.PrimaryHDU(data=psf_ON)
hdu.header.append("EXPTIME")
hdu.header.append("CDELT1")
hdu.header.append("CDELT2")
hdu.header.append("CTYPE1")
hdu.header.append("CTYPE2")
hdu.header.append("CUNIT1")
hdu.header.append("CUNIT2")
hdu.header.append("CRPIX1")
hdu.header.append("CRPIX2")
hdu.header.append("CRVAL1")
hdu.header.append("CRVAL2")
hdu.header.append("BUNIT")
#hdu.header.append("OFFAXISTRANS")
#hdu.header.append("MODE")
hdu.header.append("BAND")
hdu.header.append("LAM")
hdu.header.append("FILTER")

hdu.header['CDELT1']=5.47/1000*0.000277778 #conf['band_specs'][hdu.header['BAND']]['pscale']*0.000277778/1000.
hdu.header['CDELT2']=5.47/1000*0.000277778 #conf['band_specs'][hdu.header['BAND']]['pscale']*0.000277778/1000. # mas to degrees
#print(conf)
hdu.header['EXPTIME']=0.04 #dit#conf['dit']
hdu.header["CTYPE1"]  = 'RA---TAN'
hdu.header["CTYPE2"]  = 'DEC--TAN'
hdu.header['CUNIT1']='deg'
hdu.header['CUNIT2']='deg'

hdu.header['BUNIT']="photons/s" # BUNIT overridden by use of Vega spectrum in scopesim
hdu.header["CRPIX1"]=400
hdu.header["CRPIX2"]=400
hdu.header["CRVAL1"]=0
hdu.header["CRVAL2"]=0

WORKDIR = "/home/gpplotten/projects/scopesim/irdb/" # this folder contains a vega spectrum
PKGSDIR = "/home/gpplotten/projects/scopesim/irdb/inst_pkgs/" # this folder contains the IRDB packages

PKGS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(PKGSDIR)))
sim.rc.__config__['!SIM.file.local_packages_path'] = PKGS_DIR

cmd=sim.UserCommands(use_instrument="METIS", set_modes=["img_n"],properties={"!OBS.filter_name": "N1","!OBS.auto_exposure.fill_frac":1e6,"!OBS.exptime":hdu.header['EXPTIME'],"!OBS.ndit":1,"!OBS.dit": hdu.header['EXPTIME'],"!OBS.detector_readout_mode":"high_capacity","!DET.width":800,"!DET.height":800})
#cmd=sim.UserCommands(use_instrument="METIS", set_modes=["img_lm"],properties={"!OBS.filter_name": "HCI_L_long","!OBS.auto_exposure.fill_frac":1e6,"!OBS.exptime":hdu.header['EXPTIME'],"!OBS.ndit":1,"!OBS.dit": hdu.header['EXPTIME'],"!OBS.detector_readout_mode":"fast","!DET.width":800,"!DET.height":800})



metis_on = sim.OpticalTrain(cmd)
metis_on['psf'].include=False
#metis_on['skycalc_atmosphere'].meta["use_local_skycalc_file"]=False
#metis_on['skycalc_atmosphere'].include = False
 
spec = SourceSpectrum.from_file(WORKDIR+"alpha_lyr_stis_008.fits") # wget https://ssb.stsci.edu/cdbs/calspec/alpha_lyr_stis_008.fits
src_on = sim.Source(spectra=[spec], image_hdu=hdu) # L=0 star with Vega spectrum

metis_on.observe(src_on,update=True)
outhdu = metis_on.readout()[0]
img=outhdu[1].data
plt.imshow(img)
plt.show()
