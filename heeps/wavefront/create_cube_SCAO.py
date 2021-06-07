from heeps.util.img_processing import crop_img, pad_img, resize_img
from heeps.util.multiCPU import multiCPU
from heeps.util.notify import notify
import os
import numpy as np
from astropy.io import fits

# email when finished
conf = {}
conf['send_to'] = 'cdelacroix@uliege.be'
conf['send_message'] = 'cube calculation finished OK.'

# useful inputs
tag = 'Cbasic_20201130'
prefix = 'Residual_phase_screen_'#'tarPhase_1hr_100ms_'
suffix = 'ms'
duration = 3600
samp = 300
start = 2101#0#1001
nimg = 720
npupil = 285
pad_frame = False
savename = 'cube_%s_%ss_%sms_0piston_meters_scao_only_%s.fits'%(tag, duration, samp, npupil)

#input_folder = '/mnt/disk4tb/METIS/METIS_COMPASS_RAW_PRODUCTS/gorban_metis_baseline_Cbasic_2020-10-16T10:25:14/residualPhaseScreens'
#input_folder = '/mnt/disk4tb/METIS/METIS_COMPASS_RAW_PRODUCTS/gorban_metis_baseline_Cbasic_2020-11-05T12:40:27/residualPhaseScreens'
input_folder = '/mnt/disk4tb/METIS/METIS_COMPASS_RAW_PRODUCTS/gorban_metis_baseline_Cbasic_2020-11-30T20:52:24/residualPhaseScreens'
output_folder = 'METIS_CBASIC_CUBES'
cpu_count = None

# mask
mask = fits.getdata(os.path.join(input_folder, 'Telescope_Pupil.fits'))
mask = crop_img(mask, nimg)
mask_pupil = np.rot90(resize_img(mask, npupil))
fits.writeto(os.path.join(output_folder, 'mask_%s_%s.fits'%(tag, npupil)), np.float32(mask_pupil), overwrite=True)

# filenames
nframes = len([name for name in os.listdir(input_folder) if name.startswith(prefix)])
frames = [str(frame).zfill(6) if pad_frame is True else str(frame) \
    for frame in range(start, start + nframes*samp, samp)]
filenames = np.array([os.path.join(input_folder, '%s%s%s.fits'%(prefix, frame, suffix)) \
    for frame in frames])

def remove_piston(filename):
    data = np.float32(fits.getdata(filename))
    data = crop_img(data, nimg)
    data -= np.mean(data[mask!=0])      # remove piston
    data[mask==0] = 0
    data = resize_img(data, npupil)
    data = np.rot90(data) * 1e-6        # rotate, convert to meters
    return data

# create cube
cube = multiCPU(remove_piston, posvars=[filenames], case='create cube SCAO', cpu_count=cpu_count)
print(cube.shape)
fits.writeto(os.path.join(output_folder, savename), np.float32(cube), overwrite=True)
notify(conf['send_message'], conf['send_to'])