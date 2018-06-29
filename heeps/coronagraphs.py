#!/usr/bin/env python3


# Required libraries
#import proper # library for propagation of wavefront
import numpy as np
from astropy.io import fits    # to read and write fits files

from heeps import apodization, vortex,lyotstop # loads all HEEPS scripts required for simuation



def coronagraphs(wfo, r_obstr,npupil, phase_apodizer_file,amplitude_apodizer_file,apodizer_misalignment,charge,f_lens,diam,LS_amplitude_apodizer_file,LS_misalignment,LS,LS_parameters,spiders_angle, LS_phase_apodizer_file, Debug_print,pixelsize, Debug, coronagraph_type='None'):
    
    if coronagraph_type == 'RAVC':
        phase_apodizer_file = 0
        RAVC = True
        apodization(wfo, r_obstr, npupil, RAVC=True, phase_apodizer_file=phase_apodizer_file, amplitude_apodizer_file=amplitude_apodizer_file, apodizer_misalignment=apodizer_misalignment, Debug_print=Debug_print)
        vortex(wfo, charge, f_lens,diam, pixelsize, Debug_print = Debug_print)
        lyotstop(wfo, diam, r_obstr, npupil, RAVC, LS, LS_parameters, spiders_angle, LS_phase_apodizer_file, 
         LS_amplitude_apodizer_file, LS_misalignment, Debug_print, Debug)

    elif coronagraph_type == "VC":
        phase_apodizer_file = 0
        RAVC = False
        vortex(wfo, charge, f_lens,diam, pixelsize, Debug_print = Debug_print)
        lyotstop(wfo, diam, r_obstr, npupil, RAVC, LS, LS_parameters, spiders_angle, LS_phase_apodizer_file, 
         LS_amplitude_apodizer_file, LS_misalignment, Debug_print, Debug)

    elif coronagraph_type == 'APP':
        RAVC = False
        apodization(wfo, r_obstr, npupil, RAVC=False, phase_apodizer_file=phase_apodizer_file, amplitude_apodizer_file=amplitude_apodizer_file, apodizer_misalignment=apodizer_misalignment, Debug_print=Debug_print)
        lyotstop(wfo, diam, r_obstr, npupil, RAVC, LS, LS_parameters, spiders_angle, LS_phase_apodizer_file, 
         LS_amplitude_apodizer_file, LS_misalignment, Debug_print, Debug)

    else:
        print('No Coronagraph')    
        RAVC = False
        lyotstop(wfo, diam, r_obstr, npupil, RAVC, LS, LS_parameters, spiders_angle, LS_phase_apodizer_file, 
         LS_amplitude_apodizer_file, LS_misalignment, Debug_print, Debug)

    return wfo


