#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 12:41:17 2017

@author: kostenko

Simulate makes fake polychromatic x-ray CT data

"""

#import tomopy.misc
#import tomo_wrap as tw
#import xraydb

class spectra():
    '''
    Simulates spectral phenomena that involve x-ray-matter interaction
    '''
    import xraylib
    
    @staticmethod
    def total_attenuation(energy, compound):
        '''
        Total X-ray absorption for a given compound in cm2g. Energy is given in KeV
        '''
        return xraylib.CS_Total_CP(compound, energy)   

    @staticmethod    
    def compton(energy, compound):    
        '''
        Compton scaterring crossection for a given compound in cm2g. Energy is given in KeV
        '''
        return xraylib.CS_Compt_CP(compound, energy)   
    
    @staticmethod
    def rayleigh(energy, compound):
        '''
        Compton scaterring crossection for a given compound in cm2g. Energy is given in KeV
        '''
        return xraylib.CS_Rayl_CP(compound, energy)   
        
    @staticmethod    
    def photoelectric(energy, compound):    
        '''
        Photoelectric effect for a given compound in cm2g. Energy is given in KeV
        '''
        return xraylib.CS_Photo_CP(compound, energy)   
        
    @staticmethod    
    def bremsstrahlung(energy, energy_max):
        '''
        Simple bremstrahlung model (Kramer formula). Emax
        '''
        spectrum = energy * (energy_max - energy)
        spectrum[spectrum < 0] = 0
            
        # Normalize:
        return spectrum / spectrum.max()
    
    @staticmethod
    def gaussian_spectrum(energy, energy_mean, energy_sigma):
        '''
        Generates gaussian-like spectrum with given mean and STD.
        '''
        return numpy.exp(-(energy - energy_mean)**2 / (2*energy_sigma**2))
        
    @staticmethod          
    def scintillator_efficiency(energy, compound = 'BaFBr', rho = 5, thickness = 100):
        '''
        Generate QDE of a detector (scintillator). Units: KeV, g/cm3, micron.
        '''       
        # Thickness to cm:
        thickness /= 1e4 
        
        # Attenuation by the photoelectric effect:
        spectrum = 1 - numpy.exp(- thickness * rho * spectra.photoelectric(energy, compound))
            
        # Normalize:
        return spectrum / spectrum.max()

    @staticmethod 
    def transmission(energy, compound, rho, thickness):
        '''
        Compute fraction of x-rays transmitted through the filter. 
        Units: KeV, g/cm3, micron.
        '''
        # Thickness to cm:
        thickness /= 1e4 
        
        # Attenuation by the photoelectric effect:
        return 1 - numpy.exp(- thickness * rho * spectra.total_attenuation(energy, compound))
    
    @staticmethod 
    def attenuation(energy, compound, rho, thickness):
        '''
        Compute fraction of x-rays attenuated by the filter
        '''
        # Thickness microns to cm:
        thickness /= 1e4 
        
        return numpy.exp(- thickness * rho * spectra.total_attenuation(energy, compound))
        
class nist():
    @staticmethod 
    def list_names():
        return xraylib.GetCompoundDataNISTList()
        
    @staticmethod     
    def find_name(compound_name):    
        return xraylib.GetCompoundDataNISTByName(compound_name)
    
    @staticmethod     
    def parse_compound(compund):
        return xraylib.CompoundParser(compund)
        
class phantom():    
    ''' 
    Use tomopy phantom module for now
    '''
    
    @staticmethod     
    def shepp3d(sz = 512):
        import tomopy.misc
        import tomo_wrap
        
        tomo_wrap.volume(tomopy.misc.phantom.shepp3d(sz))
    
class tomography():
    '''
    Forward projection into the projection data space
    '''
    
    @staticmethod
    def project(volume, sino):
        '''
        Forward projects a volume into a sinogram
        '''
        sino.reconstruct._initialize_astra()
                
        sino.meta.history['N.B.'] = 'faker.tomography.project is used to generate the data'
        
        sino.data.set_data(sino.reconstruct._forwardproject(volume.data._data))
        
        return sino
        
    '''
class faker():
    '''
    '''
    phantom = []
    spectra = []
    tomography = []

    
    def __init__(self):
        self.phantom = tomopy.misc.phantom
        self.spectra = spectra()
        self.tomography = tomography()
    '''

