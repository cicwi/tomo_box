#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 12:41:17 2017

@author: kostenko

Simulate makes fake polychromatic x-ray CT data

"""

#import xraydb

import tomobox
import numpy
import xraylib
import matplotlib.pyplot as plt

class spectra():
    '''
    Simulates spectral phenomena that involve x-ray-matter interaction
    '''
    
    import matplotlib.pyplot as plt
    
    @staticmethod
    def mass_attenuation(energy, compound):
        '''
        Total X-ray absorption for a given compound in cm2g. Energy is given in KeV
        '''
        if numpy.size(energy) == 1:
            return xraylib.CS_Total_CP(compound, energy)   
        else:
            return numpy.array([xraylib.CS_Total_CP(compound, e) for e in energy])

    def linear_attenuation(energy, compound, rho, thickness):
        '''
        Total X-ray absorption for a given compound in cm2g. Energy is given in KeV
        '''
        return thickness * rho * spectra.mass_attenuation(energy, compound)
        
    @staticmethod    
    def compton(energy, compound):    
        '''
        Compton scaterring crossection for a given compound in cm2g. Energy is given in KeV
        '''
        if numpy.size(energy) == 1:
            return xraylib.CS_Compt_CP(compound, energy)   
        else:
            return numpy.array([xraylib.CS_Compt_CP(compound, e) for e in energy])
    
    @staticmethod
    def rayleigh(energy, compound):
        '''
        Compton scaterring crossection for a given compound in cm2g. Energy is given in KeV
        '''
        if numpy.size(energy) == 1:
            return xraylib.CS_Rayl_CP(compound, energy)   
        else:
            return numpy.array([xraylib.CS_Rayl_CP(compound, e) for e in energy])
        
    @staticmethod    
    def photoelectric(energy, compound):    
        '''
        Photoelectric effect for a given compound in cm2g. Energy is given in KeV
        '''
        if numpy.size(energy) == 1:
            return xraylib.CS_Photo_CP(compound, energy)   
        else:
            return numpy.array([xraylib.CS_Photo_CP(compound, e) for e in energy])
        
    @staticmethod          
    def scintillator_efficiency(energy, compound = 'BaFBr', rho = 5, thickness = 100):
        '''
        Generate QDE of a detector (scintillator). Units: KeV, g/cm3, cm.
        '''              
        # Attenuation by the photoelectric effect:
        spectrum = 1 - numpy.exp(- thickness * rho * spectra.photoelectric(energy, compound))
            
        # Normalize:
        return spectrum / spectrum.max()

    @staticmethod 
    def total_transmission(energy, compound, rho, thickness):
        '''
        Compute fraction of x-rays transmitted through the filter. 
        Units: KeV, g/cm3, cm.
        '''        
        # Attenuation by the photoelectric effect:
        return numpy.exp(-spectra.linear_attenuation(energy, compound, rho, thickness))
    
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
    def calibrate_energy_spectrum(projections, volume, energy = numpy.linspace(11,100, 90), compound = 'Al', density = 2.7, force_threshold = None):
        '''
        Use the projection stack of a homogeneous object to estimate system's 
        effective spectrum.
        Can be used by process.equivalent_thickness to produce an equivalent 
        thickness projection stack.
        Please, use conventional geometry. 
        ''' 
        
        sz = projections.data._data.shape
        
        trim_proj = projections.copy()
        trim_vol = volume.copy()
        
        # Get 100 central slices:
        window = 50   
        trim_proj.data._data = trim_proj.data._data[(sz[0]//2-window):(sz[0]//2+window), :, :]        
        trim_vol.data._data = trim_vol.data._data[(sz[0]//2-window):(sz[0]//2+window), :, :]        
        
        trim_vol.display.slice()                                          
                                       
        # Find the shape of the object:                                                    
        trim_vol.postprocess.threshold(threshold = force_threshold)                         
        
        trim_vol.display.slice()                                          
        
        synth_proj = trim_proj.copy()
        
        # Forward project the shape:                  
        print('Calculating the attenuation length.')    
        synth_proj = tomography.project(trim_vol, synth_proj)
                
        # Projected length and intensity (only central slices):
        length = synth_proj.data._data[window//2:-window//2,:,:]
        intensity = trim_proj.data._data[window//2:-window//2,:,:]
        
        plt.figure()
        plt.imshow(intensity[:,1,:])
        plt.colorbar()
        plt.title('intensity')
        
        plt.figure()
        plt.imshow(intensity[:,100,:])
        plt.colorbar()
        plt.title('intensity')
        
        plt.figure()
        plt.imshow(intensity[:,300,:])
        plt.colorbar()
        plt.title('intensity')
        
        plt.figure()
        plt.imshow(length[:,1,:])
        plt.colorbar()
        plt.title('length')
        
        plt.figure()
        plt.imshow(length[:,100,:])
        plt.colorbar()
        plt.title('length')
        
        plt.figure()
        plt.imshow(length[:,300,:])
        plt.colorbar()
        plt.title('length')

        length = length.ravel()
        intensity = intensity.ravel()
        
        print('Number of intensity-length pairs:', length.size)
        
        print('Computing the intensity-length transfer function.')
        
        intensity = numpy.exp(-intensity)
        
        # Bin length (with half a pixel bin size):
        #max_len = length.max()    
        #bin_n = numpy.int(max_len * 2)
        bin_n = 1000
        
        bins = numpy.linspace(0, length.max(), bin_n)
        idx  = numpy.digitize(length, bins)

        # Rebin length and intensity:        
        length_0 = bins - (bins[1]-bins[0]) / 2
        intensity_0 = [numpy.median(intensity[idx==k]) for k in range(bin_n)]

        intensity_0 = numpy.array(intensity_0)
        length_0 = numpy.array(length_0)
        
        # Get rid of nans and more than 1 values:
        length_0 = length_0[intensity_0 < 1]
        intensity_0 = intensity_0[intensity_0 < 1]
        
        # Enforce zero-one values:
        length_0 = numpy.insert(length_0, 0, 0)
        intensity_0 = numpy.insert(intensity_0, 0, 1)
        
        print('Intensity-length curve rebinned.')
        
        # Display:
        plt.figure()
        plt.scatter(length[::100], intensity[::100], color='k', alpha=.2, s=2)
        plt.plot(length_0, intensity_0, 'r--', lw=4, alpha=.8)
        plt.axis('tight')
        plt.title('Intensity v.s. absorption length.')
        plt.show() 
        
        print('Computing the spectrum by Expectation Maximization.')
        
        print('Number of length bins:', intensity_0.size)
        print('Number of energy bins:', energy.size)
        
        nb_iter = 100000
        mu = spectra.linear_attenuation(energy, compound, density, thickness = 0.1)
        exp_matrix = numpy.exp(-numpy.outer(length_0, mu))

        spec = numpy.ones_like(energy)
        
        norm_sum = exp_matrix.sum(0)
        
        for iter in range(nb_iter): 
            spec = spec * exp_matrix.T.dot(intensity_0 / exp_matrix.dot(spec)) / norm_sum

            # Make sure that the total count of spec is 1
            spec = spec / spec.sum()

        print('Spectrum computed.')
             
        plt.figure()
        plt.plot(energy, spec) 
        plt.title('Calculated spectrum.')
            
        i_synthetic = exp_matrix.dot(spec)
        
        # Synthetic spread of Al:
        plt.figure()
        plt.plot(length_0, intensity_0) 
        plt.plot(length_0, i_synthetic) 
        plt.legend(['Measured intensity','Synthetic intensity'])
        
        return energy, spec, length_0, intensity_0
            
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
        
        #dim = numpy.array([sz, sz, sz])
        #space = odl.uniform_discr(min_pt = -dim / 2, max_pt = dim / 2, shape=dim, dtype='float32')

        #x = odl.phantom.transmission.shepp_logan(space)
        
        #vol = tomobox.volume(numpy.transpose(x.asarray(), axes = [2, 0, 1])[:,::-1,:])
        
        vol = tomobox.volume(tomopy.misc.phantom.shepp3d(sz))
        vol.meta.history.add_record('SheppLogan phantom is generated using tomopi shepp_logan()', sz)
        
        return vol
    
    @staticmethod             
    def checkers(sz = 512, frequency = 8):
        
        vol = numpy.zeros([sz, sz, sz], dtype='bool')
        
        step = sz // frequency
        
        #for ii in numpy.linspace(0, sz, frequency):
        for ii in range(0, frequency):
            sl = slice(ii*step, int((ii+0.5) * step))
            vol[sl, :, :] = ~vol[sl, :, :]
        
        for ii in range(0, frequency):
            sl = slice(ii*step, int((ii+0.5) * step))
            vol[:, sl, :] = ~vol[:, sl, :]

        for ii in range(0, frequency):
            sl = slice(ii*step, int((ii+0.5) * step))
            vol[:, :, sl] = ~vol[:, :, sl]
 
        vol = tomobox.volume(numpy.float32(vol))
        vol.meta.history.add_record('Checkers phantom is generated.', sz)
        
        return vol
    
class tomography():
    '''
    Forward projection into the projection data space
    '''        
    @staticmethod
    def project(volume, tomo):
        '''
        Forward projects a volume into a tomogram
        '''
        # In case the simulated volume is bigger than the detector size:
        if ((volume.data.shape[0] - tomo.data.shape[0]) != 0) | ((volume.data.shape[2] - tomo.data.shape[2]) != 0):
            tomo.reconstruct._initialize_astra(force_vol_size = volume.data.shape)
        
        tomo.data._data = tomo.reconstruct._forwardproject(numpy.ascontiguousarray(volume.data._data))
        tomo.meta.history.add_record('simulate.tomography.project was used to generate the data')
        
        return tomo
        
   
