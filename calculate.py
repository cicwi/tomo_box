#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 14:03:25 2017

@author: kostenko

Contains static methods for computating stuff.

"""
import numpy
    
def frft2_centered(x, alpha):
    '''
    Fractional Fourier Transform, computing the transform
              
            y[n]=sum_{k=0}^{N-1} x(k)*exp(-i*2*pi*k*n*alpha)  n=-N/2, ... ,N/2-1
 
    For alpha=1/N we get the regular FFT, and for alpha=-1/N we get the regular IFFT.
    Written by Michael Elad on March 20th 2005.
    '''
    x = x.ravel()
    n = x.size
    
    F2 = numpy.exp(1j*numpy.pi * numpy.arange(0, n) * n * alpha)
    x_tilde = x * F2
    
    nn = numpy.arange(-n, n, dtype = 'complex64')
    nn = numpy.fft.fftshift(nn)
    
    F = numpy.exp(-1j * numpy.pi * alpha * nn**2)
    
    x_tilde = numpy.concatenate((x_tilde, numpy.zeros(n, dtype = 'complex64')))
    x_tilde = x_tilde * F    
    
    
    
    xx = numpy.fft.fft(x_tilde)
    yy = numpy.fft.fft(numpy.conj(F))
    
    y = numpy.fft.ifft(xx * yy)
    y = y * F
    y = y[0:n]
    
    return y
    
def frft2(x, alpha):
    '''
    Fractional Fourier Transform, computing the transform
                  
                y[n]=sum_{k=0}^{N-1} x(k)*exp(-i*2*pi*k*n*alpha)  n=0,1,2, ... ,N-1
    
    So that for alpha=1/N we get the regular FFT, and for alpha=-1/N we get the regular
    IFFT.
    
    '''
    x = x.ravel()
    n = x.size
    
    nn = numpy.arange(-n, n, dtype = 'complex64')
    nn = numpy.fft.fftshift(nn)
    
    F = numpy.exp(-1j * numpy.pi * alpha * nn**2)
          
    x_tilde = numpy.concatenate((x, numpy.zeros(n, dtype = 'complex64')))
    x_tilde = x_tilde * F  
    
    xx = numpy.fft.fft(x_tilde)
    yy = numpy.fft.fft(numpy.conj(F))
    
    y = numpy.fft.ifft(xx * yy)
    y = y * F
    y = y[0:n]
    
    return y


def ppft2(im):
    '''
    Adopted from the matlab code by Michael Elad.
    
    The function computes the 2-D pseudo-polar Fourier transform (PFFT)
    '''
    # im should be n * n
    # Make sure we are compatible with complex numbers
    im = numpy.complex64(im)
    
    sz = im.shape
    
    # Make sure the image is square and the dimensions are even:
    n = int(numpy.ceil(max(sz)/2)*2)
    
    im_ = numpy.zeros([n,n], dtype = 'complex64')
    im_[n//2-(sz[0]//2):n//2-sz[0]//2+sz[0], n//2-sz[1]//2:n//2-sz[1]//2+sz[1]] = im
    im = im_
    
    # Output array:
    y = numpy.zeros([2*n,2*n], dtype = 'complex64')
    
    # Quadrant 1 and 3:
    f_tilde = numpy.fft.fft(numpy.concatenate((im, numpy.zeros([n, n], dtype = 'complex64')), 0),axis= 0)          
    f_tilde = numpy.fft.fftshift(f_tilde, 0)    
    
    for ll in range(-n, n):
        y[ll+n, n-1::-1] = frft2_centered(f_tilde[ll + n, :], ll/(n**2)).T
    
    # Quadrant 2 and 4
    f_tilde = numpy.fft.fft(numpy.concatenate((im, numpy.zeros([n, n], dtype = 'complex64')), 1), axis=1)
    f_tilde=numpy.fft.fftshift(f_tilde,1)
    f_tilde=numpy.conj(f_tilde.T)
    
    for ll in range(-n, n):
        
         F = numpy.exp(1j * 2 * numpy.pi*numpy.arange(0, n) * (n/2-1)*ll/(n**2))        
         y[ll+n, -1:n-1:-1] = frft2(f_tilde[ll+n,:] * F, ll/(n**2)).T
    
    return y  