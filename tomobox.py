#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:39:33 2017

@author: kostenko & der sarkissian

 ***********    Pilot for the new tomobox  *************

"""


#%% Initialization

import matplotlib.pyplot as plt
from scipy import misc  # Reading BMPs
import os
import numpy
import re
import time
import transforms3d
from transforms3d import euler 
    
#import pkg_resources
#pkg_resources.require("dxchange==0.1.2")
#from mayavi import mlab
#from tvtk.util.ctf import ColorTransferFunction

# **************************************************************
#           Parent class for all sinogram subclasses:
# **************************************************************

class subclass(object):
    def __init__(self, parent):
        self._parent = parent

# **************************************************************
#           Geometry class
# **************************************************************
class history():
    '''
    Container for the history reconrds of operations applied to the data.
    '''
    
    _records = []
    
    @property
    def records(self):
        return self._records.copy()
        
    def __init__(self):
        
        self._records = []
        self.add_record('Created')
    
    def add_record(self, operation = '', properties = None):
        
        # Add a new history record:
        timestamp = time.ctime()
        
        # Add new record:
        self._records.append([operation, properties, timestamp, numpy.shape(self._records)[0] ])
    
    @property
    def keys(self):
        '''
        Return the list of operations.
        '''
        return [ii[0] for ii in self._records]
        
    def find_record(self, operation):
        # Find the last record by its operation name:
            
        result = None
        
        result = [ii for ii in self._records if operation == ii[0]]

        if numpy.size(result) > 0:
            return result[-1]    
        else:
            return None
    
    def time_last(self):
        # Time between two latest records:
        if self._records.size() > 0:
            return self._records[-1][2] - self._records[-2][2]
            
    def _delete_after(self, operation):
        # Delete the history after the last backup record:
            
        record = self.find_record(self, operation)
        
        # Delete all records after the one that was found:
        if not record is None:
            self._records = self._records[0:record[3]]
    
# **************************************************************
#           Geometry class
# **************************************************************
class geometry(subclass):
    '''
    For now, this class describes circular motion cone beam geometry.
    It contains standard global parameters such as source to detector distance and magnification
    but also allows to add modifiers to many degrees of freedom for each projection separately.    
    '''    
    
    # Private properties:
    _src2obj = 0
    _det2obj = 0
    _det_pixel = [1, 1]
    _thetas = [0, numpy.pi * 2]

    # Additional misc public:
    roi_fov = False
    
    '''
    Modifiers (dictionary of geometry modifiers that can be applied globaly or per projection)
    VRT, HRZ and MAG are vertical, horizontal and prependicular directions relative to the original detector orientation     
    '''
        
    def __init__(self, parent):
        subclass.__init__(self, parent)
        
        self.modifiers = {'det_vrt': 0, 'det_hrz': 0, 'det_mag': 0, 'src_vrt': 0, 'src_hrz': 0, 'src_mag': 0, 'det_rot': 0, 'dtheta':0, 'vol_x_tra': 0, 'vol_y_tra':0, 'vol_z_tra':0, 'vol_x_rot':0, 'vol_y_rot':0, 'vol_z_rot':0}    

    def initialize(self, src2obj, det2obj, det_pixel, theta_range, theta_n):
        '''
        Make sure that all relevant properties are set to some value.
        '''
        self._src2obj = src2obj
        self._det2obj = det2obj
        self._det_pixel = det_pixel
        self.init_thetas(theta_range, theta_n)
        
    def modifiers_reset(self):
        for key in self.modifiers.keys():
            self.modifiers[key] = 0
    
    def find_modifier(self, key, index = None):
        '''
        Get a geometry modifier for a prjection with index = index. Or take the first modifier that corresponds to the key.
        '''
        
        if (numpy.size(self.modifiers[key]) == 1) or (index is None):
            return self.modifiers[key]

        elif numpy.size(self.modifiers[key]) > index:
            return self.modifiers[key][index]

        else: print('Geometry modifier not found!')
        #else: self._parent.error('Geometry modifier not found!')
        
        return None

    def translate_volume(self, vector):
        self.modifiers['vol_x_tra'] += vector[0]
        self.modifiers['vol_y_tra'] += vector[1]
        self.modifiers['vol_z_tra'] += vector[2]

    def rotate_volume(self, vector):
        self.modifiers['vol_x_rot'] += vector[0]
        self.modifiers['vol_y_rot'] += vector[1]
        self.modifiers['vol_z_rot'] += vector[2]

    def thermal_shift(self, thermal_shifts, additive = False):
        '''
        Shift the source according to the thermal shift data
        '''
        if additive:
            self.modifiers['src_hrz'] += thermal_shifts[:,0]/(self.magnification - 1) 
            self.modifiers['src_vrt'] += thermal_shifts[:,1]/(self.magnification - 1) 
        else:
            self.modifiers['src_hrz'] = thermal_shifts[:,0]/(self.magnification - 1) 
            self.modifiers['src_vrt'] = thermal_shifts[:,1]/(self.magnification - 1) 

    def rotation_axis_shift(self, shift, additive = False):
        if additive:
            self.modifiers['det_hrz'] += -shift / self.magnification
            self.modifiers['src_hrz'] += -shift / self.magnification
        else:
            self.modifiers['det_hrz'] = -shift / self.magnification
            self.modifiers['src_hrz'] = -shift / self.magnification          

    def optical_axis_shift(self, shift, additive = False):
        if additive:
            self.modifiers['det_vrt'] += shift
        else:
            self.modifiers['det_vrt'] = shift

        # Center the volume around the new axis:
        self.translate_volume([0, 0, shift])
                
    def origin_shift(self):
        '''
        Compute the shift of the volume central point [x, y, z] due to offsets in the optical and rotation axes.
        '''    
        hrz = (self.modifiers['det_hrz'] * self.src2obj + self.modifiers['src_hrz'] * self.det2obj) / self.src2det
        vrt = (self.modifiers['det_vrt'] * self.src2obj + self.modifiers['src_vrt'] * self.det2obj) / self.src2det
        
        # Take into account global shifts:
        hrz = numpy.max([numpy.abs(hrz + self.modifiers['vol_x_tra']), hrz + numpy.abs(self.modifiers['vol_y_tra'])])
        vrt += self.modifiers['vol_z_tra']
        
        return [hrz, vrt]
    
    # Set/Get methods (very bodring part of code but, hopefully, it will make geometry look prettier from outside):       
        
    @property
    def src2obj(self):
        return self._src2obj
        
    @src2obj.setter
    def src2obj(self, src2obj):
        self._src2obj = src2obj
        
    @property
    def det2obj(self):
        return self._det2obj
        
    @det2obj.setter
    def det2obj(self, det2obj):
        self._det2obj = det2obj
        
    @property
    def magnification(self):
        return (self._det2obj + self._src2obj) / self._src2obj
        
    @property
    def src2det(self):
        return self._src2obj + self._det2obj
        
    @property
    def det_pixel(self):
        return self._det_pixel
        
    @det_pixel.setter
    def det_pixel(self, det_pixel):
        self._det_pixel = det_pixel
        
    @property
    def img_pixel(self):
        return [self._det_pixel[0] / self.magnification, self._det_pixel[1] / self.magnification]   
        
    @img_pixel.setter
    def img_pixel(self, img_pixel):
        self._det_pixel = [img_pixel[0] * self.magnification, img_pixel[1] * self.magnification]         
        
    @property
    def det_size(self):
        
        # We wont take into account the det_size from the log file. Only use actual data size.
        if self._parent.data is None:
            self._parent.warning('No raw data in the pipeline. The detector size is not known.')
        else:
            return self._det_pixel * self._parent.data.shape[::2]
        
    @property
    def thetas(self):
        dt = 1#self._parent._data_sampling
        
        if self._parent.data.shape[1] != numpy.size(self._thetas[::dt]):
            self._parent.warning('Length of thetas array is not consistent with the data shape! Will try to initialize thetas using the data shape.')
            
            self.init_thetas(theta_n = self._parent.data.shape[1])
            
        return numpy.array(self._thetas[::dt])
        
    @thetas.setter
    def thetas(self, thetas):
        dt = 1 #self._parent._data_sampling
        
        self._thetas[::dt] = numpy.array(thetas)
    
    @property
    def theta_n(self):
        return self.thetas.size    
        
    @property
    def theta_range(self):
        return (self.thetas[0], self.thetas[-1])
     
    @theta_range.setter    
    def theta_range(self, theta_range):
        # Change the theta range:
        len = numpy.size(self.thetas)     
        
        if len > 2:
            self.thetas = numpy.linspace(theta_range[0], theta_range[1], len)
        else:
            self.thetas = [theta_range[0], theta_range[1]]
        
    @property        
    def theta_step(self):
        return numpy.mean(self._thetas[1:] - self._thetas[0:-1])  
        
    def init_thetas(self, theta_range = [], theta_n = 2):
        # Initialize thetas array. You can first initialize with theta_range, and add theta_n later.
        if theta_range == []:
            self._thetas = numpy.linspace(self._thetas[0], self._thetas[-1], theta_n)
        else:    
            self._thetas = numpy.linspace(theta_range[0], theta_range[1], theta_n)
        
# **************************************************************
#           META class and subclasses
# **************************************************************

class meta(subclass):
    '''
    This object contains various properties of the imaging system and the history of pre-processing.
    '''
    geometry = None
    history = history()

    def __init__(self, parent):
        subclass.__init__(self, parent)
        self.geometry = geometry(self._parent)
        
    physics = {'voltage': 0, 'current':0, 'exposure': 0}
    lyrics = ''
    
        
# **************************************************************
#           DATA class
# **************************************************************

import sys

class data(subclass):
    '''
    Memory allocation, reading and writing the data. This version only supports data stored in CPU memory.
    '''
    # Raw data, flat field (reference), dark field and backup
    _data = None
    _ref = None
    _dark = None
    _backup = None

    # Keep a second copy of the data each time the data is modified?    
    _backup_update = False
    
    # Get/Set methods:
    @property
    def data(self):
        dx = self._parent._data_sampling[1]
        dz = self._parent._data_sampling[0]
        dt = 1#self._parent._data_sampling
        
        if dx + dz + dt > 3:
            return numpy.ascontiguousarray(self._data[::dz, ::dt, ::dx])
        else:
            return self._data
        
    @data.setter
    def data(self, data):
        dx = self._parent._data_sampling[1]
        dz = self._parent._data_sampling[0]
        dt = 1#self._parent._data_sampling
        
        if self._backup_update:
            self._parent.io.save_backup()
            
        self._data[::dz, ::dt, ::dx] = data
        
        self._parent.meta.history.add_record('set data.data', [])
        
    def get_ref(self, proj_num = 0):
        '''
        Returns a reference image. Interpolated if sefveral reference images are available.
        '''
        
        # Return reference image for the current projection:
        if self._ref.ndim > 2:
           
            if self._data is None:
                self._parent.warning('No raw data available. We don`t know how many projections there are in order to interpolate the reference image properly. Read raw data first.')
                dsz = self._ref.shape[1]

            else:
                dsz = self.data.shape[1]
                
            # Several flat field images are available:
            ref = self._ref
           
            sz = ref.shape

            # This implementation is too slow:
            #proj_index = numpy.linspace(0, dsz-1, sz[1])
            #interp_grid  = numpy.array(numpy.meshgrid(numpy.arange(sz[0]), proj_num, numpy.arange(sz[2])))
            #interp_grid = numpy.transpose(interp_grid, (2,1,3,0))
            #original_grid = (numpy.arange(sz[0]), proj_index, numpy.arange(sz[2]))
            #return interp_sc.interpn(original_grid, ref, interp_grid) 
            
            proj_index = numpy.linspace(0, sz[1]-1, dsz)
            
            a = proj_index[proj_num]
            fract = a - numpy.floor(a)            
            a = int(numpy.floor(a))            
            
            if a < (dsz-1): 
                b = int(numpy.ceil(proj_index[proj_num]))
            else: 
                b = a
                
            return self._ref[:, a, :] * (1 - fract) + self._ref[:, b, :] * fract
           
        else:
            
           # One flat field image is available: 
           return self._ref  
           
    def float32_to_uint8(self):       
        mn = self._data.min()   
        mx = self._data.max()   
    
        self._data = numpy.uint8((self._data - mn) / (mx - mn) * 255)
        
        return mn, mx
        
    def float32_to_uint16(self):       
        mn = self._data.min()   
        mx = self._data.max()   
    
        self._data = numpy.uint16((self._data - mn) / (mx - mn) * 65535)
        
        return mn, mx
        
    def float32_to_float16(self):       
        self._data = numpy.float16(self._data)
        
        return self._data.min(), self._data.max()    
        
    @property        
    def shape(self):
        dx = self._parent._data_sampling[1]
        dz = self._parent._data_sampling[0]
        dt = 1#self._parent._data_sampling
        
        return numpy.array(self._data[::dz, ::dt, ::dx].shape, dtype = 'int')
        
    @property            
    def size_mb(self):
        '''
        Get the size of the data object in MB.
        '''
        return sys.getsizeof(self)    
        
    # Public methods:    
    def data_at_theta(self, target_theta):
        '''
        Use interpolation to get a projection at a given theta
        '''
        thetas = self._parent.meta.geometry.thetas

        if (target_theta > thetas.max()) |(target_theta < thetas.min()):
            self._parent.error('Theta is out of range!')
            
        # Interpolation is too slow:
        #interp_grid = numpy.transpose(numpy.meshgrid(target_theta, numpy.arange(sz[0]), numpy.arange(sz[2])), (1,2,3,0))
        #original_grid = (numpy.arange(sz[0]), thetas, numpy.arange(sz[2]))
        #return interp_sc.interpn(original_grid, self.data, interp_grid)
        
        a = numpy.sum(thetas < target_theta)
        
        fract = target_theta - thetas[a]
        if a < (thetas.size -1):
            b = a + 1
        else:
            b = a
            
        return self.data[:, a, :] * (1 - fract) + self.data[:, b, :] * fract
        

# **************************************************************
#           IO class and subroutines
# **************************************************************
from stat import ST_CTIME
import gc
import csv

def sort_by_date(files):
    '''
    Sort file entries by date
    '''
    # get all entries in the directory w/ stats
    entries = [(os.stat(path)[ST_CTIME], path) for path in files]

    return [path for date, path in sorted(entries)]

def sort_natural(files):
    '''
    Sort file entries using the natural (human) sorting
    '''
    # Keys
    keys = [int(re.findall('\d+', f)[-1]) for f in files]

    # Sort files using keys:
    files = [f for (k, f) in sorted(zip(keys, files))]

    return files

def read_image_stack(file):
    '''
    Read a stack of some image files
    '''
    # Remove the extention and the last few letters:
    name = os.path.basename(file)
    ext = os.path.splitext(name)[1]
    name = os.path.splitext(name)[0]
    digits = len(re.findall('\d+$', name)[0])
    name_nonb = re.sub('\d+$', '', name)
    path = os.path.dirname(file)
    
    # Get the files of the same extension that finish by the same amount of numbers:
    files = os.listdir(path)
    files = [x for x in files if (re.findall('\d+$', os.path.splitext(x)[0]) and len(re.findall('\d+$', os.path.splitext(x)[0])[0]) == digits)]

    # Get the files that are alike and sort:
    files = [os.path.join(path,x) for x in files if ((name_nonb in x) and (os.path.splitext(x)[-1] == ext))]
    
    #files = sorted(files)
    files = sort_natural(files)

    #print(files)

    # Read the first file:
    image = misc.imread(files[0], flatten= 0)
    sz = numpy.shape(image)
    
    data = numpy.zeros((len(files), sz[0], sz[1]), dtype = numpy.float32)

    # Read all files:
    ii = 0
    for filename in files:
        a = misc.imread(filename, flatten= 0)
        if a.ndim > 2:
          a = a.mean(2)
        data[ii, :, :] = a
        ii = ii + 1

    print(ii, 'files were loaded.')

    return data

def extract_2d_array(dimension, index, data):
    '''
    Extract a 2d array from 3d.
    '''
    if dimension == 0:
        return data[index, :, :]
    elif dimension == 1:
        return data[:, index, :]
    else:
        return data[:, :, index]
              
class io(subclass):
    '''
    Static class for loading / saving the data
    '''

    path = ''

    #settings = {'sort_by_date':False}
    
    # Private helper routines:
    def _make_path(self, path):
        '''
        Make path if it doesn't exist.
        '''    
        if not os.path.exists(path):
            os.makedirs(path)
    
    def _update_path(self, path):
        '''
        Memorize the path if it is provided, otherwise use the one that remember.
        '''
        if path == '':
            path = self.path
        else:
            self.path = path
    
        if path == '':
            self._parent.error('Path to the file was not specified.')
    
        return path
    
    # Public routoines:    
    def manual_init(self, src2obj = 100, det2obj = 100, theta_n = 128, theta_range = [0, 2*numpy.pi], det_width = 128, det_height = 128, det_pixel = [0.1, 0.1]):
        '''
        Manual initialization can be used when log file with methadata can not be read or
        if a sinthetic data needs to be created.
        '''
        prnt = self._parent
        # Initialize the geometry data:
        prnt.meta.geometry.initialize(src2obj, det2obj, det_pixel, theta_range, theta_n) 
        
        # Make an empty projections:        
        prnt.data._data = (numpy.zeros([det_height, theta_n, det_width]))
        prnt.meta.history.add_record('io.manual_init')

    def read_raw(self, path = '', filter = '', projections = True, index_range = [], y_range = [], x_range = []):   
        '''
        Read projection files automatically.
        This will look for files with numbers in the last 4 letters of their names.
        '''
        path = self._update_path(path)

        # Free memory:
        self._parent.data._data = None
        gc.collect()

        # if it's a file, read all alike, if a directory find a file to start from:
        if os.path.isfile(path):
            filename = os.path.basename(path)
            path = os.path.dirname(path)

            # if file name is provided, the range is needed:
            if index_range != []:    
                first = index_range[0]
                last = index_range[1]

        else:
            # Try to find how many files do we need to read:

            # Get the files only:
            files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]

            # Get the last 4 letters:
            index = [os.path.splitext(x)[0][-4:] for x in files]

            # Filter out non-numbers:
            index = [int(re.findall('\d+', x)[0]) for x in index if re.findall('\d+', x)]

            # Extract a number from the first element of the list:
            first = min(index)

            # Extract a number from the first element of the list:
            last = max(index)

            print('We found projections in the range from ', first, 'to ', last, flush=True)

            # Find the file with the maximum index value:
            filename = [x for x in os.listdir(path) if str(last) in x][0]

            # Find the file with the minimum index:
            filename = sorted([x for x in os.listdir(path) if (filename[:-8] in x)&(filename[-3:] in x)])[0]

            print('Reading a stack of images')
            print('Seed file name is:', filename)
            #srt = self.settings['sort_by_date']AMT24-25-SU1/

        if self._parent:
            self._parent.data._data = (read_image_stack(os.path.join(path,filename)))
        else:
            return read_image_stack(os.path.join(path,filename))

        
        # Trim the data with the provided inputs
        if (index_range != []):
            print(index_range)
            self._parent.data._data = self._parent.data._data[index_range[0]:index_range[1], :, :]
        if (y_range != []):
            self._parent.data._data = self._parent.data._data[:, y_range[0]:y_range[1], :]
        if (x_range != []):
            self._parent.data._data = self._parent.data._data[:, :, x_range[0]:x_range[1]]

        # Transpose to satisfy ASTRA dimensions if loading projection data:
        if projections:
            self._parent.data._data = numpy.transpose(self._parent.data._data, (1, 0, 2))
            self._parent.data._data = numpy.flipud(self._parent.data._data)
            self._parent.data._data = numpy.ascontiguousarray(self._parent.data._data, dtype=numpy.float32)

        # add record to the history:
        self._parent.meta.history.add_record('io.read_raw', path)

    def read_ref(self, path_files):
        '''
        Read reference flat field. Can specify a single file path or an array with several files.

        '''
        ref = []

        if type(path_files) == str:
            ref  = numpy.flipud(misc.imread(path_files, flatten= 0))
            
        elif type(path_files) == list:
            for file in path_files:
                ref.append(numpy.flipud(misc.imread(file, flatten= 0)))   
                
        else:
            self._parent.error('path_files parameter in read_ref() should be iether a full file name or a list of file names')
            
        # Swap the axses for ASTRA:
        #ref = numpy.transpose(ref, ,0,2])    

        self._parent.data._ref = ref

        # Cast to float to avoid problems with divisions in the future:
        self._parent.data._ref = numpy.float32(self._parent.data._ref)

        # add record to the history:
        self._parent.meta.history.add_record('io.read_ref', path_files)

        self._parent.message('Flat field reference image loaded.')
        
    def read_dark(self, path_file):
        '''
        Read reference flat field.

        '''
        dark = misc.imread(path_file, flatten= 0)

        if self._parent:
            self._parent.data._dark =numpy.flipud(dark)

        # Cast to float to avoid problems with divisions in the future:
        self._parent.data._dark = numpy.float32(self._parent.data._dark)

        # add record to the history:
        self._parent.meta.history.add_record('io.read_dark', path_file)

        self._parent.message('Flat field reference image loaded.')    

    def save_backup(self):
        '''
        Make a copy of data in memory, just in case.
        '''
        self._parent.data._backup = (self._parent.data._data.copy(), self._parent.meta.geometry.thetas.copy())

        # add record to the history:
        self._parent.meta.history.add_record('io.save_backup', 'backup saved')

        self._parent.message('Backup saved.')
        
        # In case the user wants to keep the backup...
        return self._parent.data._backup

    def load_backup(self, backup = None):
        '''
        Retrieve a copy of data from the backup.
        '''
        # If backup is provided:
        if not backup is None:
            self._parent.data._data = backup[0].copy()
            self._parent.meta.geometry.thetas = backup[1].copy()    
            
        else:    
            if self._parent.data._backup == [] or self._parent.data._backup is None:
                self._parent.error('I can`t find a backup, master.')

            self._parent.data._data = self._parent.data._backup[0].copy()
            self._parent.meta.geometry.thetas = self._parent.data._backup[1].copy()

            # Clean memory:
            self._parent.data._backup = None
            gc.collect()

        # Add record to the history:
        self._parent.meta.history.add_record('io.load_backup', 'backup loaded')

        self._parent.message('Backup loaded.')

    def read_meta(self, path = '', kind = 'flexray'):
        '''
        Parser for the metadata file that contains information about the acquisition system.
        '''
        path = self._update_path(path)

        if (str.lower(kind) == 'skyscan'):
            
            # Parse the SkyScan log file
            self._parse_skyscan_meta(path)

        elif (str.lower(kind) == 'flexray'):
            # Parse the SkyScan log file
            self._parse_flexray_meta(path)

        elif (str.lower(kind) == 'asi'):
            # Parse the ASI log file
            self._parse_asi_meta(path)
            
        # add record to the history:
        self._parent.meta.history.add_record('io.read_meta', path)

        self._parent.message('Meta data loaded.')
        
    def read_flexray(self, path):
        '''
        Read raw projecitions, dark and flat-field, scan parameters,
        '''
        self.read_dark(path + '/di0000.tif')
        
        self.read_ref([path + '/io0000.tif', path + '/io0001.tif'])
        
        self.read_raw(path)
        
        self.read_meta(path + '', kind = 'flexray')
        
    # **************************************************************
    # Parsers for metadata files
    # **************************************************************
    def _parse_asi_meta(self, path = ''):
        '''
        Use this routine to parse a text file generated by Navrit
        '''
        path = self._update_path(path)

        # Try to find the log file in the selected path
        log_file = [x for x in os.listdir(path) if (os.path.isfile(os.path.join(path, x)) and 'txt' in os.path.join(path, x))]

        if len(log_file) == 0:
            raise FileNotFoundError('Log file not found in path: ' + path)
        if len(log_file) > 1:
            self._parent.warning('Found several log files. Currently using: ' + log_file[0])
            log_file = os.path.join(path, log_file[0])
        else:
            log_file = os.path.join(path, log_file[0])
       
        # Create an empty dictionary:
        records = {}

        # Create a dictionary of keywords (skyscan -> our geometry definition):
        geom_dict = {'pixel pitch':'det_pixel', 'object to source':'src2obj', 'object to detector':'det2obj', 'tube voltage':'voltage', 'tube power':'power', 'tube current':'current'}

        with open(log_file, 'r') as logfile:
            for line in logfile:
                name, var = line.partition("=")[::2]
                name = name.strip().lower()

                # If there is unit after the value:
                if len(var.split()) > 1:
                    unit = var.split()[1]
                    var = var.split()[0]

                # If name contains one of the keys (names can contain other stuff like units):
                geom_key = [geom_dict[key] for key in geom_dict.keys() if key in name]

                if geom_key != []:
                    factor = self._parse_unit(unit)
                    records[geom_key[0]] = float(var)*factor

        # Convert the geometry dictionary to geometry object:
        self._parent.meta.geometry.src2obj = records['src2obj']
        self._parent.meta.geometry.det2obj = records['det2obj']
        self._parent.meta.geometry.det_pixel = [records['det_pixel'], records['det_pixel']] * self._parse_unit('um') 
        self._parent.meta.geometry.theta_range = [0, 2*numpy.pi]
        
        # Set some physics properties:
        self._parent.meta.physics['voltage'] = records['voltage']
        self._parent.meta.physics['power'] = records['power']
        self._parent.meta.physics['current'] = records['current']

    def _parse_flexray_meta(self, path = ''):
        '''
        Use this routine to parse 'scan settings.txt' file generated by FlexRay machine
        '''
        
        import re
        
        path = self._update_path(path)

        # Try to find the log file in the selected path
        log_file = [x for x in os.listdir(path) if (os.path.isfile(os.path.join(path, x)) and 'data settings XRE.txt' in os.path.join(path, x))]

        if len(log_file) == 0:
            raise FileNotFoundError('Log file not found in path: ' + path)
            
        if len(log_file) > 1:
            #raise UserWarning('Found several log files. Currently using: ' + log_file[0])
            self._parent.warning('Found several log files. Currently using: ' + log_file[0])
            log_file = os.path.join(path, log_file[0])
        else:
            log_file = os.path.join(path, log_file[0])

        # Create an empty dictionary:
        records = {}

        # Create a dictionary of keywords (skyscan -> our geometry definition):
        geom_dict = {'voxel size':'img_pixel', 'sod':'src2obj', 'sdd':'src2det', '# projections':'theta_n',
                     'last angle':'last_angle', 'start angle':'first_angle', 'tube voltage':'voltage', 'tube power':'power', 'Exposure time (ms)':'exposure'}

        with open(log_file, 'r') as logfile:
            for line in logfile:
                name = line.partition('=')[0]
                var = re.findall('"([^"]*)"', line)

                name = name.strip().lower()
                
                # If name contains one of the keys (names can contain other stuff like units):
                geom_key = [geom_dict[key] for key in geom_dict.keys() if key in name]

                if geom_key != []:
                    #factor = self._parse_unit(name)
                    factor = 1 # there are no units mentioned in flexray file
                    
                    if var != []:
                        records[geom_key[0]] = float(var[0])*factor

        # Convert the geometry dictionary to geometry object:        
        self._parent.meta.geometry.src2obj = records['src2obj']
        self._parent.meta.geometry.det2obj = records['src2det'] - records['src2obj']
        self._parent.meta.geometry.img_pixel = [records['img_pixel'] * self._parse_unit('um'), records['img_pixel'] * self._parse_unit('um')]  
        self._parent.meta.geometry.theta_range = [records['first_angle']* self._parse_unit('deg'), records['last_angle']* self._parse_unit('deg')] 
        
        if self._parent.data.data.size > 0:                                              
            self._parent.meta.geometry.init_thetas(theta_n = self._parent.data.shape[1])

        # Set some physics properties:
        self._parent.meta.physics['voltage'] = records.get('voltage')
        self._parent.meta.physics['power'] = records.get('power')
        self._parent.meta.physics['current'] = records.get('current')
        self._parent.meta.physics['current'] = records.get('exposure')

    def _parse_skyscan_meta(self, path = ''):

        path = self._update_path(path)

        # Try to find the log file in the selected path
        log_file = [x for x in os.listdir(path) if (os.path.isfile(os.path.join(path, x)) and os.path.splitext(os.path.join(path, x))[1] == '.log')]

        if len(log_file) == 0:
            raise FileNotFoundError('Log file not found in path: ' + path)
        if len(log_file) > 1:
            #raise UserWarning('Found several log files. Currently using: ' + log_file[0])
            self._parent.warning('Found several log files. Currently using: ' + log_file[0])
            log_file = os.path.join(path, log_file[0])
        else:
            log_file = os.path.join(path, log_file[0])

        #Once the file is found, parse it
        records = {}

        # Create a dictionary of keywords (skyscan -> our geometry definition):
        geom_dict = {'camera pixel size': 'det_pixel', 'image pixel size': 'img_pixel', 'object to source':'src2obj', 'camera to source':'src2det', 'Number of Files':'theta_n',
        'optical axis':'optical_axis', 'rotation step':'rot_step', 'exposure':'exposure', 'source voltage':'voltage', 'source current':'current',
        'camera binning':'det_binning', 'image rotation':'det_tilt', 'number of rows':'det_rows', 'number of columns':'det_cols', 'postalignment':'det_offset', 'object bigger than fov':'roi_fov'}
        
        with open(log_file, 'r') as logfile:
            for line in logfile:
                name, val = line.partition("=")[::2]
                name = name.strip().lower()
                
                # If name contains one of the keys (names can contain other stuff like units):
                geom_key = [geom_dict[key] for key in geom_dict.keys() if key in name]
                
                if geom_key != [] and (geom_key[0] != 'det_binning') and (geom_key[0] != 'det_offset') and (geom_key[0] != 'large_object') :
                    factor = self._parse_unit(name)
                    records[geom_key[0]] = float(val)*factor
                elif geom_key != [] and geom_key[0] == 'det_binning':
                    # Parse with the 'x' separator
                    bin_x, bin_y = val.partition("x")[::2]
                    records[geom_key[0]] = [float(bin_x), float(bin_y)]
                elif geom_key != [] and geom_key[0] == 'det_offset':
                    records[geom_key[0]][0] = float(val)
                elif geom_key != [] and geom_key[0] == 'roi_fov':
                    if val.lower == 'off':
                        records[geom_key[0]] = False
                    else:
                        records[geom_key[0]] = True
     
        # Convert the geometry dictionary to geometry object:        
        self._parent.meta.geometry.src2obj = records['src2obj']
        self._parent.meta.geometry.det2obj = records['src2det'] - records['src2obj']

        self._parent.meta.geometry.src2obj = records['src2obj']
        self._parent.meta.geometry.roi_fov = records['roi_fov']

        self._parent.meta.geometry.det_pixel = [records['det_pixel'] * records['det_binning'], records['det_pixel'] * records['det_binning']]
        self._parent.meta.geometry.init_theta([0, (records['theta_n']-1) * records['rot_step']], records['theta_n']-1)
        
        # Set some physics properties:
        self._parent.meta.physics['voltage'] = records['voltage']
        self._parent.meta.physics['power'] = records['power']
        self._parent.meta.physics['current'] = records['current']
        self._parent.meta.physics['exposure'] = records['exposure']
  
        # Convert optical axis into detector offset (skyscan measures lines from the bottom)
        self._parent.meta.geometry.modifiers['det_vrt'] = records['optical_axis'] - records['det_rows']/2.0
        self._parent.meta.geometry.rotation_center = records['det_offset'] - records['det_cols']/2.0
        
        # Convert detector tilt into radian units (degrees assumed)
        if 'det_tilt' in geometry:
            self._parent.meta.geometry.modifiers['det_rot'] = records['det_tilt'] * self._parse_unit('deg')
            
    def _parse_unit(self, string):
            # Look at the inside of trailing parenthesis
            unit = ''
            factor = 1.0
            if string[-1] == ')':
                unit = string[string.rfind('(')+1:-1].strip().lower()
            else:
                unit = string.strip().lower()

            units_dictionary = {'um':0.001, 'mm':1, 'cm':10.0, 'm':1e3, 'rad':1, 'deg':numpy.pi / 180.0, 'ms':1, 's':1e3, 'us':0.001, 'kev':1, 'mev':1e3, 'ev':0.001,
                                'kv':1, 'mv':1e3, 'v':0.001, 'ua':1, 'ma':1e3, 'a':1e6, 'line':1}    
            
            if unit in units_dictionary.keys():
                factor = units_dictionary[unit]
            else:
                factor = 1
                self._parent.warning('Unknown unit: ' + unit + '. Skipping.')

            return factor            
        
    
    def read_skyscan_thermalshifts(self, path = ''):
        path = self._update_path(path)

        # Try to find the log file in the selected path
        fname = [x for x in os.listdir(path) if (os.path.isfile(os.path.join(path, x)) and os.path.splitext(os.path.join(path, x))[1] == '.csv')]
        if len(fname) == 0:
            raise FileNotFoundError('XY shifts csv file not found in path: ' + path)
        if len(fname) > 1:
            #raise UserWarning('Found several log files. Currently using: ' + log_file[0])
            self._parent.warning('Found several csv files. Currently using: ' + fname[0])
            fname = os.path.join(path, fname[0])
        else:
            fname = os.path.join(path, fname[0])
            
        with open(fname) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=['slice', 'x','y'])
            for row in reader:
                # Find the first useful row
                if row['slice'].replace('.','',1).isdigit():
                    break
    
            
            shifts = [[float(row['x']), float(row['y'])]]
            #[row['x'], row['y']]
            for row in reader:
                shifts.append([float(row['x']), float(row['y'])])
        
            self._parent.meta.geometry.add_thermal_shifts(numpy.array(shifts))
    
    def compress_zip(self, path = '/ufs/kostenko/GitProjects/Print3D/fossil_henri_0/'):
        '''
        Create a zip file from saved slices. Need this for 3D printing.
        '''
        from zipfile import ZipFile

        # initializing empty file paths list
        file_paths = []

        # crawling through directory and subdirectories
        for root, directories, files in os.walk(path):
            for filename in files:
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)
        
        with ZipFile(os.path.join(path, 'fossil_.zip'),'w') as zip:
                # writing each file one by one
                for full_path in file_paths:
                    zip.write(full_path, os.path.basename(full_path))
        
    def save_titan_png(self, path = '', fname='data', axis = 2):
        '''
        '''
        from scipy import misc
        import os
                
        data = self._parent.data._data
        
        # Make sure that the path exists
        self._make_path(path)
        
        for ii in range(0, data.shape[axis]):
            fname_tmp = fname+"_%04d.png" % (ii+1)
            
            im = numpy.zeros([1080, 1920])
            
            margin = 400
            
            if axis == 0:
                im[:data.shape[1], margin:(margin+data.shape[2])] = data[ii,:,:]
            elif axis == 1:
                im[:data.shape[0], margin:(margin+data.shape[2])] = data[:,ii,:]
            elif axis == 2:
                im[:data.shape[0], margin:(margin+data.shape[1])] = data[:,:,ii]
                   
            misc.imsave(name = os.path.join(path, fname_tmp), arr = im)
            
            print("\r Progress {:2.1%}".format((ii+1) / data.shape[axis]), end=" ")        
            
    def save_tiff(self, path = '', fname='data', digits = 4, axis = 0):        
        '''
        Saves the data to tiff files
        '''
        from PIL import Image
        
        # Make sure that the path exists
        self._make_path(path)
        
        if self._parent.data._data is not None:
            
            # First check if digit is large enough, otherwise add a digit
            im_nb = self._parent.data._data.shape[0]
            if digits <= numpy.log10(im_nb):
                digits = int(numpy.log10(im_nb)) + 1
                
            path = self._update_path(path)
            fname = os.path.join(path, fname)
            
            for i in range(0,self._parent.data._data.shape[0]):
                fname_tmp = fname
                fname_tmp += '_'
                fname_tmp += str(i).zfill(digits)
                fname_tmp += '.tiff'
                if axis == 0:
                    im = Image.fromarray(self._parent.data._data[i,:,:])
                elif axis == 1:
                    im = Image.fromarray(self._parent.data._data[:,i,:])
                elif axis == 2:
                    im = Image.fromarray(self._parent.data._data[:,:,i])
                else:
                    self._parent.error('Wrong axis!')
                    
                im.save(fname_tmp)

    def save_slices(self, path = '', fname='data', digits = 4):
        '''
        Saves the data to tiff files
        '''
        from PIL import Image
        
        # Make sure that the path exists
        self._make_path(path)
        
        if self._parent.data._data is not None:
            
            # First check if digit is large enough, otherwise add a digit
            im_nb = self._parent.data._data.shape[0]
            if digits <= numpy.log10(im_nb):
                digits = int(numpy.log10(im_nb)) + 1
                
            path = self._update_path(path)
            fname = os.path.join(path, fname)
            
            for i in range(0,self._parent.data._data.shape[0]):
                fname_tmp = fname
                fname_tmp += '_'
                fname_tmp += str(i).zfill(digits)
                fname_tmp += '.tiff'
                im = Image.fromarray(self._parent.data._data[i,:,:])
                im.save(fname_tmp)
                
    def save_projections(self, path = '', fname='data', digits = 4):
        '''
        Saves the data to tiff files
        '''
        from PIL import Image
        
        # Make sure that the path exists
        self._make_path(path)
        
        if self._parent.data._data is not None:
        
            # First check if digit is large enough, otherwise add a digit
            im_nb = self._parent.data._data.shape[1]
            if digits <= numpy.log10(im_nb):
                digits = int(numpy.log10(im_nb)) + 1
                
            path = self._update_path(path)
            fname = os.path.join(path, fname)
            
            for i in range(0,self._parent.data._data.shape[1]):
                fname_tmp = fname
                fname_tmp += '_'
                fname_tmp += str(i).zfill(digits)
                fname_tmp += '.tiff'
                im = Image.fromarray(numpy.flipud(self._parent.data._data[:,i,:]))
                im.save(fname_tmp)
                #misc.imsave(name = os.path.join(path, fname_tmp), arr = self._parent.data._data[i,:,:])
                #dxchange.writer.write_tiff_stack(self._parent.data.get_data(),fname=os.path.join(path, fname), axis=axis,overwrite=True)

# **************************************************************
#           DISPLAY class and subclasses
# **************************************************************
class display(subclass):
    '''
    This is a collection of display tools for the raw and reconstructed data
    '''
    
    def __init__(self, parent = []):
        subclass.__init__(self, parent)
        
        self._cmap = 'gray'
        self._dynamic_range = []
        self._mirror = False
        self._upsidedown = False
        
    def set_options(self, cmap = 'gray', dynamic_range = [], mirror = False, upsidedown = False):    
        '''
        Set options for visualization.
        '''
        self._cmap = cmap
        self._dynamic_range = dynamic_range
        self._mirror = mirror
        self._upsidedown = upsidedown

    def _figure_maker_(self, fig_num):
        '''
        Make a new figure or use old one.
        '''
        if fig_num:
            plt.figure(fig_num)
        else:
            plt.figure()


    def slice(self, slice_num = None, dim_num = 0, fig_num = [], mirror = False, upsidedown = False):
        '''
        Display a 2D slice of 3D volumel
        '''
        self._figure_maker_(fig_num)

        if slice_num is None:
            slice_num = self._parent.data.shape[dim_num] // 2

        img = extract_2d_array(dim_num, slice_num, self._parent.data.data)

        if mirror: img = numpy.fliplr(img)
        if upsidedown: img = numpy.flipud(img)
        
        #plt.imshow(img, cmap = self._cmap, origin='lower', vmin = self._dynamic_range[0], vmax =self._dynamic_range[1])
        plt.imshow(img, cmap = self._cmap, origin='lower')
        plt.colorbar()
        plt.show()
        plt.pause(0.0001)

    def slice_movie(self, dim_num = 1, fig_num = []):
        '''
        Display a 2D slice of 3D volumel
        '''
        self._figure_maker_(fig_num)

        slice_num = 0
        img = extract_2d_array(dim_num, slice_num, self._parent.data.data)
        fig = plt.imshow(img, cmap = self._cmap)

        plt.colorbar()
        plt.show()

        for slice_num in range(1, self._parent.data.shape[dim_num]):
            img = extract_2d_array(dim_num, slice_num, self._parent.data.data)
            fig.set_data(img)
            plt.show()
            plt.title(slice_num)
            plt.pause(0.0001)

    def projection(self, dim_num = 1, fig_num = []):
        '''
        Get a projection image of the 3d data.
        '''
        self._figure_maker_(fig_num)

        img = self._parent.data.data.sum(dim_num)
        plt.imshow(img, cmap = self._cmap)
        
        plt.colorbar()
        plt.show()
        plt.pause(0.0001)
            
    def max_projection(self, dim_num = 1, fig_num = []):
        '''
        Get maximum projection image of the 3d data.
        '''
        self._figure_maker_(fig_num)

        img = self._parent.data.data.max(dim_num)
        plt.imshow(img, cmap = self._cmap)
        
        plt.colorbar()
        plt.show()
        plt.pause(0.0001)

    def min_projection(self, dim_num = 1, fig_num = []):
        '''
        Get maximum projection image of the 3d data.
        '''
        self._figure_maker_(fig_num)

        img = self._parent.data.data.min(dim_num)
        plt.imshow(img, cmap = self._cmap)
        
        plt.colorbar()
        plt.show()
        plt.pause(0.0001)

    def volume_viewer(self, orientation = 'x_axes', min_max = []):
        '''
        Use mayavi to view the volume slice by slice
        '''
        data = self._parent.data.data.copy()

        # Clip intensities if needed:
        if numpy.size(min_max) == 2:
            data[data < min_max[0]] = min_max[0]
            data[data < min_max[1]] = min_max[1]

        mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(data),
                            plane_orientation=orientation,
                            slice_index=0, colormap='gray')
        mlab.colorbar()

        mlab.outline()

    def render(self, min_max = []):
        '''
        Render volume using mayavi routines
        '''
        data = self._parent.data.data.copy()

        # Clip intensities if needed:
        if numpy.size(min_max) == 2:
            data[data < min_max[0]] = min_max[0]
            data[data < min_max[1]] = min_max[1]

        vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(numpy.fliplr(data)), vmin = 0.001, vmax = 0.01)
        mlab.colorbar()

        # Adjust colors:
        ctf = ColorTransferFunction()

        for ii in numpy.linspace(0, 1, 10):
            ctf.add_hsv_point(ii * 0.01, 0.99 - ii, 1, 1)

        ctf.range= [0, 1]
        vol._volume_property.set_color(ctf)
        vol._ctf = ctf
        vol.update_ctf = True

        mlab.outline()

# **************************************************************
#           ANALYSE class and subclasses
# **************************************************************
from scipy.ndimage import measurements

class analyse(subclass):
    '''
    This is an anlysis toolbox for the raw and reconstructed data
    '''
    
    def l2_norm(self):
        return numpy.sum(self._parent.data.data ** 2)
        
    def l1_norm(self):
        return numpy.sum(numpy.abs(self._parent.data.data))

    def mean(self):
        return numpy.mean(self._parent.data.data)

    def min(self):
        return numpy.min(self._parent.data.data)

    def max(self):
        return numpy.max(self._parent.data.data)

    def center_of_mass(self):
        return measurements.center_of_mass(self._parent.data.data.max(1))

    def histogram(self, nbin = 256, plot = True, log = False):

        mi = self.min()
        ma = self.max()

        a, b = numpy.histogram(self._parent.data.data, bins = nbin, range = [mi, ma])

        # Set bin values to the middle of the bin:
        b = (b[0:-1] + b[1:]) / 2

        if plot:
            plt.figure()
            if log:
                plt.semilogy(b, a)
            else:
                plt.plot(b, a)
            plt.show()

        return a, b       

# **************************************************************
#           PROCESS class and subclasses
# **************************************************************
from scipy import ndimage
import simulate
#from tomopy.recon import rotation
import scipy.ndimage.interpolation as interp

class process(subclass):
    '''
    Various preprocessing routines
    '''
    def arbitrary_function(self, func):
        '''
        Apply an arbitrary function:
        '''
        print(func)
        self._parent.data.data = func(self._parent.data.data)

        # add a record to the history:
        self._parent.meta.history.add_record('process.arbitrary_function', func.__name__)

        self._parent.message('Arbitrary function applied.')

    def interpolate_holes(self, kernel = [3,3,3], min_valid = 0, max_valid = numpy.inf):
        '''
        Fill in the holes, for instance, saturated pixels.
        '''
        mask = numpy.float32((self._parent.data._data > min_valid) & (self._parent.data._data < max_valid))
        tmp = ndimage.filters.gaussian_filter(self._parent.data._data * mask, sigma = kernel) / ndimage.filters.gaussian_filter(mask, sigma = kernel)
        
        tmp[~numpy.isfinite(tmp)] = 0

        mask = ~numpy.bool8(mask)
        
        self._parent.data._data[mask] = tmp[mask]
        
    def residual_rings(self, kernel=[3, 3]):
        '''
        Apply correction by computing outlayers .
        '''
        # Compute mean image of intensity variations that are < 5x5 pixels
        self._parent.message('Starting pixel callibration. This can take some time, depending on the kernel size.')
        
        res = self._parent.data.data - ndimage.filters.median_filter(self._parent.data.data, [kernel[1], 1, kernel[0]])
        res = res.mean(1)
        self._parent.data.data -= res.reshape((res.shape[0], 1, res.shape[1]))
        self._parent.meta.history.add_record('residual_rings', 1)
        self._parent.message('Residual ring correcion applied.')

    def medipix_quadrant_shift(self):
        '''
        Expand the middle line
        '''
        self._parent.data.data[:,:, 0:self._parent.data.shape[2]//2 - 2] = self._parent.data.data[:,:, 2:self._parent.data.shape[2]//2]
        self._parent.data.data[:,:, self._parent.data.shape[2]//2 + 2:] = self._parent.data.data[:,:, self._parent.data.shape[2]//2:-2]

        # Fill in two extra pixels:
        for ii in range(-2,2):
            closest_offset = -3 if (numpy.abs(-3-ii) < numpy.abs(2-ii)) else 2
            self._parent.data.data[:,:, self._parent.data.shape[2]//2 - ii] = self._parent.data.data[:,:, self._parent.data.shape[2]//2 + closest_offset]

        # Then in columns
        self._parent.data.data[0:self._parent.data.shape[0]//2 - 2,:,:] = self._parent.data.data[2:self._parent.data.shape[0]//2,:,:]
        self._parent.data.data[self._parent.data.shape[0]//2 + 2:, :, :] = self._parent.data.data[self._parent.data.shape[0]//2:-2,:,:]

        # Fill in two extra pixels:
        for jj in range(-2,2):
            closest_offset = -3 if (numpy.abs(-3-jj) < numpy.abs(2-jj)) else 2
            self._parent.data.data[self._parent.data.shape[0]//2 - jj,:,:] = self._parent.data.data[self._parent.data.shape[0]//2 + closest_offset,:,:]

        self._parent.meta.history.add_record('Quadrant shift', 1)
        self._parent.message('Medipix quadrant shift applied.')

    def flat_field(self, kind=''):
        '''
        Apply flat field correction.
        '''

        if (str.lower(kind) == 'skyscan'):
            if self._parent.meta.geometry.roi_fov:
                self._parent.message('Object is larger than the FOV!')
                
                air_values = numpy.ones_like(self._parent.data.data[:,:,0]) * 2**16 - 1
            else:
                air_values = numpy.max(self._parent.data.data, axis = 2)
                
            air_values = air_values.reshape((air_values.shape[0],air_values.shape[1],1))
            self._parent.data.data = self._parent.data.data / air_values
            
            # add a record to the history: 
            self._parent.meta.history.add_record('process.flat_field', 'skyscan')

            self._parent.message('Skyscan flat field correction applied.')

        else:
            # Treat projections separately, to save memory:
            for ii in range(0, self._parent.data.shape[1]):
                
                # Get the reference image for the current projection:
                ref = self._parent.data.get_ref(proj_num = ii)
                
                if numpy.min(ref) <= 0:
                    self._parent.warning('Flat field reference image contains zero (or negative) values! Will replace those with little tiny numbers.')

                    tiny = ref[ref > 0].min()
                    ref[ref <= 0] = tiny

                # If there is dark field, use it.
                if not self._parent.data._dark is None:    
                    self._parent.data.data[:, ii, :] = self._parent.data.data[:, ii, :] - self._parent.data._dark
                    ref = ref - self._parent.data._dark
                    
                # Use flat field:
                self._parent.data.data[:, ii, :] = self._parent.data.data[:, ii, :] / ref    
            
            # add a record to the history:
            self._parent.meta.history.add_record('process.flat_field', 1)

            self._parent.message('Flat field correction applied.')


    def short_scan_weights(self, fan_angle):
        '''
        Apply parker weights correction.
        '''
        def _Parker_window(theta, gamma, fan):
            weight = 0.0
            if (0 <= theta < 2*(gamma+fan)):
                weight = numpy.sin((numpy.pi/4)*(theta/(gamma+fan)))**2
            elif (2*(gamma+fan) <= theta < numpy.pi + 2*gamma):
                weight = 1.0
            elif (numpy.pi + 2*gamma <= theta < numpy.pi + 2*fan):
                weight = numpy.sin((numpy.pi/4)*((numpy.pi + 2*fan - theta)/(gamma+fan)))**2
            else:
                weight = 0.0
            return weight

        weights = numpy.zeros_like(self._parent.data.data, dtype=numpy.float32)
        sdd = self._parent.meta.geometry.src2det
        for u in range(0,weights.shape[2]):
            weights[:,:,u] = u

        weights = weights - weights.shape[2]/2
        weights = self._parent.meta.geometry.det_pixel[1]*weights
        weights = numpy.arctan(weights/sdd)

        theta = self._parent.meta.geometry.thetas
        
        for ang in range(0,theta.shape[0]):
            tet = theta[ang]
            for u in range(0, weights.shape[2]):
                weights[:,ang,u] = _Parker_window(theta = tet, gamma = weights[0,ang,u], fan=fan_angle)

        self._parent.data.data *= weights
        # add a record to the history:
        self._parent.meta.history.add_record('process.short_scan', 1)

        self._parent.message('Short scan correction applied.')


    def log(self, air_intensity = 1.0, lower_bound = -numpy.log(2), upper_bound = numpy.log(2**12)):
        '''
        Apply -log(x) to the sinogram. Lower and upper bounds are given for the attenuation coefficient.
        Default upper_bound assumes that values below 1/2^12 are outside of the dynamic range of the camera.
        Lover bound of - log(2) means there should be no intensity values higher than 2 after normalization.
        '''
        # Check if the log was already applied:
        #self._parent._check_double_hist('process.log(upper_bound)')

        # If not, apply!
        if (air_intensity != 1.0):
            self._parent.data.data /= air_intensity
            
        # Apply a bound to large values:
        numpy.clip(self._parent.data.data, a_min = numpy.exp(-upper_bound), a_max = numpy.exp(-lower_bound), out = self._parent.data.data)
                   
        # In-place negative logarithm
        numpy.log(self._parent.data.data, out = self._parent.data.data)
        numpy.negative(self._parent.data.data, out = self._parent.data.data)
        self._parent.data.data = numpy.float32(self._parent.data.data)
        
        self._parent.message('Logarithm is applied.')
        self._parent.meta.history.add_record('process.log(air_intensity, bounds)', [air_intensity, lower_bound, upper_bound])

    def salt_pepper(self, kernel = 3):
        '''
        Gets rid of nasty speakles
        '''
        # Make a smooth version of the data and look for outlayers:
        smooth = ndimage.filters.median_filter(self._parent.data.data, [kernel, 1, kernel])
        mask = self._parent.data.data / smooth
        mask = (numpy.abs(mask) > 1.5) | (numpy.abs(mask) < 0.75)

        self._parent.data.data[mask] = smooth[mask]

        self._parent.message('Salt and pepper filter is applied.')

        self._parent.meta.history.add_record('process.salt_pepper(kernel)', kernel)

    def simple_tilt(self, tilt):
        '''
        Tilts the sinogram
        '''
        for ii in range(0, self._parent.data.shape[1]):
            self._parent.data.data[:, ii, :] = interp.rotate(numpy.squeeze(self._parent.data.data[:, ii, :]), -tilt, reshape=False)
            
        self._parent.message('Tilt is applied.')
        
    def auto_crop(self, threshold = 0.01):
        '''
        Crops projection data living only the area that is > than the maximum * threshold
        '''
        
        # Project along the theta dimension and apply threshold:
        proj = self._parent.data.data.sum(1)
        
        # Check if the logarithm was applied. If not treat the rpojections as intensity:
        if not 'process.log(air_intensity, bounds)' in self._parent.meta.history.keys:                        
            self._parent.warning('Logarithm was not found in history. Will assume that projections are intensity images.')
            proj = -numpy.log(proj / proj.max())
            
        proj = proj > proj.max() * threshold

        # Project vertically and horizontally:
        xproj = proj.max(0)
        yproj = proj.max(1)
        
        # Convert projections to indexes:
        xproj = numpy.where(xproj > 0)
        yproj = numpy.where(yproj > 0)
                   
        crop_length_x = numpy.min([xproj[0] -5, self._parent.data._data.shape[2]-6 - xproj[-1]])
        crop_length_y = numpy.min([yproj[0] -5, self._parent.data._data.shape[0]-6 - yproj[-1]])
        
        if crop_length_x < 0: crop_length_x =0
        if crop_length_y < 0: crop_length_y =0
        
        print('We decided to crop in x and y: ', [crop_length_x, crop_length_y])
        
        self.crop([crop_length_x, crop_length_y], [crop_length_x, crop_length_y])
            
        
    def crop(self, top_left, bottom_right):
        '''
        Crop the sinogram
        '''
        # Make sure there are no negative indexes:
        if top_left[0] < 0: top_left[0] = 0
        if bottom_right[0] < 0: bottom_right[1] = 0
        if top_left[1] < 0: top_left[0] = 0
        if bottom_right[1] < 0: bottom_right[1] = 0
            
        if bottom_right[1] > 0:
            self._parent.data._data = self._parent.data._data[top_left[1]:-bottom_right[1], :, :]
        else:
            self._parent.data._data = self._parent.data._data[top_left[1]:, :, :]

        if bottom_right[0] > 0:
            self._parent.data._data = self._parent.data._data[:, :, top_left[0]:-bottom_right[0]]
        else:
            self._parent.data._data = self._parent.data._data[:, :, top_left[0]:]

        self._parent.data._data = numpy.ascontiguousarray(self._parent.data._data, dtype=numpy.float32)
        gc.collect()

        self._parent.meta.history.add_record('process.ccrop(top_left, bottom_right)', [top_left, bottom_right])

        self._parent.message('Sinogram cropped.')

    def crop_centered(self, center, dimensions):
        '''
        Crop the sinogram
        '''
        self._parent.data._data = self._parent.data._data[center[0] - dimensions[0]//2:center[0] + dimensions[0]//2, :, center[1] - dimensions[1]//2:center[1] + dimensions[1]//2]
        self._parent.data._data = numpy.ascontiguousarray(self._parent.data._data, dtype=numpy.float32)
        gc.collect()

        self._parent.meta.history.add_record('process.crop_centered(center, dimensions)', [center, dimensions])

        self._parent.message('Sinogram cropped.')
        
    def equivalent_thickness(self, energy, spectrum, compound, density):
        '''
        Transfer intensity values to equivalent thickness
        '''
        sino = self._parent.data._data
        
        # Assuming that we have log data!
        if not 'process.log(air_intensity, bounds)' in self._parent.meta.history.keys:                        
            self._parent.error('Logarithm was not found in history of the projection stack. Apply log first!')
        
        print('Generating the transfer function.')
        
        # Attenuation of 1 mm:
        mu = simulate.spectra.linear_attenuation(energy, compound, density, thickness = 0.1)
        
        # Make thickness range that is sufficient for interpolation:
        thickness_min = 0
        thickness_max = sino.shape[2] * self._parent.meta.geometry.img_pixel[1]
        
        print('Assuming thickness range:', [thickness_min, thickness_max])
        thickness = numpy.linspace(thickness_min, thickness_max, 1000)
        
        exp_matrix = numpy.exp(-numpy.outer(thickness, mu))
        synth_counts = exp_matrix.dot(spectrum)
        
        plt.figure()
        plt.plot(thickness, synth_counts, 'r--', lw=4, alpha=.8)
        plt.axis('tight')
        plt.title('Intensity v.s. absorption length.')
        plt.show()
        
        synth_counts = -numpy.log(synth_counts)
        
        print('Callibration intensity range:', [synth_counts[0], synth_counts[-1]])
        print('Data intensity range:', [sino.min(), sino.max()])

        print('Applying transfer function.')    
        sino_interp = numpy.interp(sino, synth_counts, thickness)
        
        self._parent.data._data = numpy.array(sino_interp, dtype = 'float32')
        
    def bin_projections(self):
        '''
        Bin data with a factor of two in detector plane.
        '''
        self._parent.data._data = (self._parent.data._data[:, :, 0:-1:2] + self._parent.data._data[:, :, 1::2]) / 2
        self._parent.data._data = (self._parent.data._data[0:-1:2, :, :] + self._parent.data._data[1::2, :, :]) / 2

# **************************************************************
#           OPTIMIZE class and subclasses
# **************************************************************
from scipy import optimize as op
#from scipy.optimize import minimize_scalar

class optimize(subclass):
    '''
    Use various optimization schemes to better align the projection data.
    '''
    def __init__(self, parent):
        subclass.__init__(self, parent)
        
    def _modifier_l2cost(self, value, modifier = 'rotation_axis'):
        '''
        Cost function based on L2 norm of the first derivative of the volume. Computation of the first derivative is done by FDK with pre-initialized reconstruction filter.
        '''
        # Compute an image from the shifted data:
        if modifier == 'rotation_axis':
            self._parent.meta.geometry.rotation_axis_shift(value)
            
        elif modifier in self._parent.meta.geometry.modifiers.keys():
            self._parent.meta.geometry.modifiers[modifier] = value           

        else:
            self._parent.error('Modifier not found!')
            
        vol = self._parent.reconstruct.FDK()
        l2 = -vol.analyse.l2_norm()
        
        # Try to release some memory...
        vol = None
        gc.collect()
         
        return l2
        
    @staticmethod     
    def _parabolic_min(values, index, space):    
        '''
        Use parabolic interpolation to find the extremum close to the index value:
        '''
        if (index > 0) & (index < (values.size - 1)):
            # Compute parabolae:
            x = space[index-1:index+2]    
            y = values[index-1:index+2]

            denom = (x[0]-x[1]) * (x[0]-x[2]) * (x[1]-x[2])
            A = (x[2] * (y[1]-y[0]) + x[1] * (y[0]-y[2]) + x[0] * (y[2]-y[1])) / denom
            B = (x[2]*x[2] * (y[0]-y[1]) + x[1]*x[1] * (y[2]-y[0]) + x[0]*x[0] * (y[1]-y[2])) / denom
		 
            x0 = -B / 2 / A   
            
        else:
            
            x0 = space[index]

        return x0
            
    @staticmethod         
    def _full_search(func, bounds, maxiter, args):
        '''
        Performs a full search of a minimum inside the given bounds.
        '''
        func_values = numpy.zeros(maxiter)
        
        space = numpy.linspace(bounds[0], bounds[1], maxiter)
        
        print('Starting a full search')
        
        ii = 0
        for val in space:
            
            print('Step %0d / %0d' % (ii+1, maxiter))
            
            func_values[ii] = func(val, modifier = args)
            ii += 1           
        
        min_index = func_values.argmin()    
        
        return optimize._parabolic_min(func_values, min_index, space)
    
    def optimize_rotation_center(self, guess = 0, subscale = 1, full_search = True, center_of_mass = True):
        '''
        Find a center of rotation. If you can, use the center_of_mass option to get the initial guess. 
        If that fails - use a subscale larger than the potential deviation from the center. Usually, 8 or 16 works fine!
        '''
        # Usually a good initial guess is the center of mass of the projection data:
        if  center_of_mass:   
            guess = self._parent.analyse.center_of_mass()[1] - self._parent.data.shape[2] // 2
            print('The initial guess for the rotation axis shift is %0.2f' % guess)
        else:
            guess = 0

        # Downscale the data:
        while subscale >= 1:
            
            # Check that subscale is 1 or divisible by 2:
            if (subscale != 1) & (subscale // 2 != subscale / 2): self._parent.error('Subscale factor should be a power of 2! Aborting...')
            
            self._parent.message('Subscale factor %1d' % subscale)    
     
            # We will use constant subscale in the vertical direction but vary the horizontal subscale:
            samp =  [10, subscale]

            guess = subscale * self._optimize_modifier_subsample(guess / subscale, 'rotation_axis', samp, full_search, display =True)
            
            self._parent.message('Current guess is %0.2f' % guess)
            
            subscale = subscale // 2
            
        self._parent._data_sampling = [1, 1]    
        self._parent.reconstruct._initialize_ramp_filter(power = 1)    
                
        self._parent.meta.geometry.rotation_axis_shift(guess)
        
        return guess
        
    def optimize_geometry_modifier(self, modifier = 'rotation_axis', guess = 0, subscale = 8, full_search = True):
        '''
        Maximize the sharpness of the reconstruction by optimizing one of the geometry modifiers:
        '''
        
        self._parent.message('Optimization is started...')
        self._parent.message('Initial guess is %0.2f' % guess)
        
        # Downscale the data:
        while subscale >= 1:
            
            # Check that subscale is 1 or divisible by 2:
            if (subscale != 1) & (subscale // 2 != subscale / 2): self._parent.error('Subscale factor should be a power of 2! Aborting...')
            
            self._parent.message('Subscale factor %1d' % subscale)    
            
             # We will use same subscale in the both vertical and horizontal directions:
            samp =  [subscale, subscale]
            guess = subscale * self._optimize_modifier_subsample(guess / subscale, modifier, samp, full_search, display =True)
            
            self._parent.message('Current guess is %0.2f' % guess)
            
            subscale = subscale // 2
            
        self._parent._data_sampling = 1    
        self._parent.reconstruct._initialize_ramp_filter(power = 1)    
        
        return guess
        
    def _optimize_modifier_subsample(self, guess, modifier, samp = [1, 1], full_search = True, display = True):  
        '''
        Optimize a modifier using a particular sampling of the projection data.
        '''
        # 
        self._parent._data_sampling = samp
        
        # Create a ramp filter so FDK will calculate a gradient:
        self._parent.reconstruct._initialize_ramp_filter(power = 2)
                        
        if full_search:
            guess = self._full_search(self._modifier_l2cost, bounds = [guess - 2, guess + 2], maxiter = 5, args = modifier) 

        else:                    
            opt = op.minimize(self._modifier_l2cost, x0 = guess, bounds = ((guess - 2, guess + 2),), method='COBYLA', 
                                    options = {'maxiter': 15, 'disp': False}, args = modifier)
            
            guess = opt.x
        
        if display:
            vol = self._parent.reconstruct.FDK()
            vol.display.slice()
            
        return guess    
            
# **************************************************************
#           RECONSTRUCT class and subclasses
# **************************************************************
import astra
from scipy import interpolate
import math
import odl

class reconstruct(object):
    '''
    Reconstruction algorithms: FDK, SIRT, KL, FISTA etc.
    '''
    # Some precalculated masks for ASTRA:
    _projection_mask = None
    _reconstruction_mask = None
    _projection_filter = None
    
    # Values for the symmetrical volume crop:
    _vol_crop = numpy.array([0, 0, 0])

    # Display while computing:
    _display_callback = False
    
    # Link to the projection data
    _projections = []

    # ASTRA geometries:
    vol_geom = None
    proj_geom = None
        
    def __init__(self, proj):    
        
        self._projections = [proj,]           

    def auto_crop_volume(self, threshold = 0.01):
        '''
        Compute minimal volume size needed for the reconstruction using downsampled FDK. Use with care when multiple projection stacks are present.
        It will put the _data_sampling setting to the default value [1, 1].
        '''
        
        # Donsample the data before reconstructing:
        subsample = 1
        for stack in self._projections: 
            stack._data_sampling = [subsample, subsample]

        print('Calculating downsampled FDK...')
        
        # First compute a downsampled FDK ...
        self._vol_crop = numpy.array([0, 0, 0])
        vol = self.FDK()                
        
        for stack in self._projections: 
            stack._data_sampling = [1,1]
        
        # Create projections of the reconstructed volume:
        yz = vol.data.data.sum(0)
        xz = vol.data.data.sum(1)
        
        # Find where the object is:
        yz = yz > yz.max() * threshold
        xz = xz > xz.max() * threshold

        # Find projections on axes:
        xproj = xz.max(1)
        yproj = yz.max(1)
        zproj = xz.max(0)
                
        # Convert projections to indexes:
        xproj = numpy.where(xproj > 0) * subsample
        yproj = numpy.where(yproj > 0) * subsample
        zproj = numpy.where(zproj > 0) * subsample

        # Create crop:
        self._vol_crop[0] = numpy.min([xproj[0] , self._total_data_shape()[2]-1 - xproj[-1]])
        self._vol_crop[1] = numpy.min([yproj[0], self._total_data_shape()[2]-1 - yproj[-1]])
        self._vol_crop[2] = numpy.min([zproj[0], self._total_data_shape()[0]-1 - zproj[-1]])
        
        # Buffer:
        self._vol_crop -= 10
        
        # No cropping with negative values:
        self._vol_crop[self._vol_crop < 0] = 0

        print('We have decided to auto-crop using the following margins:',  self._vol_crop)
        
    def add_stack(self, proj):
        '''
        Add projection data to the reconstructor.
        '''
        self._projections.append(proj)
        
    def _total_data_shape(self):
        '''
        Combined length of the projection data along the angular dimension.
        '''
        # Shape of one projection:
        sz = self._projections[0].data.shape
        
        # Initialize projection data variable:            
        total_lengh = self._projections[0].data.shape[1]

        if numpy.size(self._projections) > 1:
            for proj in self._projections[1:]: total_lengh += proj.data.shape[1]

        return [sz[0], total_lengh, sz[2]]
        
    def _total_data(self):
        '''
        Get the combined projection data.
        '''
        
        # Return at least the data inside the first projections object:
        data = self._projections[0].data.data
        
        # Check if the data is consistent:    
        shape = data.shape
    
        if numpy.size(self._projections) > 1:
            for proj in self._projections[1:]:
                if shape != proj.data.shape: proj.error('Projection data dimensions are not consistent among datasets.')
        
        if numpy.size(self._projections) > 1:
            for proj in self._projections[1:]:
                data = numpy.append(data, proj.data.data, axis = 1)
                
        return numpy.ascontiguousarray(data)
        
    def _total_theta(self):
        
        thetas = self._projections[0].meta.geometry.thetas

        if numpy.size(self._projections) > 1:
            for proj in self._projections[1:]:
                thetas = numpy.append(thetas, self._projections[0].meta.geometry.thetas)
                
        return thetas       

    def initialize_projection_mask(self, weight_poisson = False, weight_histogram = None, pixel_mask = None):
        '''
        Genrate weights proportional to the square root of intensity that map onto the projection data

        weight_poisson  -  weight rays according to the square root of the normalized intensity

        weight_histogram  -  weight intensities according to a predifined hystogram, defined as (x, y),
        where x is intensity value and y is the corresponding weight

        pixel_mask  -  assign different weights depending on the pixel location

        '''
        # Initialize ASTRA:
        self._initialize_astra()

        # Create a volume containing only ones for forward projection weights
        sz = self._total_data_shape()

        self._projection_mask = numpy.ones(self._total_data_shape())

        # Total data:
        data = self._total_data()
        
        # if weight_poisson: introduce weights based on the value of intensity image:
        if not weight_poisson is None:
            self._projection_mask = self._projection_mask * numpy.sqrt(numpy.exp(-data))

        # if weight_histogram is provided:
        if not weight_histogram is None:
            x = weight_histogram[0]
            y = weight_histogram[1]
            f = interpolate.interp1d(x, y, kind = 'linear', fill_value = 'extrapolate')

            self._projection_mask = self._projection_mask * f(numpy.exp(-data))

        # apply pixel mask to every projection if it is provided:
        if not pixel_mask is None:
            for ii in range(0, sz[1]):
                self._projection_mask = self._projection_mask[:, ii, :] * pixel_mask

        self._projection_mask = numpy.array(self._projection_mask, dtype='float32')
       
        #prnt.message('Projection mask is initialized')

    def initialize_reconstruction_mask(self):
        '''
        Make volume mask to avoid projecting errors into the corners of the volume that are not properly sampled.
        '''
        sz = self._total_data_shape()

        self._reconstruction_mask = numpy.ones([sz[0], sz[2], sz[2]], dtype = 'bool')
        
        # Loop through all projection datasets:
        for proj in self._projections:   
            # compute radius of the defined cylinder
            det_width = sz[2] / 2
            src2obj = proj.meta.geometry.src2obj
            total = proj.meta.geometry.src2det
            pixel = proj.meta.geometry.det_pixel

            # Compute the smallest radius and cut the cornenrs:
            radius = 2 * det_width * src2obj / numpy.sqrt(total**2 + (det_width*pixel[0])**2) - 3

            # Create 2D mask:
            yy,xx = numpy.ogrid[-sz[2]//2:sz[2]//2, -sz[2]//2:sz[2]//2]

            self._reconstruction_mask = numpy.array(xx**2 + yy**2 < radius**2, dtype = 'float32')

            # Replicate to 3D:           
            self._reconstruction_mask = (numpy.tile(self._reconstruction_mask[None, :,:], [sz[0], 1, 1]))
            
        self._reconstruction_mask = numpy.ascontiguousarray(self._reconstruction_mask)
                                                       
        print('Reconstruction mask is initialized')

    def _initialize_odl(self):
        '''
        Initialize da RayTransform!
        '''
        sz = self._total_data_shape()
        
        geom = self._projections[0].meta.geometry

        # Discrete reconstruction space: discretized functions on the rectangle.
        dim = numpy.array([sz[0], sz[2], sz[2]])
        space = odl.uniform_discr(min_pt = -dim / 2 * geom['img_pixel'], max_pt = dim / 2 * geom['img_pixel'], shape=dim, dtype='float32')

        # Angles: uniformly spaced, n = 1000, min = 0, max = pi
        angle_partition = odl.uniform_partition(geom['theta_range'][0], geom['theta_range'][1], geom['theta_n'])

        # Detector: uniformly sampled, n = 500, min = -30, max = 30
        dim = numpy.array([sz[0], sz[2]])
        detector_partition = odl.uniform_partition(-dim / 2 * geom['det_pixel'], dim / 2 * geom['det_pixel'], dim)

        # Make a parallel beam geometry with flat detector
        geometry = odl.tomo.CircularConeFlatGeometry(angle_partition, detector_partition, src_radius=geom['src2obj'], det_radius=geom['det2obj'])

        # Ray transform (= forward projection). We use the ASTRA CUDA backend.
        ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')        
        
        return ray_trafo, space 
        
    def odl_TV(self, iterations = 10, lam = 0.01, min_l1_norm = False):
        '''

        '''
        
        ray_trafo, space = self._initialize_odl()
        
        # Initialize gradient operator
        gradient = odl.Gradient(space, method='forward')

        # Column vector of two operators
        op = odl.BroadcastOperator(ray_trafo, gradient)

        # Do not use the g functional, set it to zero.
        g = odl.solvers.ZeroFunctional(op.domain)
        
        # Chambol pock with TV

        # Isotropic TV-regularization i.e. the l1-norm
        # l2-squared data matching unless min_l1_norm == True
        if min_l1_norm:
            l2_norm = (odl.solvers.L1Norm(ray_trafo.range)).translated(numpy.transpose(self._parent.data.data, axes = [1, 2, 0]))
        else:
            l2_norm = (odl.solvers.L2NormSquared(ray_trafo.range)).translated(numpy.transpose(self._parent.data.data, axes = [1, 2, 0]))
        
        if not self._projection_mask is None:
            l2_norm = l2_norm * ray_trafo.range.element(self._projection_mask)
        
        l1_norm = lam * odl.solvers.L1Norm(gradient.range)

        # Combine functionals, order must correspond to the operator K
        f = odl.solvers.SeparableSum(l2_norm, l1_norm)

        # Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
        op_norm = 1.1 * odl.power_method_opnorm(op)

        tau = 1.0 / op_norm  # Step size for the primal variable
        sigma = 1.0 / op_norm  # Step size for the dual variable
        gamma = 0.2

        # Optionally pass callback to the solver to display intermediate results
        if self._display_callback:
            callback = (odl.solvers.CallbackShow())
        else:
            callback = None

        # Choose a starting point
        x = op.domain.zero()

        # Run the algorithm
        odl.solvers.chambolle_pock_solver(
                                          x, f, g, op, tau=tau, sigma=sigma, niter=iterations, gamma=gamma,
                                          callback=callback)

        return volume(numpy.transpose(x.asarray(), axes = [2, 0, 1])[:,::-1,:])

    def odl_FBP(self):
        '''

        '''
        import odl

        ray_trafo, space = self._initialize_odl()

        # FBP:
        fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Shepp-Logan', frequency_scaling=0.8)

        # Run the algorithm
        x = fbp(numpy.transpose(self._parent.data.data, axes = [1, 2, 0]))
        
        return volume(numpy.transpose(x.asarray(), axes = [2, 0, 1])[:,::-1,:])
        
    def odl_EM(self, iterations = 10):
        '''
        Expect the Maximamum

        '''
        import odl

        ray_trafo, space = self._initialize_odl()
        
        # Optionally pass callback to the solver to display intermediate results
        if self._display_callback:
            callback = (odl.solvers.CallbackShow())
        else:
            callback = None
        
        # Choose a starting point
        x = ray_trafo.domain.one()
        
        # FBP:
        odl.solvers.mlem(ray_trafo, x, numpy.transpose(self._parent.data.data, axes = [1, 2, 0]), 
                                niter = iterations, callback = callback)
        
        return volume(numpy.transpose(x.asarray(), axes = [2, 0, 1])[:,::-1,:])   
    
    ## ******************************** ASTRA ************************************************ ##
    
    def _modifiers2vectors(self, proj_geom, geom):
        '''
        Apply arbitrary geometrical modifiers to the ASTRA projection geometry vector
        ''' 
        proj_geom = astra.functions.geom_2vec(proj_geom)
            
        vectors = proj_geom['Vectors']

        for ii in range(0, vectors.shape[0]):
            
            # Define vectors:
            src_vect = vectors[ii, 0:3]    
            det_vect = vectors[ii, 3:6]    
            det_axis_hrz = vectors[ii, 6:9]           
            det_axis_vrt = vectors[ii, 9:12]           

            #Precalculate vector perpendicular to the detector plane:
            det_normal = numpy.cross(det_axis_hrz, det_axis_vrt)
            det_normal = det_normal / numpy.sqrt(numpy.dot(det_normal, det_normal))
            
            # Translations relative to the detecotor plane:
                
            #Detector shift (V):
            det_vect += geom.find_modifier('det_vrt', ii) * det_axis_vrt 
    
            #Detector shift (H):
            det_vect += geom.find_modifier('det_hrz', ii) * det_axis_hrz
    
            #Detector shift (M):
            det_vect += geom.find_modifier('det_mag', ii) * det_normal
    
            #Source shift (V):
            src_vect += geom.find_modifier('src_vrt', ii) * det_axis_vrt   
    
            #Source shift (H):
            src_vect += geom.find_modifier('src_hrz', ii) * det_axis_hrz
    
            #Source shift (M):
            src_vect += geom.find_modifier('det_mag', ii) * det_normal
    
            # Rotation relative to the detector plane:
            # Compute rotation matrix
        
            T = transforms3d.axangles.axangle2mat(det_normal, geom.find_modifier('det_rot', ii))
            
            det_axis_hrz[:] = numpy.dot(T.T, det_axis_hrz)
            det_axis_vrt[:] = numpy.dot(T, det_axis_vrt) 
        
            # Global transformation:
            # Rotation matrix based on Euler angles:
            R = euler.euler2mat(geom.find_modifier('vol_x_rot'), geom.find_modifier('vol_y_rot'), geom.find_modifier('vol_z_rot'), 'syxz')
    
            # Apply transformation:
            det_axis_hrz[:] = numpy.dot(R, det_axis_hrz)
            det_axis_vrt[:] = numpy.dot(R, det_axis_vrt) 
            src_vect[:] = numpy.dot(R, src_vect)
            det_vect[:] = numpy.dot(R, det_vect)             
            
            # Add translation:
            T = numpy.array([geom.find_modifier('vol_x_tra'), geom.find_modifier('vol_y_tra'), geom.find_modifier('vol_z_tra')])    
            src_vect[:] += T             
            det_vect[:] += T 
        
        # Modifiers applied... Extend the volume if needed                    
        return vectors
            
    def _initialize_astra(self, force_vol_size = []):

        # Shape of the thotal projection data:
        sz = self._total_data_shape()
        
        # Calculate the size of the reconstruction volume based on geometry modifier values:
        vrt_shift = 0
        hrz_shift = 0
        
        for proj in self._projections:
            shift = proj.meta.geometry.origin_shift()
            hrz_shift = numpy.max([hrz_shift, shift[0]])
            vrt_shift = numpy.max([vrt_shift, shift[1]]) 
        
        # Decide on the detector and volume size:
        det_count_x = sz[2]
        det_count_z = sz[0]
    
        # Make volume count x > detector count to include corneres of the object:
        if force_vol_size == []:    
            vol_count_x = numpy.int(sz[2] - self._vol_crop[0] + hrz_shift * 2) 
            vol_count_y = numpy.int(sz[2] - self._vol_crop[1] + hrz_shift * 2) 
            vol_count_z = numpy.int(sz[0] - self._vol_crop[2] + vrt_shift * 2)
        else:
            vol_count_x = numpy.int(force_vol_size[2])
            vol_count_y = numpy.int(force_vol_size[1])
            vol_count_z = numpy.int(force_vol_size[0])

        # Assuming that all img pixels are the same for now:
        img_pixel = self._projections[0].meta.geometry.img_pixel
        
        # Initialize the volume geometry:
        minx = -vol_count_x // 2 * img_pixel[0]
        maxx = +vol_count_x // 2 * img_pixel[0]
        miny = -vol_count_y // 2 * img_pixel[0]
        maxy = +vol_count_y // 2 * img_pixel[0]
        minz = -vol_count_z // 2 * img_pixel[1]
        maxz = +vol_count_z // 2 * img_pixel[1]

        self.vol_geom = astra.create_vol_geom(vol_count_x, vol_count_y, vol_count_z, minx, maxx, miny, maxy, minz, maxz)
        
        total_vectors = numpy.zeros((0,12))
        
        for proj in self._projections:
            det_pixel = proj.meta.geometry.det_pixel
            det2obj = proj.meta.geometry.det2obj
            src2obj = proj.meta.geometry.src2obj
            theta = proj.meta.geometry.thetas

            #M = proj.meta.geometry.magnification        

            # Initialize the local projection geometry and stitch it later with other geometries:          
            proj_geom = astra.create_proj_geom('cone', det_pixel[1], det_pixel[0], det_count_z, det_count_x, theta, src2obj, det2obj)
        
            # Convert proj_geom to vectors and apply modifiers:
            vectors = self._modifiers2vectors(proj_geom, proj.meta.geometry)
            
            # Extend vectors:
            total_vectors = numpy.append(total_vectors, vectors, axis=0) 
            
        # Make a new geometry based on the total vector:    
        self.proj_geom = astra.create_proj_geom('cone_vec', det_count_z, det_count_x, total_vectors)    
        
    def _initialize_ramp_filter(self, power = 1):
      sz = self._total_data_shape()

      # Next power of 2:
      order = numpy.int32(2 ** numpy.ceil(math.log2(sz[2]) - 1))
      n = numpy.arange(0, order)

      # Create 1D array:
      filtImpResp = numpy.zeros(order+1)

      # Populate it with ramp
      filtImpResp[0] = 1/4
      filtImpResp[1::2] = -1 / ((numpy.pi * n[1::2]) ** 2)

      filtImpResp = numpy.concatenate([filtImpResp, filtImpResp[::-1]])

      filtImpResp = filtImpResp[:-1]

      filt = numpy.real(numpy.fft.fft(filtImpResp)) ** power
      
      # Back to 32 bit...
      filt = numpy.float32(filt)

      self._projection_filter = numpy.matlib.repmat(filt, sz[1], 1)
      
    def _backproject(self, y, algorithm = 'FDK_CUDA', iterations=1, min_constraint = None):

      cfg = astra.astra_dict(algorithm)
      cfg['option'] = {}

      theta = self._total_theta()
      
      if (algorithm == 'FDK_CUDA') & ((theta.max() - theta.min()) < (numpy.pi * 2)) :
          cfg['option']['ShortScan'] = True
      
      if (min_constraint is not None):
          cfg['option']['MinConstraint'] = min_constraint
      
      geom_sz = astra.functions.geom_size(self.vol_geom)
      
      output = numpy.zeros(geom_sz, dtype=numpy.float32)
      rec_id = []
      sinogram_id = []
      alg_id = []
      
      try:
          rec_id = astra.data3d.link('-vol', self.vol_geom, output)
          sinogram_id = astra.data3d.link('-sino', self.proj_geom, y)
    
          cfg['ReconstructionDataId'] = rec_id
          cfg['ProjectionDataId'] = sinogram_id
    
          #cfg['option'] = {}
    
          # Use projection and reconstruction masks:
          if not self._reconstruction_mask is None:
              
              if self._reconstruction_mask.shape != geom_sz:
                  print('Reconstruction mask shape is wrong! We won`t use it in the reconstruction.')
              else:    
                  mask_id = astra.data3d.link('-vol', self.vol_geom, self._reconstruction_mask)
                  cfg['option']['ReconstructionMaskId'] = mask_id
    
          if not self._projection_mask is None:
              mask_id = astra.data3d.link('-sino', self.proj_geom, self._projection_mask)
              cfg['option']['SinogramMaskId'] = mask_id
    
          # Use modified filter:
          if (not self._projection_filter is None) & (algorithm != 'BP3D_CUDA'):
    
              sz = self._projection_filter.shape
    
              slice_proj_geom = astra.create_proj_geom('parallel', 1.0, sz[1], self._total_theta())
    
              filt_id = astra.data2d.link('-sino', slice_proj_geom, self._projection_filter)
              cfg['option']['FilterSinogramId'] = filt_id
              
          alg_id = astra.algorithm.create(cfg)
          astra.algorithm.run(alg_id, iterations)

      finally:
          astra.algorithm.delete(alg_id)
          #astra.data3d.delete([rec_id, sinogram_id])
          astra.data3d.delete(rec_id)
          astra.data3d.delete(sinogram_id)

      return output #astra.data3d.get(self.rec_id)
  
    

    def _forwardproject(self, x, algorithm = 'FP3D_CUDA'):

      cfg = astra.astra_dict(algorithm)
                
      output = numpy.zeros(astra.functions.geom_size(self.proj_geom), dtype=numpy.float32)
      rec_id = []
      sinogram_id = []
      alg_id = []
      
      try:          
                    
          rec_id = astra.data3d.link('-vol', self.vol_geom, x)
          
          sinogram_id = astra.data3d.link('-sino', self.proj_geom, output)
              
          cfg['VolumeDataId'] = rec_id
          cfg['ProjectionDataId'] = sinogram_id
    
          alg_id = astra.algorithm.create(cfg)
    
          #astra.data3d.store(self.rec_id, x)
          astra.algorithm.run(alg_id, 1)

      finally:
          #astra.data3d.delete([rec_id, sinogram_id])
          astra.algorithm.delete(alg_id)
          
          astra.data3d.delete(rec_id)
          astra.data3d.delete(sinogram_id)

      return output

    def FDK(self, force_vol_size = []):
        '''
        FDK reconstruction based on ASTRA.
        '''

        # Initialize ASTRA:
        self._initialize_astra(force_vol_size)

        # Run the reconstruction:
        vol = self._backproject(self._total_data(), algorithm='FDK_CUDA')
        
        # Reconstruction mask is applied only in native ASTRA SIRT. Apply it here:
        if not self._reconstruction_mask is None:
            vol = self._reconstruction_mask * vol
        
        vol = volume(vol)
        
        vol.meta.history.add_record('set data.data', [])
        return vol
        # No need to make a history record - sinogram is not changed.


    def SIRT(self, iterations = 10, min_constraint = None, force_vol_size = []):
        '''
        '''

        # Initialize ASTRA:
        self._initialize_astra(force_vol_size)

        # Run the reconstruction:
        vol = self._backproject(self._total_data(), algorithm = 'SIRT3D_CUDA', iterations = iterations, min_constraint= min_constraint)

        return volume(vol)
        # No need to make a history record - sinogram is not changed.


    def SIRT_CPU(self, iterations = 10, min_constraint = 0, relative_constraint = None, force_vol_size = []):
        '''

        '''
        # Initialize ASTRA:
        self._initialize_astra(force_vol_size)

        # Create a volume containing only ones for forward projection weights
        sz = self._total_data_shape()
        
        # Initialize weights:
        vol_ones = numpy.ones(astra.functions.geom_size(self.vol_geom), dtype=numpy.float32)
        vol = numpy.zeros_like(vol_ones, dtype=numpy.float32)
        
        vol_obj = volume(vol)
        
        weights = self._forwardproject(vol_ones)
        weights = 1.0 / (weights + (weights == 0))

        bwd_weights = 1.0 / sz[1]

        vol = numpy.zeros(astra.functions.geom_size(self.vol_geom), dtype=numpy.float32)

        for ii_iter in range(iterations):
            
            print('SIRT_CPU. Iteration %01d' % ii_iter)
            
            fwd_proj_vols = self._forwardproject(vol)

            residual = (self._total_data() - fwd_proj_vols) * weights

            if not self._projection_mask is None:
                residual *= self._projection_mask
            
            vol += bwd_weights * self._backproject(residual, algorithm='BP3D_CUDA')
            
            # If relative_constraint is used, force values below relative threshold to 0:
            if relative_constraint != None:
                threshold = vol.max() * relative_constraint
                vol[vol < threshold] = 0

            # Enforce non-negativity or similar:
            if min_constraint != None:
                vol[vol < min_constraint] = min_constraint

        vol_obj.data._data = vol    

        return vol_obj
        # No need to make a history record - sinogram is not changed.
        
    def EM_CPU(self, iterations = 10, force_vol_size = []):
        '''
        Expectation Maximization based of ASTRA projectors, however, reconstruction and input data are keept in the CPU memory.
        '''
         # Initialize ASTRA:
        self._initialize_astra(force_vol_size)

        # Create a volume containing only ones for forward projection weights
        #sz = self._total_data_shape()
        
        data = self._total_data() 
        
        print('EM_CPU. Computing weights.')
        
        # Initialize weights:
        weights = self._backproject(numpy.ones_like(data), algorithm='BP3D_CUDA')
        reg = weights.max() / 100
        weights[weights < reg] = reg
        
        # Regularization epsilon for the forward projection:
        eps = data[:,0,:].max() / 100

        # Initializae volume:
        vol = numpy.ones(astra.functions.geom_size(self.vol_geom), dtype=numpy.float32)
        vol_obj = volume(vol)
        
        for ii_iter in range(0, iterations):
            
            print('EM_CPU. Iteration %01d' % ii_iter)
            
            forward = self._forwardproject(vol)
            forward[forward < eps] = eps

            #plt.figure()
            #plt.imshow(data[200,:,:], cmap='gray')

            #plt.figure()
            #plt.imshow(forward[200,:,:], cmap='gray')
        
            vol *= self._backproject(data / forward, algorithm='BP3D_CUDA') / weights

            #plt.figure()
            #plt.imshow(vol[200,:,:], cmap='gray')
            
            if not self._projection_mask is None:
                vol *= self._projection_mask
                
        vol_obj.data._data = vol    

        return vol_obj

    def CPLS(self, iterations = 10, min_constraint = None, force_vol_size = []):
        '''
        Chambolle-Pock Least Squares
        '''
        prnt = self._parent

        # Initialize ASTRA:
        self._initialize_astra(force_vol_size)

        # Create a volume containing only ones for forward projection weights
        sz = self._total_data_shape()

        vol_ones = numpy.ones(astra.functions.geom_size(self.vol_geom), dtype=numpy.float32)
        vol = numpy.zeros_like(vol_ones, dtype=numpy.float32)
        
        sigma = self._forwardproject(vol_ones)
        sigma = 1.0 / (sigma + (sigma == 0))
        sigma_1 = 1.0  / (1.0 + sigma)
        tau = 1.0 / sz[1]

        p = numpy.zeros_like(prnt.data.data)
        ehn_sol = vol.copy()

        data = self._total_data()
        
        for ii_iter in range(iterations):
            p = (p + data - self._forwardproject(ehn_sol) * sigma) * sigma_1

            old_vol = vol.copy()
            vol += self._backproject(p, algorithm='BP3D_CUDA', min_constraint=min_constraint) * tau
            vol *= (vol > 0)

            ehn_sol = vol + (vol - old_vol)
            gc.collect()

        return volume(vol)
        # No need to make a history record - sinogram is not changed.



    def CGLS(self, iterations = 10, min_constraint = None, force_vol_size = []):
        '''
        
        '''
        # Initialize ASTRA:
        self._initialize_astra(force_vol_size)

        # Run the reconstruction:
        vol = self._backproject(self._total_data(), algorithm = 'CGLS3D_CUDA', iterations = iterations, min_constraint=min_constraint)

        return volume(vol)
        # No need to make a history record - sinogram is not changed.   
        
        
    def project_thetas(self, parameter_value = 0, parameter = 'axis_offset'):
        '''
        This routine produces a single slice with a sigle ray projected into it from each projection angle.
        Can be used as a simple diagnostics for angle coverage.
        '''
        pass
        '''
        sz = self._total_data_shape()

        # Make a synthetic sinogram:
        sinogram = numpy.zeroes((1, sz[1], sz[2]))

        # For compatibility purposes make sure that the result is 3D:
        sinogram = numpy.ascontiguousarray(sinogram)

        # Initialize ASTRA:
        theta = prnt.meta.geometry.thetas

        # Synthetic sinogram contains values of thetas at the central pixel
        ii = 0
        for theta_i in theta:
            sinogram[:, ii, sz[2]//2] = theta_i
            ii += 1

        self._initialize_astra()

        # Run the reconstruction:
        epsilon = self._parse_unit('deg') # 1 degree
        short_scan = numpy.abs(theta[-1] - 2*numpy.pi) > epsilon
        vol = self._backproject(prnt.data.data, algorithm='FDK_CUDA', short_scan=short_scan)

        return volume(vol)
        # No need to make a history record - sinogram is not changed.
        '''

    def get_vol_ROI(self):
        '''
        Computes a mask of minimal projection ROI needed to reconstruct a ROI for FDK
        '''
        
        # Initialize ASTRA:
        self._initialize_astra()

        # Run the reconstruction:
        vol = self._backproject(numpy.ones(self._total_data_shape(), dtype = 'float32'))

        return volume(vol)

    def get_proj_ROI(self, rows=[0,512], cols=[0,512], algorithm='FP3D_CUDA'):
        '''
        Computes a mask of minimal projection ROI needed to reconstruct a ROI for FDK
        '''
        
        # Initialize ASTRA:
        sz = self._total_data_shape()
        
        prnt = self._projections[0]
        
        pixel_size = prnt.meta.geometry.det_pixel
        det2obj = prnt.meta.geometry.det2obj
        src2obj = prnt.meta.geometry.src2obj
        theta = prnt.meta.geometry.thetas

        roi = numpy.zeros((sz[0],sz[2], sz[2]), dtype=numpy.float32)
        roi[rows[0]:rows[1],cols[0]:cols[1],cols[0]:cols[1]] = 1.0
        self._initialize_astra(sz, pixel_size, det2obj, src2obj, theta)

        mask = self._forwardproject(roi, algorithm=algorithm)

        # TODO: Compute the bounds of the minimal non-zero rectangle
        '''
        mask[mask>0]=1.0
        bounds = [[0,0],[0,0]]
        bounds[0][0] = numpy.min(numpy.argmax(numpy.argmax(mask,axis=2),axis=1))
        for row in range(mask.shape[0],-1,-1))
        bounds[0][1] = numpy.argmin(mask,axis=0)
        bounds[1][0] = numpy.argmax(mask,axis=2)
        bounds[1][1] = numpy.argmin(mask,axis=2)

        print(bounds)
        '''
        return mask
        
# **************************************************************
#           VOLUME class and subclasses
# **************************************************************
from skimage import morphology
import calculate as calc

class postprocess(subclass):
    
    '''
    Includes postprocessing of the reconstructed volume.
    '''

    def find_rotation(a, b):    

        # Find image shapes:
        sz = a.shape
        n = int(numpy.ceil(numpy.max(sz[0:2]) / 2) * 2)
    
        # Where to put the result of FFT:
        a_ = numpy.zeros((n*2, n*2, sz[2]), dtype = 'complex64')
        b_ = numpy.zeros((n*2, n*2, sz[2]), dtype = 'complex64')
    
        print('Comuting 2D PPFT of the volumes.')
        
        # Iterate along Z:
        for ii in range(sz[2]):
            
            a_[:,:,ii] = calc.ppft2(a[:,:,ii])
            b_[:,:,ii] = calc.ppft2(b[:,:,ii])  
        
            print("\r Progress {:2.1%}".format((ii+1) / sz[2]), end=" ")    
            
        print('Comuting FFT of the volumes along Z.')
        a_ = numpy.abs(numpy.fft.fft(a_, axis = 2))
        b_ = numpy.abs(numpy.fft.fft(b_, axis = 2))
              
        # Compute the difference and integrate along Z:
        correlation = numpy.abs(a_ - b_[:, ::-1,:]).sum(2)
        
        # Integrate along radius:
        correlation = correlation.sum(0)
    
    
        # Shift in pixels:    
        shift = (numpy.argmin(correlation) - n)      
                
        # shift in degrees:  
        if shift > 0:
            return numpy.arctan(shift / n * 2 - 1) * 360 / numpy.pi + 90
        else:
            return numpy.arctan(1 + (shift+1) / n * 2) * 360 / numpy.pi - 90 

    
    def bin_volume(self):
        '''
        Bin data with a factor of two in each direction.
        '''
        self._parent.data._data = (self._parent.data._data[:, :, 0:-1:2] + self._parent.data._data[:, :, 1::2]) / 2
        self._parent.data._data = (self._parent.data._data[0:-1:2, :, :] + self._parent.data._data[1::2, :, :]) / 2
        self._parent.data._data = (self._parent.data._data[:, 0:-1:2, :] + self._parent.data._data[:, 1::2, :]) / 2
        
    def threshold(self, threshold = None, relative = False, binary = True):
        '''
        Apply simple segmentation or discard small values.
        '''
        volume = self._parent
        
        if threshold is None: 
            #threshold = volume.analyse.max() / 2
            threshold = (numpy.percentile(volume.data.data, 98)+numpy.percentile(volume.data.data, 90) ) / 2
        elif relative: 
            threshold = (numpy.percentile(volume.data.data, 98)+numpy.percentile(volume.data.data, 90)) * threshold
        
        if binary:
            volume.data._data = numpy.array((volume.data._data > threshold) * 1.0, dtype = 'float32')
        else:
            volume.data._data[volume.data._data < threshold] = 0

    def measure_thickness(self, volume, obj_intensity = None):
        '''
        Measure average thickness of an object.
        '''

        # Apply threshold:
        self.treshold(volume, obj_intensity)

        # Skeletonize:
        skeleton = morphology.skeletonize3d(volume.data.data)

        # Compute distance across the wall:
        distance = ndimage.distance_transform_bf(volume.data.data) * 2

        # Average distance:
        return numpy.mean(distance[skeleton])
    
    def mirror(self, axis):
        self._parent.data._data = numpy.flip(self._parent.data._data, axis)
        
    def transpose(self, axes = [0, 1, 2]):
        self._parent.data._data = numpy.transpose(self._parent.data._data, axes)
    
    def rotate(self, angle, dim_num = 0):
        '''
        Rotate the volume around dim_num.
        '''
        axes = numpy.array([0, 1, 2])
        axes = axes[axes != dim_num]

        print('Rotating the volume.')
        self._parent.data._data = interp.rotate(self._parent.data._data, angle, axes = axes, reshape=False)
                
    def printer_support_seeds(self, min_dist = 2, spike_support = 100, generate_internal = True, max_spikes = 9999):
        '''
        Generate volumes with seeds for support structures for 3D printing.
        '''
        obj_vol = self._parent.data.data > 0
        
        # Map of distances from the closest support point:    
        support_distance = numpy.zeros(obj_vol.shape, dtype='float32')
        
        # Seeds of internal and external supports:
        seeds_external = numpy.zeros(obj_vol.shape, dtype='bool')
        
        if generate_internal:
            seeds_internal = numpy.zeros(obj_vol.shape, dtype='bool')
        else:
            seeds_internal = None
                    
        # Crossection that is supported by the previous slice:
        support_slice = numpy.zeros(obj_vol.shape[0:-1], dtype='Bool')
        
        # Crossection that is blocked by underlying slices:
        blocked_slice = numpy.zeros(obj_vol.shape[0:-1], dtype='Bool')
        
        # Map of projected seed locations:
        ext_seed_mask = numpy.ones(obj_vol.shape[0:-1], dtype='Bool')
        int_seed_mask = numpy.ones(obj_vol.shape[0:-1], dtype='Bool')
        
        # Coordinate grids
        xx, yy = numpy.indices(obj_vol.shape[0:-1])
        
        # Counter of seeds:
        num_spikes = 0
        
        # Vertical loop to find unsupported regions
        for ii in range(0, obj_vol.shape[2]):
            print('...')
            print("\r Progress {:2.1%}".format((ii+1) / obj_vol.shape[2]), end=" ")
                
            obj_slice = obj_vol[:, :, ii]
                
            # find distance from the supported area:
            if support_slice.sum() > 0:
                distance = obj_slice * ndimage.distance_transform_cdt(~support_slice)
            else:
                distance = obj_slice * numpy.ones(obj_vol.shape[0:-1], dtype='float16') * 1000
            
            unblocked_dist = distance * ~blocked_slice * ext_seed_mask
            blocked_dist = distance * blocked_slice * int_seed_mask
            
            # Place seeds on the most unsupported locations:
            while (num_spikes < max_spikes) & (unblocked_dist.max() > min_dist):
                    
                    # get the position of the region further away from support:
                    coords = numpy.unravel_index(unblocked_dist.argmax(), unblocked_dist.shape)
                    
                    num_spikes += 1
                    
                    # create a seed:
                    seeds_external[coords[0],coords[1],ii] = 1
                    
                    # Make a hole in the distance map:
                    ext_seed_mask = ext_seed_mask & ((xx - coords[0])**2 + (yy - coords[1])**2 > spike_support ** 2)
                    unblocked_dist *= ext_seed_mask
                    
            # Same as before but now in the blocked areas:
            while (num_spikes < max_spikes) & (generate_internal) & (blocked_dist.max() > min_dist):
                    
                    # get the position of the region further away from support:
                    coords = numpy.unravel_index(blocked_dist.argmax(), blocked_dist.shape)
                    
                    num_spikes += 1
                    
                    # create a seed:
                    seeds_internal[coords[0],coords[1],ii] = 1
        
                    # Add a hole to the total map:
                    int_seed_mask = int_seed_mask & ((xx - coords[0])**2 + (yy - coords[1])**2 > spike_support ** 2)            
                    blocked_dist *= int_seed_mask
                   
            # Next slice will be supported by areas of the current one plus some distance:    
            support_slice = ndimage.binary_dilation(obj_slice)
            
            # Keep track of what is blocked at the bottom:
            blocked_slice = blocked_slice | obj_slice
            
            support_distance[:,:,ii] = (blocked_dist + unblocked_dist) + obj_slice

        self._parent.message('Seed generation is done!')
        
        print('Total number of seeds:', num_spikes)
      
        return seeds_external, seeds_internal, support_distance
      
    def printer_support_grow(self, seeds_external, seeds_internal = None, radius_body = 12, radius_head = 4, radius_internal = 10):
        '''
        Grow internal and external support structures from seeds. Internal supports will be constant width, 
        while the external ones will have small heads, thicker bodies, and yet thicker feet.
        '''
        
        obj_vol = self._parent.data.data > 0
        
        feet_space = 30
        base_space = 5
        
        # Make body radius relative:
        radius_body = radius_body - radius_head
                    
        if not seeds_internal is None:
            print('Making seeds grow for internal spikes')
            
            # Vertical projection of seeds:
            projection = numpy.zeros(seeds_internal.shape[0:2], dtype='int16')
            
            # Grow seeds downwards:
            for ii in range(1, obj_vol.shape[2]-feet_space):
                
                # Make new seeds grow:
                if seeds_internal[:,:,-ii].max():      
                    
                    # Top phase:
                    seeds_internal[:,:,-ii] = seeds_internal[:,:,-ii] | ndimage.binary_dilation(seeds_internal[:,:,-ii], iterations = radius_internal)        
                
                # Grow old seeds:
                seeds_internal[:,:,-ii] = seeds_internal[:,:,-ii] | projection
                
                projection = seeds_internal[:,:,-ii]  

                # Stop growing seeds where there is an intesection with the object:
                projection = projection & ~obj_vol[:,:,-ii]
                
            print('Internal spikes grow feet')        
            
            projection = numpy.zeros(seeds_internal.shape[0:2], dtype='int16')
            for ii in range(obj_vol.shape[2]-1-feet_space, obj_vol.shape[2]+1):
                  
                seeds_internal[:,:,-ii] = seeds_internal[:,:,-ii] | ndimage.binary_dilation(projection)
                
                projection = seeds_internal[:,:,-ii]
        else:
            seeds_internal = 0

        print('Making seeds for external spikes')

        for ii in range(1, obj_vol.shape[2]+1):
            
            # Make new seeds grow:
            if seeds_external[:,:,-ii].max():      
                # Top phase:
                seeds_external[:,:,-ii] = ndimage.binary_dilation(seeds_external[:,:,-ii], iterations = radius_head)   
                
        print('External spikes grow width')        

        seeds_external = numpy.int16(seeds_external)

        projection = numpy.zeros(seeds_external.shape[0:2], dtype='int16')
        
        for ii in range(1, obj_vol.shape[2]+1):
            print('...')
            print("\r Progress {:2.1%}".format((ii+1) / obj_vol.shape[2]), end=" ")
            
            # Stop growing thick spikes:
            projection *= (projection < radius_body)
            
            seeds_external[:,:,-ii] = seeds_external[:,:,-ii] | (ndimage.grey_dilation(projection + (projection > 0), 
                                                                                         size = (3,3)))  
            
            projection = seeds_external[:,:,-ii]       

        print('External spikes constant width')        

        projection = numpy.zeros(seeds_external.shape[0:2], dtype='int16')
        
        for ii in range(1, obj_vol.shape[2]-feet_space):
            print('...')
            print("\r Progress {:2.1%}".format((ii+1) / (obj_vol.shape[2]-feet_space)), end=" ")
              
            seeds_external[:,:,-ii] = seeds_external[:,:,-ii] | projection  
            
            projection = seeds_external[:,:,-ii]
            
        print('External spikes grow base')        
        
        projection = numpy.zeros(seeds_external.shape[0:2], dtype='int16')
        for ii in range(obj_vol.shape[2]-1-feet_space, obj_vol.shape[2]+1):
            
            print('...')
            print("\r Progress {:2.1%}".format((ii+1) / feet_space), end=" ")
            
            seeds_external[:,:,-ii] = seeds_external[:,:,-ii] | ndimage.binary_dilation(projection)
            
            projection = seeds_external[:,:,-ii]

        # Put volumes together:
        obj_vol = (seeds_external + seeds_internal + obj_vol) > 0

        print('Creating da bottom!')        
        projection = obj_vol.max(2) > 0
        
        for ii in range(obj_vol.shape[2]-1-base_space, obj_vol.shape[2]+1):            
            obj_vol[:,:,-ii] = obj_vol[:,:,-ii] | projection 

        self._parent.data.data = obj_vol
        
        print('Yeeey!!!! Go, print me baby!')        

    def _unwrap_angle(self, angles, ceil=2 * numpy.pi):
        angles += ceil / 2.0
        angles %= ceil
        angles -= ceil / 2.0
        return angles

    def _get_log_base(self, shape, new_r):
        """
        Base of the log-polar transform.
        The following holds:
        :math:`log\_base = \exp( \ln [ \mathit{spectrum\_dim} ] / \mathit{loglpolar\_scale\_dim} )`,
        or the equivalent :math:`log\_base^{\mathit{loglpolar\_scale\_dim}} = \mathit{spectrum\_dim}`.
        """    
        old_r = shape[0] * 1.1
        
        # COnvert form diameter to radius:
        old_r /= 2.0
        log_base = numpy.exp(numpy.log(old_r) / new_r)
        
        return log_base
    
    def _logpolar2d(self, image, pcorr_shape, log_base):
               
        imshape = image.shape
        
        center = imshape[0] / 2.0, imshape[1] / 2.0
                
        # 0 .. pi = only half of the spectrum is used
        theta = numpy.zeros(imshape, dtype=numpy.float32)
        theta -= numpy.linspace(-numpy.pi, numpy.pi, imshape[0], endpoint=False)[:, None]
        
        radius = numpy.zeros(imshape, dtype=numpy.float32)
        radius += numpy.power(log_base, numpy.arange(imshape[1], dtype=float))[None, :]
        
        scale_y = 1 / imshape[0] * float(imshape[1])
        
        x = radius * numpy.cos(theta) + center[0]
        y = scale_y * radius * numpy.sin(theta) + center[1]
            
        output = numpy.empty_like(y)
        
        # map_coordinates cannot handle complex numbers
        if image.dtype == numpy.complex: 
            image = abs(image)
        
        # Transform to a new coordinate system
        image = interp.map_coordinates(image, [x,y], output=output, order=3,
                             mode="constant", cval=0)
        return output
    
    def _cross_correlation(self, im0, im1):
        """
        Computes cross-correlation between im0 and im1
        """

        # Cross-correlate:        
        im0 = abs(numpy.fft.ifft2((im0 * im1.conjugate())))        
        im0 = numpy.fft.fftshift(im0)
    
        # Find coordinates of the maximum:    
        amax = numpy.argmax(im0)
        vec = list(numpy.unravel_index(amax, im0.shape))
        
        # _compensate_fftshift is not appropriate here, this is OK.   
        vec -= numpy.array(im0.shape, int) // 2
        
        return vec
        
    def _subpixel_shift(self, im0, im1, margin = [0, 0]):
        '''
        Calculate a subpixel shift using derivative method.
        Pixels belonging to the 'margin' are discarded. Use margin if the image was shifted or infilled with zeros, for instance.
        '''   
        # Apply margins:
        im0 = im0[margin[0]:, margin[1]:]
        im1 = im1[margin[0]:, margin[1]:]
            
        if margin[0] > 0:
            im0 = im0[:-margin[0], :]
            im1 = im1[:-margin[0], :]
        
        if margin[1] > 0:
            im0 = im0[: ,:-margin[1]]
            im1 = im1[: ,:-margin[1]]
        
        # Compute derivative of im1 (symmetrical)
        im_dx = (im0[2:, :] - im0[0:-2, :]) / 2
        im_dy = (im0[:, 2:] - im0[:, 0:-2]) / 2
        
        # Difference between images will be compared with the derivative:
        im0 = (im0 - im1)
        
        # Comute abses:
        a_im_dx = numpy.abs(im_dx)
        a_im_dy = numpy.abs(im_dy)
        
        # Compute shifts down to 0.1 pixel.           
        im_dx = (im0[1:-1,:]) / im_dx   
        im_dx = numpy.mean(im_dx[a_im_dx > a_im_dx.max() / 5])
        im_dx = numpy.round(im_dx, decimals=2)
        
        im_dy = (im0[:,1:-1]) / im_dy    
        im_dy = numpy.mean(im_dy[a_im_dy > a_im_dy.max() / 5])
        im_dy = numpy.round(im_dy, decimals=2)
        
        if (numpy.abs(im_dx) > 1) | (numpy.abs(im_dy) > 1): 
            self._parent.warning('Residual derivative should be smaller than 1 in _subpixel_shift(). Subpixel shift won`t be applied.')
            
            return [0, 0]
        
        return [im_dx, im_dy] 
   
    def _get_rot_scale(self, im0, im1):
        """
        Given two images, return their scale and angle difference.
        """    
        #print('Calculate rotation and scale change.')
            
        shape = im0.shape
        
        # Pcorr_shape...
        pcorr_shape = (int(max(shape) * 1.0),) * 3
        log_base = self._get_log_base(shape, pcorr_shape[1])
        
        # Transform to logpolar:
        im0 = self._logpolar2d(im0, pcorr_shape, log_base) 
        im1 = self._logpolar2d(im1, pcorr_shape, log_base)
        
        # Compute DFTs:
        im0_fft = numpy.fft.fftshift(numpy.fft.fftn(im0))
        im1_fft = numpy.fft.fftshift(numpy.fft.fftn(im1))
        
        # Filter:
        xx, yy = numpy.indices(shape)
        rads = numpy.sqrt(yy ** 2 + xx ** 2)    
        filt = 1.0 - numpy.cos(rads) ** 2
        filt[numpy.abs(rads) > numpy.pi / 2] = 1
    
        im0_fft = im0_fft * filt
        im1_fft = im1_fft * filt
            
        (theta, rad) = self._cross_correlation(im0_fft, im1_fft)  
        
        # Apply corrections and refine the estimate in real space:
        im1 = interp.shift(im1, [theta, rad])
               
        # Only if theta and rad are small, it makes sense to compute subpixel shift.
        # Currently it will actually fail when margin is too large.
        if (abs(theta)+abs(rad)) < 10:
            subpix = self._subpixel_shift(im0, im1, [abs(theta), abs(rad)])
        
            theta = numpy.round(subpix[0] - theta, decimals = 2)
            rad = subpix[1] - rad
            
        theta = numpy.rad2deg(-numpy.pi * 2 * theta / float(pcorr_shape[0]))
        theta = -self._unwrap_angle(theta, 360)
        
        scale = 1.0 / log_base ** rad
        
        return theta, scale
    
    def _get_translation(self, im0, im1):
        """
        Find translation by cross_correlation of two 2D images.
        """
        
        #print('Calculate translateion.')
        
        x,y = self._cross_correlation(numpy.fft.fft2(im0), numpy.fft.fft2(im1))
        
        im1 = interp.shift(im1, [x, y], )
        
        dx, dy = self._subpixel_shift(im0, im1, [0, 0])
        
        x = numpy.round(x - dx, decimals=2)
        y = numpy.round(y - dy, decimals=2)
            
        return x, y, im1
        
    def volume_transform(self, vol, translations = [0, 0, 0], angles = [0, 0, 0], scale = 1.0):
    
        print('Applying 3D transformation to the volume.')
        if scale != 1.0:
            vol = interp.zoom(vol, scale)
            
        axes = numpy.array([0,1,2])    
        
        # I guess, applying rotations one by one is really wrong. But it works for now...
        for axis in range(0,3):
            if angles[axis] != 0:
                
                vol = interp.rotate(vol, angles[axis], axes=axes[axes != axis], reshape = False)
    
        if numpy.abs(translations).min() != 0:
            vol = interp.shift(vol, translations)
            
        return vol
    
    def _progress_registration(self, im0_proj, im1_proj, iteration):
        '''
        Make a plot of registration progrress
        '''
        
        plt.figure()
        plt.subplot(231)
        plt.imshow(im0_proj[0])
        plt.subplot(232)
        plt.imshow(im0_proj[1])
        plt.subplot(233)
        plt.imshow(im0_proj[2])
        plt.subplot(234)
        plt.imshow(im1_proj[0])
        plt.subplot(235)
        plt.imshow(im1_proj[1])
        plt.subplot(236)
        plt.imshow(im1_proj[2])
        plt.show()
        plt.pause(0.0001)
        
    def volume_registration(self, master_volume, numiter=1):
        '''
        Use cross-correlation approach to align two 3D volumes.
        '''
        # My parent:
        prnt = self._parent
        
        # Volumes to align:
        im1 = prnt.data._data
        im0 = master_volume.data._data
               
        shape = im0.shape
    
        if shape != im1.shape:
            prnt.error('Volumes must have the same size.')
            return None
            
        # Right now scale is fixed to one. Testing needed to see if scale estimation works.
        scale = 1.0
        angle = numpy.zeros(3)
        translation = numpy.zeros(3)
        
        # Compute projections for the volume 1:
        im0_proj = [im0.sum(0), im0.sum(1), im0.sum(2)]
        
        # We will put the transformed image in im2:
        im2 = im1.copy()
            
        for ii in range(numiter):
            
            # Compute projections for the volume 2:
            im1_proj = [im2.sum(0), im2.sum(1), im2.sum(2)]
                        
            # Find rotations based on updated projections:
            a0, scale0 = self._get_rot_scale(im0_proj[0], im1_proj[0])
            a1, scale1 = self._get_rot_scale(im0_proj[1], im1_proj[1])
            a2, scale2 = self._get_rot_scale(im0_proj[2], im1_proj[2])

            angle += numpy.array([a0, a1, a2])
            
            im2 = self.volume_transform(im1, translation, -angle, 1.0)
            
            # Find translations first (+ update the image):
            t1, t2, im1_proj[0] = self._get_translation(im0_proj[0], im1_proj[0])
            t0, t2_, im1_proj[1] = self._get_translation(im0_proj[1], im1_proj[1])
            t0_, t1_, im1_proj[2] = self._get_translation(im0_proj[2], im1_proj[2])

            
            # Show progress:
            self._progress_registration(im0_proj, im1_proj, ii)
                
            # All rotations are updated at once at the moment. 
            # For now we will discard the calculated scale.
            
            # Let's average estimated shifts and angles and use them to update the volume:
            t0 = (t0 + t0_) / 2
            t1 = (t1 + t1_) / 2
            t2 = (t2 + t2_) / 2
            
            translation += numpy.array([t0, t1, t2])
            
            print('Applying shifts:', translation)
            print('Applying rotations:', angle)        
        
        self._parent.data._data = im2
        
        return translation, angle, scale
    
class volume(object):
    data = []
    io = []
    analyse = []
    display = []
    meta = []
    postprocess = []

    # Sampling of the projection data volume:
    _data_sampling = [1, 1]

    def __init__(self, vol = [], meta_data = []):
        self.io = io(self)
        self.display = display(self)
        self.analyse = analyse(self)
        self.data = data(self)
        self.postprocess = postprocess(self)
        
        if meta_data == []:
            self.meta = meta(self)
        else:
            self.meta = meta_data.copy()

        # Get the data in:
        self.data._data = vol
        
    def copy(self):
        '''
        Deep copy of the object:
        '''
        
        # Create a new volume object to avoid copying pointers to data that is contained in the old object
        vol = volume()
        
        # Here hopefully reconstruct will not have links with the old data object
        
        vol.meta = copy.deepcopy(self.meta)
        vol.data = copy.deepcopy(self.data)
        
        return vol
        
    def message(self, msg):
        '''
        Send a message to IPython console.
        '''
        #log = logging.getLogger()
        #log.setLevel(logging.DEBUG)
        #log.debug(msg)
        print(msg)

    def error(self, msg):
        '''
        Throw an error:
        '''
        self.meta.history.add_record('error', msg)
        raise ValueError(msg)

    def warning(self, msg):
        '''
        Throw a warning. In their face!
        '''
        self.meta.history.add_record('warning', msg)
        warnings.warn(msg)

# **************************************************************
#           SINOGRAM class
# **************************************************************
import warnings
import logging
import copy

_min_history = ['io.read_raw', 'io.read_ref', 'io.read_meta', 'process.flat_field', 'process.log']

_wisdoms = ['You’d better get busy, though, buddy. The goddam *sands* run out on you \
every time you turn around. I know what I’m talking about. You’re lucky if \
you get time to sneeze in this goddam phenomenal world.',
'Work done with anxiety about results is far inferior to work done without\
such anxiety, in the calm of self-surrender. Seek refuge in the knowledge\
of Brahman. They who work selfishly for results are miserable.',
'You have the right to work, but for the work`s sake only. You have no right\
to the fruits of work. Desire for the fruits of work must never be your\
motive in working. Never give way to laziness, either.',
'Perform every action with your heart fixed on the Supreme Lord. Renounce\
attachment to the fruits. Be even-tempered [underlined by one of the \
cal-ligraphers] in success and failure; for it is this evenness of temper which is meant by yoga.',
'God instructs the heart, not by ideas but by pains and contradictions.',
'Sir, we ought to teach the people that they are doing wrong in worshipping\
the images and pictures in the temple.',
'Hard work beats talent.', 'It will never be perfect. Make it work!',
'Although, many of us fear death, I think there is something illogical about it.',
'I have nothing but respect for you -- and not much of that.',
'Prediction is very difficult, especially about the future.',
'You rely too much on brain. The brain is the most overrated organ.',
'A First Sign of the Beginning of Understanding is the Wish to Die.']

class projections(object):
    '''

    Class that will contain the raw data and links to all operations that we need
    to process and reconstruct it.

    '''
    # Public stuff:
    io = []
    meta = []
    display = []
    analyse = []
    process = []
    reconstruct = []
    data = []

    # Private:
    _wisdom_status = 1
    
    # Sampling of the projection data volume:
    _data_sampling = [1, 1]

    def __init__(self):
        self.io = io(self)
        self.meta = meta(self)
        self.display = display(self)
        self.analyse = analyse(self)
        self.process = process(self)
        self.optimize = optimize(self)
        self.reconstruct = reconstruct(self)
        self.data = data(self)
        
    def message(self, msg):
        '''
        Send a message to IPython console.
        '''
        #log = logging.getLogger()
        #log.setLevel(logging.DEBUG)
        #log.debug(msg)
        print(msg)

    def error(self, msg):
        '''
        Throw an error:
        '''
        self.meta.history.add_record('error', msg)
        raise ValueError(msg)

    def warning(self, msg):
        '''
        Throw a warning. In their face!
        '''
        self.meta.history.add_record('warning', msg)
        warnings.warn(msg)

    def what_to_do(self):

        if not self._pronounce_wisdom():
            self._check_min_hist_keys()

    def copy(self):
        '''
        Deep copy of the projections object:
        '''
        
        # Create a new projctions object to avoid copying pointers to data that is contained in the old object
        proj = projections()
        
        # Here hopefully reconstruct will not have links with the old data object
        
        proj.meta = copy.deepcopy(self.meta)
        proj.data = copy.deepcopy(self.data)
        
        return proj

    def _pronounce_wisdom(self):

        randomator = 0
        # Beef up the randomator:
        for ii in range(0, self._wisdom_status):
            randomator += numpy.random.randint(0, 100)

        # If randomator is small, utter a wisdom!
        if (randomator < 30):
           self._wisdom_status += 1

           # Pick one wisdom:
           l = numpy.size(_wisdoms)
           self.message(_wisdoms[numpy.random.randint(0, l)])

           return 1

        return 0

    def _check_min_hist_keys(self):
        '''
        Check the history and tell the user which operation should be used next.
        '''
        finished = True

        for k in _min_history:
            if not(self.meta.history.find_record(k)):
                self.message('You should use ' + k + ' as a next step')
                finished = False
                break

            if finished:
                self.message('All basic processing steps were done. Use "reconstruct.FDK" to compute filtered backprojection.')

    def _check_double_hist(self, new_key):
        '''
        Check if the operation was already done
        '''
        if new_key in self.meta.history.keys:
            self.error(new_key + ' is found in the history of operations! Aborting.')
