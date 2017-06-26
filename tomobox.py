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
        self.add_record('Created')
    
    def add_record(self, operation = '', properties = []):
        
        # Add a new history record:
        timestamp = time.ctime()
        
        # Add new record:
        self._records.append([operation, properties, timestamp, numpy.size(self._records) ])
    
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
        
        return result[-1]    
    
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
class geometry():
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
        
    def __init__(self):
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
        return self._det_pixel / self.magnification   
        
    @img_pixel.setter
    def img_pixel(self, img_pixel):
        self._det_pixel = img_pixel * self.magnification        
        
    @property
    def det_size(self):
        
        # We wont take into account the det_size from the log file. Only use actual data size.
        if self._parent.data is None:
            self._parent.warning('No raw data in the pipeline. The detector size is not known.')
        else:
            return self._det_pixel * self._parent.data.shape[::2]
        
    @property
    def thetas(self):
        return self._thetas
        
    @thetas.setter
    def thetas(self, thetas):
        self._thetas = numpy.array(thetas)
    
    @property
    def theta_n(self):
        return self._thetas.size    
        
    @property
    def theta_range(self):
        return (self._thetas[0], self._thetas[-1])
     
    @theta_range.setter    
    def theta_range(self, theta_range):
        # Change the theta range:
        if self._thetas.size() > 2:
            self._thetas = numpy.linspace(theta_range[0], theta_range[1], self._thetas.size())
        else:
            self._thetas = [theta_range[0], theta_range[1]]
        
    @property        
    def theta_step(self):
        return numpy.mean(self._thetas[1:] - self._thetas[0:-1])  
        
    def init_thetas(self, theta_range = [], theta_n = 2):
        # Initialize thetas array. You can first initialize with theta_range, and add theta_n later.
        self.thetas = numpy.linspace(theta_range[0], theta_range[1], theta_n)
        
        if theta_range == []:
            self.thetas = numpy.linspace(self.thetas[0], self.thetas[-1], theta_n)

# **************************************************************
#           META class and subclasses
# **************************************************************

class meta(subclass):
    '''
    This object contains various properties of the imaging system and the history of pre-processing.
    '''
    geometry = geometry()
    history = history()

    physics = {'voltage': 0, 'current':0, 'exposure': 0}
    lyrics = ''
    
        
# **************************************************************
#           DATA class
# **************************************************************
import scipy.interpolate as interp_sc
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
        return self._data
        
    @data.setter
    def data(self, data):
        
        if self._backup_update:
            self._backup = data.copy()
            
        self._data = data
        
        self._parent.meta.history.add_record('set data.data', [])
        
    @property        
    def shape(self):
        return self._data.shape
        
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
        sz = self.shape
        thetas = self._parent.meta.geometry.thetas

        interp_grid = numpy.transpose(numpy.meshgrid(target_theta, numpy.arange(sz[0]), numpy.arange(sz[2])), (1,2,3,0))
        original_grid = (numpy.arange(sz[0]), thetas, numpy.arange(sz[2]))

        return interp_sc.interpn(original_grid, self._data, interp_grid)

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


    print('********************')
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

def update_path(path, io):
    '''
    Memorize the path if it is provided, otherwise use the one that remember.
    '''
    if path == '':
        path = io.path
    else:
        io.path = path

    if path == '':
        io._parent.error('Path to the file was not specified.')

    return path

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

    def read_raw(self, path = '', index_range = [], y_range = [], x_range = []):   
        '''
        Read projection files automatically.
        This will look for files with numbers in the last 4 letters of their names.
        '''
        path = update_path(path, self)

        # Free memory:
        self._parent.data._data = None
        gc.collect()

        # if it's a file, read all alike, if a directory find a file to start from:
        if os.path.isfile(path):
            filename = path
            path = os.path.dirname(path)

            # if file name is provided, the range is needed:
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

        # Transpose to satisfy ASTRA dimensions:
        self._parent.data._data = numpy.transpose(self._parent.data._data, (1, 0, 2))
        self._parent.data._data = numpy.flipud(self._parent.data._data)
        self._parent.data._data = numpy.ascontiguousarray(self._parent.data._data, dtype=numpy.float32)

        # add record to the history:
        self._parent.meta.history.add_record('io.read_raw', path)

    def read_ref(self, path_file):
        '''
        Read reference flat field.

        '''
        ref  = misc.imread(path_file, flatten= 0)

        if self._parent:
            self._parent.data._ref = ref

        # Cast to float to avoid problems with divisions in the future:
        self._parent.data._ref = numpy.float32(self._parent.data._ref)

        # add record to the history:
        self._parent.meta.history.add_record('io.read_ref', path_file)

        self._parent.message('Flat field reference image loaded.')
        
    def read_dark(self, path_file):
        '''
        Read reference flat field.

        '''
        dark = misc.imread(path_file, flatten= 0)

        if self._parent:
            self._parent.data._dark = dark

        # Cast to float to avoid problems with divisions in the future:
        self._parent.data._dark = numpy.float32(self._parent.data._dark)

        # add record to the history:
        self._parent.meta.history.add_record('io.read_ref', path_file)

        self._parent.message('Flat field reference image loaded.')    

    def save_backup(self):
        '''
        Make a copy of data in memory, just in case.
        '''
        self._parent.data._backup = self._parent.data._data.copy()

        # add record to the history:
        self._parent.meta.history.add_record('io.save_backup', 'backup saved')

        self._parent.message('Backup saved.')

    def load_backup(self):
        '''
        Retrieve a copy of data from the backup.
        '''
        if self._parent.data._backup == [] or self._parent.data._backup is None:
            self._parent.error('No backup found in memory!')

        self._parent.data._data = self._parent.data._backup.copy()

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
        path = update_path(path, self)

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
        
    # **************************************************************
    # Parsers for metadata files
    # **************************************************************
    def _parse_asi_meta(self, path = ''):
        '''
        Use this routine to parse a text file generated by Navrit
        '''
        path = update_path(path,self)

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
        path = update_path(path,self)

        # Try to find the log file in the selected path
        log_file = [x for x in os.listdir(path) if (os.path.isfile(os.path.join(path, x)) and 'settings.txt' in os.path.join(path, x))]

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
                name, var = line.partition(":")[::2]
                name = name.strip().lower()

                # If name contains one of the keys (names can contain other stuff like units):
                geom_key = [geom_dict[key] for key in geom_dict.keys() if key in name]

                if geom_key != []:
                    factor = self._parse_unit(name)
                    records[geom_key[0]] = float(var)*factor

        # Convert the geometry dictionary to geometry object:        
        self._parent.meta.geometry.src2obj = records['src2obj']
        self._parent.meta.geometry.det2obj = records['det2obj']
        self._parent.meta.geometry.img_pixel = [records['img_pixel'], records['img_pixel']] * self._parse_unit('um') 
        self._parent.meta.geometry.theta_range = [records['first_angle'], records['last_angle']] * self._parse_unit('deg')
        
        # Set some physics properties:
        self._parent.meta.physics['voltage'] = records['voltage']
        self._parent.meta.physics['power'] = records['power']
        self._parent.meta.physics['current'] = records['current']
        self._parent.meta.physics['current'] = records['exposure']

    def _parse_skyscan_meta(self, path = ''):

        path = update_path(path,self)

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
            
            if unit in units_dictionary.keys:
                factor = units_dictionary[unit]
            else:
                factor = 1
                self._parent.warning('Unknown unit: ' + unit + '. Skipping.')

            return factor            
        
    
    def read_skyscan_thermalshifts(self, path = ''):
        path = update_path(path,self)

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
            
    def save_tiff(self, path = '', fname='data', digits = 4):
        '''
        Saves the data to tiff files
        '''
        from PIL import Image
        if self._parent.data._data is not None:
        # First check if digit is large enough, otherwise add a digit
            im_nb = self._parent.data._data.shape[0]
            if digits <= numpy.log10(im_nb):
                digits = int(numpy.log10(im_nb)) + 1
                
            path = update_path(path, self)
            fname = os.path.join(path, fname)
            
            for i in range(0,self._parent.data._data.shape[0]):
                fname_tmp = fname
                fname_tmp += '_'
                fname_tmp += str(i).zfill(digits)
                fname_tmp += '.tiff'
                im = Image.fromarray(self._parent.data._data[i,:,:])
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

    def _figure_maker_(self, fig_num):
        '''
        Make a new figure or use old one.
        '''
        if fig_num:
            plt.figure(fig_num)
        else:
            plt.figure()


    def slice(self, slice_num, dim_num = 0, fig_num = [], mirror = False, upsidedown = False):
        '''
        Display a 2D slice of 3D volumel
        '''
        self._figure_maker_(fig_num)

        img = extract_2d_array(dim_num, slice_num, self._parent.data.data)

        if mirror: img = numpy.fliplr(img)
        if upsidedown: img = numpy.flipud(img)
        plt.imshow(img, cmap = self._cmap, origin='lower')
        plt.colorbar()
        plt.show()

    def slice_movie(self, dim_num, fig_num = []):
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

    def projection(self, dim_num, fig_num = []):
        '''
        Get a projection image of the 3d data.
        '''
        self._figure_maker_(fig_num)

        img = self._parent.data._data.sum(dim_num)
        plt.imshow(img, cmap = self._cmap)
        
        plt.colorbar()
        plt.show()
            
    def max_projection(self, dim_num, fig_num = []):
        '''
        Get maximum projection image of the 3d data.
        '''
        self._figure_maker_(fig_num)

        img = self._parent.data._data.max(dim_num)
        plt.imshow(img, cmap = self._cmap)
        
        plt.colorbar()
        plt.show()

    def min_projection(self, dim_num, fig_num = []):
        '''
        Get maximum projection image of the 3d data.
        '''
        self._figure_maker_(fig_num)

        img = self._parent.data._data.min(dim_num)
        plt.imshow(img, cmap = self._cmap)
        
        plt.colorbar()
        plt.show()

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

    def threshold(self, threshold = None):
        '''
        Apply simple segmentation
        '''

# **************************************************************
#           PROCESS class and subclasses
# **************************************************************
from scipy import ndimage
from tomopy.recon import rotation
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
        self._parent.data._data = func(self._parent.data._data)

        # add a record to the history:
        self._parent.meta.history.add_record('process.arbitrary_function', func.__name__)

        self._parent.message('Arbitrary function applied.')

    def pixel_calibration(self, kernel=5):
        '''
        Apply correction to miscalibrated pixels.
        '''
        # Compute mean image of intensity variations that are < 5x5 pixels
        res = self._parent.data._data - ndimage.filters.median_filter(self._parent.data._data, [kernel, 1, kernel])
        res = res.mean(1)
        self._parent.data._data -= res.reshape((res.shape[0], 1, res.shape[1]))
        self._parent.meta.history.add_record('Pixel calibration', 1)
        self._parent.message('Pixel calibration correction applied.')

    def medipix_quadrant_shift(self):
        '''
        Expand the middle line
        '''
        self._parent.data._data[:,:, 0:self._parent.data.shape[2]//2 - 2] = self._parent.data._data[:,:, 2:self._parent.data.shape[2]/2]
        self._parent.data._data[:,:, self._parent.data.shape[2]//2 + 2:] = self._parent.data._data[:,:, self._parent.data.shape[2]//2:-2]

        # Fill in two extra pixels:
        for ii in range(-2,2):
            closest_offset = -3 if (numpy.abs(-3-ii) < numpy.abs(2-ii)) else 2
            self._parent.data._data[:,:, self._parent.data.shape[2]//2 - ii] = self._parent.data._data[:,:, self._parent.data.shape[2]//2 + closest_offset]

        # Then in columns
        self._parent.data._data[0:self._parent.data.shape[0]//2 - 2,:,:] = self._parent.data._data[2:self._parent.data.shape[0]//2,:,:]
        self._parent.data._data[self._parent.data.shape[0]//2 + 2:, :, :] = self._parent.data._data[self._parent.data.shape[0]//2:-2,:,:]

        # Fill in two extra pixels:
        for jj in range(-2,2):
            closest_offset = -3 if (numpy.abs(-3-jj) < numpy.abs(2-jj)) else 2
            self._parent.data._data[self._parent.data.shape[0]//2 - jj,:,:] = self._parent.data._data[self._parent.data.shape[0]//2 + closest_offset,:,:]

        self._parent.meta.history.add_record('Quadrant shift', 1)
        self._parent.message('Medipix quadrant shift applied.')

    def flat_field(self, kind=''):
        '''
        Apply flat field correction.
        '''

        if (str.lower(kind) == 'skyscan'):
            if self._parent.meta.geometry.roi_fov:
                self._parent.message('Object is larger than the FOV!')
                
                air_values = numpy.ones_like(self._parent.data._data[:,:,0]) * 2**16 - 1
            else:
                air_values = numpy.max(self._parent.data._data, axis = 2)
                
            air_values = air_values.reshape((air_values.shape[0],air_values.shape[1],1))
            self._parent.data._data = self._parent.data._data / air_values
            
            # add a record to the history: 
            self._parent.meta.history.add_record('process.flat_field', 1)

            self._parent.message('Skyscan flat field correction applied.')

        else:
            if numpy.min(self._parent.data._ref) <= 0:
                self._parent.warning('Flat field reference image contains zero (or negative) values! Will replace those with little tiny numbers.')

                tiny = self._parent.data._ref[self._parent.data._ref > 0].min()
                self._parent.data._ref[self._parent.data._ref <= 0] = tiny

            # How many projections:
            n_proj = self._parent.data.shape[1]

            if not self._parent.data._dark is None:
                for ii in range(0, n_proj):
                    self._parent.data._data[:, ii, :] = (self._parent.data._data[:, ii, :] - self._parent.data._dark) / (self._parent.data._ref - self._parent.data._dark)
            else:
                for ii in range(0, n_proj):
                    self._parent.data._data[:, ii, :] = (self._parent.data._data[:, ii, :]) / (self._parent.data._ref)

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

        weights = numpy.zeros_like(self._parent.data._data, dtype=numpy.float32)
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

        self._parent.data._data *= weights
        # add a record to the history:
        self._parent.meta.history.add_record('process.short_scan', 1)

        self._parent.message('Short scan correction applied.')


    def log(self, air_intensity = 1.0, lower_bound = -10, upper_bound = numpy.log(256)):
        '''
        Apply -log(x) to the sinogram
        '''
        # Check if the log was already applied:
        #self._parent._check_double_hist('process.log(upper_bound)')

        # If not, apply!
        if (air_intensity != 1.0):
            self._parent.data._data /= air_intensity
            
        # In-place negative logarithm
        numpy.log(self._parent.data._data, out = self._parent.data._data)
        numpy.negative(self._parent.data._data, out = self._parent.data._data)
        self._parent.data._data = numpy.float32(self._parent.data._data)
        
        # Apply a bound to large values:
        numpy.clip(self._parent.data._data, a_min = lower_bound, a_max = upper_bound, out = self._parent.data._data)

        self._parent.message('Logarithm is applied.')
        self._parent.meta.history.add_record('process.log(upper_bound)', upper_bound)

    def salt_pepper(self, kernel = 3):
        '''
        Gets rid of nasty speakles
        '''
        # Make a smooth version of the data and look for outlayers:
        smooth = ndimage.filters.median_filter(self._parent.data._data, [kernel, 1, kernel])
        mask = self._parent.data._data / smooth
        mask = (numpy.abs(mask) > 1.5) | (numpy.abs(mask) < 0.75)

        self._parent.data._data[mask] = smooth[mask]

        self._parent.message('Salt and pepper filter is applied.')

        self._parent.meta.history.add_record('process.salt_pepper(kernel)', kernel)

    def simple_tilt(self, tilt):
        '''
        Tilts the sinogram
        '''
        for ii in range(0, self._parent.data.shape[1]):
            self._parent.data._data[:, ii, :] = interp.rotate(numpy.squeeze(self._parent.data._data[:, ii, :]), -tilt, reshape=False)
            
        self._parent.message('Tilt is applied.')
        
    def center_shift(self, offset=None, test_path=None, ind=None, cen_range=None):
        '''
        Find the center of the sinogram and apply the shift to corect for the offset
        '''
        sz = self._parent.data.shape[2] // 2

        if test_path is not None:
            rotation.write_center(self._parent.data._data, theta=self._parent.meta.geometry.thetas, dpath=test_path, ind=ind, cen_range=cen_range, sinogram_order=True)

        else:
            if offset is None:
                offset = sz-rotation.find_center(self._parent.data._data, self._parent.meta.geometry.thetas,  sinogram_order=True)[0]
                #offset = sz-rotation.find_center_pc(self._parent.data._data[:,0,:], self._parent.data._data[:,self._parent.data.shape(1)//2,:])p

            else:
                # Do nothing is the offset is less than 1 pixel
                if (numpy.abs(offset) >= 1):
                    self._parent.data._data = interp.shift(self._parent.data._data, (0,0,offset))
                    self._parent.meta.history.add_record('process.center_shift(offset)', offset)

                    self._parent.message('Horizontal offset corrected.')
                else:
                    self._parent.warning("Center shift found an offset smaller than 1. Correction won't be applied")

    def bin_theta(self):
        '''
        Bin angles with a factor of two
        '''
        self._parent.data._data = (self._parent.data._data[:,0:-1:2,:] + self._parent.data._data[:,1::2,:]) / 2
        self._parent.meta.geometry.thetas = numpy.array(self._parent.meta.geometry.thetas[0:-1:2] + self._parent.meta.geometry.thetas[1::2]) / 2

    def bin_data(self):
        '''
        Bin data with a factor of two
        '''
        self._parent.data._data = (self._parent.data._data[:, :, 0:-1:2] + self._parent.data._data[:, :, 1::2]) / 2
        self._parent.data._data = (self._parent.data._data[0:-1:2, :, :] + self._parent.data._data[1::2, :, :]) / 2

        self._parent.meta.geometry.det_pixel *= 2


    def crop(self, top_left, bottom_right):
        '''
        Crop the sinogram
        '''
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

# **************************************************************
#           RECONSTRUCTION class and subclasses
# **************************************************************
import astra
from scipy import interpolate
import math
import odl

class reconstruct(subclass):
    '''
    Reconstruction algorithms: FDK, SIRT, KL, FISTA etc.
    '''
    _projection_mask = None
    _reconstruction_mask = None
    _projection_filter = None
    
    _display_callback = False
    _auto_adjust_volume = True
    
    def __init__(self, proj = []):
        subclass.__init__(self, proj)
        
        self.vol_geom = None
        self.proj_geom = None
        self.geom_modifier = {'det_tra_vrt': 0, 'det_tra_hrz':0, 'src_tra_vrt':0, 'src_tra_hrz':0} 
        
    def initialize_projection_mask(self, weight_poisson = False, weight_histogram = None, pixel_mask = None):
        '''
        Genrate weights proportional to the square root of intensity that map onto the projection data

        weight_poisson  -  weight rays according to the square root of the normalized intensity

        weight_histogram  -  weight intensities according to a predifined hystogram, defined as (x, y),
        where x is intensity value and y is the corresponding weight

        pixel_mask  -  assign different weights depending on the pixel location

        '''
        prnt = self._parent

        # Initialize ASTRA:
        self._initialize_astra()

        # Create a volume containing only ones for forward projection weights
        sz = self._parent.data.shape

        self._projection_mask = prnt.data._data * 0

        # if weight_poisson: introduce weights based on the value of intensity image:
        if not weight_poisson is None:
            self._projection_mask = self._projection_mask * numpy.sqrt(numpy.exp(-prnt.data._data))

        # if weight_histogram is provided:
        if not weight_histogram is None:
            x = weight_histogram[0]
            y = weight_histogram[1]
            f = interpolate.interp1d(x, y, kind = 'linear', fill_value = 'extrapolate')

            self._projection_mask = self._projection_mask * f(numpy.exp(-prnt.data._data))

        # apply pixel mask to every projection if it is provided:
        if not pixel_mask is None:
            for ii in range(0, sz[1]):
                self._projection_mask = self._projection_mask[:, ii, :] * pixel_mask

        prnt.message('Projection mask is initialized')

    def initialize_reconstruction_mask(self):
        '''
        Make volume mask to avoid projecting errors into the corners of the volume
        '''
        prnt = self._parent

        sz = prnt.data.shape()[2]

        # compute radius of the defined cylinder
        det_width = prnt.data.shape()[2] / 2
        src2obj = prnt.meta.geometry.src2obj
        total = prnt.meta.geometry.src2det
        pixel = prnt.meta.geometry.det_pixel

        # Compute the smallest radius and cut the cornenrs:
        radius = 2 * det_width * src2obj / numpy.sqrt(total**2 + (det_width*pixel)**2)

        # Create 2D mask:
        yy,xx = numpy.ogrid[-sz//2:sz//2, -sz//2:sz//2]

        self._reconstruction_mask = numpy.array(xx**2 + yy**2 <= radius**2, dtype = 'float32')

        # Replicate to 3D:
        self._reconstruction_mask = numpy.ascontiguousarray((numpy.tile(self._reconstruction_mask[None, :,:], [astra.functions.geom_size(self.vol_geom)[0], 1, 1])))

        prnt.message('Reconstruction mask is initialized')


    def project_thetas(self, parameter_value = 0, parameter = 'axis_offset'):
        '''
        This routine produces a single slice with a sigle ray projected into it from each projection angle.
        Can be used as a simple diagnostics for angle coverage.
        '''
        prnt = self._parent
        sz = prnt.data.shape()

        # Make a synthetic sinogram:
        sinogram = numpy.zeroes((1, sz[1], sz[2]))

        # For compatibility purposes make sure that the result is 3D:
        sinogram = numpy.ascontiguousarray(sinogram)

        # Initialize ASTRA:
        sz = numpy.array(prnt.data.shape())
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
        vol = self._backproject(prnt.data._data, algorithm='FDK_CUDA', short_scan=short_scan)

        return volume(vol)
        # No need to make a history record - sinogram is not changed.



    def slice_FDK(self, parameter_value = 0, parameter = 'axis_offset'):
        '''
        A quick calculation of a single central slice.
        Returns a numpy array and not a volume object!
        '''

        prnt = self._parent

        # Extract 1 pixel thin slice:
        sinogram = prnt.data._data[prnt.data.shape(0)//2, :, :]
        #sinogram = prnt.data._data

        # For compatibility purposes make sure that the result is 3D:
        sinogram = numpy.ascontiguousarray(sinogram[None, :])

        # Initialize ASTRA:
        sz = numpy.array(prnt.data.shape())
        pixel_size = prnt.meta.geometry.det_pixel
        det2obj = prnt.meta.geometry.det2obj
        src2obj = prnt.meta.geometry.src2obj
        theta = prnt.meta.geometry.thetas

        # Temporary change of one of the parameters:
        if abs(parameter_value) >0:

            if parameter == 'axis_offset':
                # Apply shift:
                sinogram = interp.shift(sinogram, (0,0, parameter_value))
            elif parameter == 'det_pixel':
                pixel_size = parameter_value

            elif parameter == 'det2obj':
                det2obj = parameter_value

            elif parameter == 'src2obj':
                src2obj = parameter_value

            else: prnt.error("Can't recognize given parameter.")

        sz[0] = 1
        self._initialize_astra(sz, pixel_size, det2obj, src2obj, theta)

        short_scan = (theta.max() - theta.min()) < (numpy.pi * 1.99)

        vol = self._backproject(sinogram, algorithm='FDK_CUDA', short_scan=short_scan)

        return vol
        # No need to make a history record - sinogram is not changed.

    def slice_scan(self, scan_range = numpy.linspace(-10, 10, 11), parameter = 'axis_offset'):
        '''
        Create a scan of different rotation axis offsets:
        '''
        sz = self._parent.data.shape

        print('Starting an', parameter, ' scan.')

        # Create a volume to put a scan into:
        vol = numpy.zeros([scan_range.shape[0], sz[2], sz[2]])

        ii = 0
        for val in scan_range:
            print(val)

            img = self.slice_FDK(val, parameter)
            vol[ii, :, :] = img
            ii += 1

        return volume(vol)

    def _initialize_odl(self):
        '''
        Initialize da RayTransform!
        '''
        
        prnt = self._parent

        sz = self._parent.data.shape
        geom = prnt.meta.geometry

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
            l2_norm = (odl.solvers.L1Norm(ray_trafo.range)).translated(numpy.transpose(self._parent.data._data, axes = [1, 2, 0]))
        else:
            l2_norm = (odl.solvers.L2NormSquared(ray_trafo.range)).translated(numpy.transpose(self._parent.data._data, axes = [1, 2, 0]))
        
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
        x = fbp(numpy.transpose(self._parent.data._data, axes = [1, 2, 0]))
        
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
        odl.solvers.mlem(ray_trafo, x, numpy.transpose(self._parent.data._data, axes = [1, 2, 0]), 
                                niter = iterations, callback = callback)
        
        return volume(numpy.transpose(x.asarray(), axes = [2, 0, 1])[:,::-1,:])    

    def FDK(self, short_scan = None):
        '''

        '''
        prnt = self._parent

        # Initialize ASTRA:
        self._initialize_astra()

        # Run the reconstruction:
        #epsilon = numpy.pi / 180.0 # 1 degree - I deleted a part of code here by accident...
        theta = self._parent.meta.geometry.thetas
        if short_scan is None:
            short_scan = (theta.max() - theta.min()) < (numpy.pi * 1.99)
        
        print('Short scan: ' + str(short_scan))
        vol = self._backproject(prnt.data._data, algorithm='FDK_CUDA', short_scan = short_scan)

        vol = volume(vol)
        #vol.history['FDK'] = 'generated in '
        self._parent.meta.history.add_record('set data.data', [])
        return vol
        # No need to make a history record - sinogram is not changed.


    def SIRT(self, iterations = 10, min_constraint = None):
        '''
        '''
        prnt = self._parent

        # Initialize ASTRA:
        self._initialize_astra()

        # Run the reconstruction:
        vol = self._backproject(prnt.data._data, algorithm = 'SIRT3D_CUDA', iterations = iterations, min_constraint= min_constraint)

        return volume(vol)
        # No need to make a history record - sinogram is not changed.


    def SIRT_CPU(self, iterations = 10, min_constraint = None):
        '''

        '''
        prnt = self._parent
        # Initialize ASTRA:
        self._initialize_astra()

        # Create a volume containing only ones for forward projection weights
        sz = self._parent.data.shape
        theta = self._parent.meta.geometry.thetas

        # Initialize weights:
        vol_ones = numpy.ones((sz[0], sz[2], sz[2]), dtype=numpy.float32)
        vol = numpy.zeros_like(vol_ones, dtype=numpy.float32)
        weights = self._forwardproject(vol_ones)
        weights = 1.0 / (weights + (weights == 0))

        bwd_weights = 1.0 / (theta.shape[0])

        vol = numpy.zeros((sz[0], sz[2], sz[2]), dtype=numpy.float32)

        for ii_iter in range(iterations):
            fwd_proj_vols = self._forwardproject(vol)

            residual = (prnt.data._data - fwd_proj_vols) * weights

            if not self._projection_mask is None:
                residual *= self._projection_mask

            vol += bwd_weights * self._backproject(residual, algorithm='BP3D_CUDA')

            if min_constraint != None:
                vol[vol < min_constraint] = min_constraint


        return volume(vol)
        # No need to make a history record - sinogram is not changed.

    def CPLS(self, iterations = 10, min_constraint = None):
        '''
        Chambolle-Pock Least Squares
        '''
        prnt = self._parent

        # Initialize ASTRA:
        self._initialize_astra()

        # Create a volume containing only ones for forward projection weights
        sz = self._parent.data.shape()
        theta = self._parent.meta.geometry.thetas

        vol_ones = numpy.ones((sz[0], sz[2], sz[2]), dtype=numpy.float32)
        vol = numpy.zeros_like(vol_ones, dtype=numpy.float32)
        theta = self._parent.meta.geometry.thetas
        sigma = self._forwardproject(vol_ones)
        sigma = 1.0 / (sigma + (sigma == 0))
        sigma_1 = 1.0  / (1.0 + sigma)
        tau = 1.0 / theta.shape[0]

        p = numpy.zeros_like(prnt.data._data)
        ehn_sol = vol.copy()

        for ii_iter in range(iterations):
            p = (p + prnt.data._data - self._forwardproject(ehn_sol) * sigma) * sigma_1

            old_vol = vol.copy()
            vol += self._backproject(p, algorithm='BP3D_CUDA', min_constraint=min_constraint) * tau
            vol *= (vol > 0)

            ehn_sol = vol + (vol - old_vol)
            gc.collect()

        return volume(vol)
        # No need to make a history record - sinogram is not changed.



    def CGLS(self, iterations = 10, min_constraint = None):
        '''
        
        '''
        prnt = self._parent

        # Initialize ASTRA:
        self._initialize_astra()

        # Run the reconstruction:
        vol = self._backproject(prnt.data._data, algorithm = 'CGLS3D_CUDA', iterations = iterations, min_constraint=min_constraint)


        return volume(vol)
        # No need to make a history record - sinogram is not changed.        
    
    def _apply_modifiers(self):
        '''
        Apply arbitrary geometrical modifiers to the ASTRA projection geometry vector
        ''' 
        if (self.proj_geom['type'] == 'cone'):
            self.proj_geom = astra.functions.geom_2vec(self.proj_geom)
            
        vectors = self.proj_geom['Vectors']

        for ii in range(0, vectors.shape[0]):
            
            # Define vectors:
            src_vect = vectors[ii, 0:3]    
            det_vect = vectors[ii, 3:6]    
            det_axis_hrz = vectors[ii, 6:9]           
            det_axis_vrt = vectors[ii, 9:12]           

            #Precalculate vector perpendicular to the detector plane:
            det_normal = numpy.cross(det_axis_hrz, det_axis_vrt)
            det_normal = det_normal / numpy.sqrt(numpy.dot(det_normal, det_normal))
            
            geom = self._parent.meta.geometry
            
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
    
            
    def _initialize_astra(self, sz = None, det_pixel = None, 
                          det2obj = None, src2obj = None, theta = None, vec_geom = False):

        if sz is None: sz = self._parent.data.shape
        if det_pixel is None: det_pixel = self._parent.meta.geometry.det_pixel
        if det2obj is None: det2obj = self._parent.meta.geometry.det2obj
        if src2obj is None: src2obj = self._parent.meta.geometry.src2obj
        if theta is None: theta = self._parent.meta.geometry.thetas

        # Initialize ASTRA (3D):
        det_count_x = sz[2]
        det_count_z = sz[0]
        
        # Make volume count x > detector count to include corneres of the object:
        vol_count_x = sz[2]
        vol_count_z = sz[0]

        M = self._parent.meta.geometry.magnification

        self.vol_geom = astra.create_vol_geom(vol_count_x, vol_count_x, vol_count_z)
            
        self.proj_geom = astra.create_proj_geom('cone', M, M, det_count_z, det_count_x, theta, (src2obj*M)/det_pixel[1], (det2obj*M)/det_pixel[0])
        
        self._apply_modifiers()

    def _initialize_ramp_filter(self, power = 1):
      sz = self._parent.data.shape

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
      #filt = filt[0:order]

      # Back to 32 bit...
      filt = numpy.float32(filt)

      self._projection_filter = numpy.matlib.repmat(filt, sz[1], 1)
      #def _geometry_modifiers(self, sx = 0, sy = 09, sz, )


    def _backproject(self, y, algorithm = 'FDK_CUDA', iterations=1, min_constraint = None, short_scan=False):

      cfg = astra.astra_dict(algorithm)
      cfg['option'] = {}
      if short_scan:
          cfg['option']['ShortScan'] = True
      print(cfg)
      if (min_constraint is not None):
          cfg['option']['MinConstraint'] = min_constraint
      
      output = numpy.zeros(astra.functions.geom_size(self.vol_geom), dtype=numpy.float32)
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
              mask_id = astra.data3d.link('-vol', self.vol_geom, self._reconstruction_mask)
              cfg['option']['ReconstructionMaskId'] = mask_id
    
          if not self._projection_mask is None:
              mask_id = astra.data3d.link('-sino', self.proj_geom, self._projection_mask)
              cfg['option']['SinogramMaskId'] = mask_id
    
          # Use modified filter:
          if not self._projection_filter is None:
    
              sz = self._projection_filter.shape
    
              slice_proj_geom = astra.create_proj_geom('parallel', 1.0, sz[1], self._parent.meta.geometry.thetas)
    
              filt_id = astra.data2d.link('-sino', slice_proj_geom, self._projection_filter)
              cfg['option']['FilterSinogramId'] = filt_id
              
          alg_id = astra.algorithm.create(cfg)
          astra.algorithm.run(alg_id, iterations)

      finally:
          astra.algorithm.delete(alg_id)
          astra.data3d.delete([rec_id, sinogram_id])

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
          astra.data3d.delete([rec_id, sinogram_id])
          astra.algorithm.delete(alg_id)

      return output


    def get_vol_ROI(self):
        # Computes a mask of minimal projection ROI needed to reconstruct a ROI for FDK
        prnt = self._parent

        # Initialize ASTRA:
        self._initialize_astra()

        # Run the reconstruction:
        vol = self._backproject(numpy.ones(prnt.data._data.shape, dtype = 'float32'))

        return volume(vol)

    def get_proj_ROI(self, rows=[0,512], cols=[0,512], algorithm='FP3D_CUDA'):
        # Computes a mask of minimal projection ROI needed to reconstruct a ROI for FDK
        prnt = self._parent

        # Initialize ASTRA:
        sz = prnt.data.shape()
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
from scipy import ndimage
from skimage import morphology

class postprocess(subclass):
    '''
    Includes postprocessing of the reconstructed volume.
    '''

    def threshold(self, volume, threshold = None):

        if threshold is None: threshold = volume.analyse.max() / 2

        volume.data.data = ((volume.data.data > threshold) * 1.0)

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

class volume(object):
    data = []
    io = []
    analyse = []
    display = []
    meta = []
    postprocess = []

    def __init__(self, vol = []):
        self.io = io(self)
        self.display = display(self)
        self.analyse = analyse(self)
        self.data = data(self)
        self.meta = meta(self)
        self.postprocess = postprocess(self)

        # Get the data in:
        self.data._data = vol

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

    def __init__(self):
        self.io = io(self)
        self.meta = meta(self)
        self.display = display(self)
        self.analyse = analyse(self)
        self.process = process(self)
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
        Deep copy of the sinogram object:
        '''
        return copy.deepcopy(self)

    def _pronounce_wisdom(self):

        randomator = 0
        # Beef up the randomator:
        for ii in range(0, self._wisdom_status):
            randomator += numpy.random.randint(0, 100)

        # If randomator is small, utter a wisdom!
        if (randomator < 50):
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
            self.message((k in self.meta.history.keys))

            if not(k in self.meta.history.key):
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
