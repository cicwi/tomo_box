#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:39:33 2017

@author: kostenko & der sarkissian

 ***********    Pilot for the new tomobox  *************

"""


#%% Initialization

import matplotlib.pyplot as plt

#%%

from scipy import misc  # Reading BMPs
import os
import numpy
import re
#import pkg_resources
import time
#from mayavi import mlab
#from tvtk.util.ctf import ColorTransferFunction


# **************************************************************
#           Parent class for all sinogram subclasses:
# **************************************************************

class subclass(object):
    def __init__(self, parent):
        self._parent = parent

# **************************************************************
#           IO class and subroutines
# **************************************************************
from stat import S_ISREG, ST_CTIME, ST_MODE
#import os
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

        # If it's a tiff, use dxchange to read tiff:
        #if ((filename[-4:] == 'tiff') | (filename[-3:] == 'tif')):

        #    print('Reading a tiff stack')
        #    if self._parent:
        #        self._parent.data._data = dxchange.reader.read_tiff_stack(os.path.join(path,filename), range(first, last + 1), digit=4)
        #    else:
        #        return dxchange.reader.read_tiff_stack(os.path.join(path,filename), range(first, last + 1))
        #else:

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

        # Cast to float to avoid problems with divisions in the future:
       # self._parent.data._data = numpy.float32(self._parent.data._data, copy=False)

        # add record to the history:
        self._parent.meta.history['io.read_raw'] = path

    def read_ref(self, path_file):
        '''
        Read reference flat field.

        '''
        #if ((os.path.splitext(path_file)[1] == '.tiff') or (os.path.splitext(path_file)[1] == '.tif')):
        #    ref = numpy.array(dxchange.reader.read_tiff(path_file))
        #else:
        ref  = misc.imread(path_file, flatten= 0)

        if self._parent:
            self._parent.data._ref = ref

        # Cast to float to avoid problems with divisions in the future:
        self._parent.data._ref = numpy.float32(self._parent.data._ref)

        # add record to the history:
        self._parent.meta.history['io.read_ref'] = path_file

        self._parent.message('Flat field reference image loaded.')

    def read_bh(self, path_file):
        '''
        Read reference foil data for signal to equivalent thickness calibration.

        '''
        #if ((path_file[-4:] == 'tiff') | (path_file[-3:] == 'tif')):
        #    ref = numpy.array(dxchange.reader.read_tiff(path_file))
        #else:
        ref  = misc.imread(path_file, flatten= 0)

        if self._parent:
            self._parent.data._ref = ref

        # Cast to float to avoid problems with divisions in the future:
        self._parent.data._ref = numpy.float32(self._parent.data._ref)

        # add record to the history:
        self._parent.meta.history['io.read_ref'] = path_file

        self._parent.message('Beam hardening correction reference images loaded.')

    def save_backup(self):
        '''
        Make a copy of data in memory, just in case.
        '''
        self._parent.data._backup = self._parent.data._data.copy()

        # add record to the history:
        self._parent.meta.history['io.save_backup'] = 'backup saved'

        self._parent.message('Backup saved.')

    def load_backup(self):
        '''
        Retrieve a copy of data from the backup.
        '''
        if self._parent.data._backup == []:
            self._parent.error('No backup found in memory!')

        self._parent.data._data = self._parent.data._backup.copy()

        # Clean memory:
        self._parent.data._backup = None
        gc.collect()

        # add record to the history:
        self._parent.meta.history['io.load_backup'] = 'backup loaded'

        self._parent.message('Backup loaded.')

    def read_meta(self, path = '', kind=None):
        '''
        Parser for the metadata file that contains information about the acquisition system.
        '''
        path = update_path(path, self)
        geometry = self._parent.meta.geometry

        # TODO: make the actual parser. For now just initialization with default

        if kind is None:
            geometry['det_pixel'] = 0.055
            geometry['src2obj'] = 210.0
            geometry['det2obj'] = 50.0
            #geometry['src2det'] = 209.610
            geometry['rot_step'] = 2*numpy.pi / (geometry['nb_angle']-1)

        elif (str.lower(kind) == 'skyscan'):
            # Parse the SkyScan log file
            self._parse_skyscan_meta(path)

            if (geometry['det2obj'] == 0.0):
                geometry['det2obj'] = geometry['src2det'] - geometry['src2obj']

        elif (str.lower(kind) == 'flexray'):
            # Parse the SkyScan log file
            self._parse_flexray_meta(path)

        elif (str.lower(kind) == 'asi'):
            # Parse the ASI log file
            self._parse_asi_meta(path)
            
        # Check geometry consistency:    
        self._geometry_consistency()

        # add record to the history:
        self._parent.meta.history['io.read_meta'] = path

        self._parent.message('Meta data loaded.')
    
    def _geometry_consistency(self):
        '''
        After parsing the meta data, check if your geometry is consistent and fill in gaps if needed.
        '''
        geometry = self._parent.meta.geometry
        
        # Check if the number of angles and the first and the last angle are present
        if not 'nb_angle' in geometry.keys():
            self._parent.warning('Number of angles in not found by parser. Will use the raw data shape instead.')
            geometry['nb_angle'] = self._parent.data.shape(1)

        if not 'first_angle' in geometry.keys():
            self._parent.warning('Assuming that the first rotation angle is zero.')
            geometry['first_angle'] = 0
            
        if not 'last_angle' in geometry.keys():
            if (not 'rot_step' in geometry.keys()) or ('rot_step' in geometry.keys() and geometry['rot_step'] == 0.0):
                self._parent.warning('Assuming that the last rotation angle is 360 degrees and rotation step is adjusted accordingly to the number of projections')
                geometry['last_angle'] = 2 * numpy.pi
                geometry['rot_step'] = (geometry['last_angle'] - geometry['first_angle']) / (geometry['nb_angle'] - 1)
            else:
                geometry['last_angle'] = geometry['first_angle'] + geometry['rot_step'] * (geometry['nb_angle'] - 1)
            
        if (not 'det2obj' in geometry.keys()) or (geometry['det2obj'] == 0.0):
            geometry['det2obj'] = geometry['src2det'] - geometry['src2obj']

        if (not 'src2obj' in geometry.keys()) or (geometry['src2obj'] == 0.0):
            geometry['src2obj'] = geometry['src2det'] - geometry['det2obj']

        if (not 'src2det' in geometry.keys()) or (geometry['src2det'] == 0.0):
            geometry['src2det'] = geometry['det2obj'] + geometry['src2obj']

        if not 'det_pixel' in geometry.keys() or geometry['det_pixel'] == 0:
            self._parent.warning("Parser didn't find det_pixel; will use img_pixel to derive det_pixel")
            geometry['det_pixel'] = geometry['img_pixel'] * geometry['src2det'] / geometry['src2obj']


        la = geometry['last_angle']
        fa = geometry['first_angle']
           
        # Initialize thetas:   
        self._parent.meta.theta = numpy.linspace(fa, la, geometry['nb_angle'])
    
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
            #raise UserWarning('Found several log files. Currently using: ' + log_file[0])
            self._parent.warning('Found several log files. Currently using: ' + log_file[0])
            log_file = os.path.join(path, log_file[0])
        else:
            log_file = os.path.join(path, log_file[0])

        #Once the file is found, parse it
        geometry = self._parent.meta.geometry
        #physics = self._parent.meta.physics

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
                    geometry[geom_key[0]] = float(var)*factor

        # Convert microns to mm:
        geometry['det_pixel'] = geometry['det_pixel'] / 1e3

        # Fill in some gaps:
        geometry['src2det'] = geometry['src2obj'] + geometry['det2obj']
        geometry['img_pixel'] = geometry['det_pixel'] / geometry['src2det'] * geometry['src2obj']

        geometry['first_angle'] = 0
        geometry['last_angle'] = 2*numpy.pi

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

        #Once the file is found, parse it
        geometry = self._parent.meta.geometry
        #physics = self._parent.meta.physics

        # Create a dictionary of keywords (skyscan -> our geometry definition):
        geom_dict = {'voxel size':'img_pixel', 'sod':'src2obj', 'sdd':'src2det', '# projections':'nb_angle',
                     'last angle':'last_angle', 'start angle':'first_angle', 'tube voltage':'voltage', 'tube power':'power', 'Exposure time (ms)':'exposure'}

        with open(log_file, 'r') as logfile:
            for line in logfile:
                name, var = line.partition(":")[::2]
                name = name.strip().lower()

                # If name contains one of the keys (names can contain other stuff like units):
                geom_key = [geom_dict[key] for key in geom_dict.keys() if key in name]

                if geom_key != []:
                    factor = self._parse_unit(name)
                    geometry[geom_key[0]] = float(var)*factor

        # Convert microns to mm:
        geometry['img_pixel'] = geometry['img_pixel'] * self._parse_unit('um')

        # Convert degrees to radian:
        geometry['last_angle'] = geometry['last_angle'] * self._parse_unit('deg')
        geometry['first_angle'] = geometry['first_angle'] * self._parse_unit('deg')

    def _parse_unit(self, string):
            # Look at the inside of trailing parenthesis
            unit = ''
            factor = 1.0
            if string[-1] == ')':
                unit = string[string.rfind('(')+1:-1].strip().lower()
            else:
                unit = string.strip().lower()

            # Metric units --> mm
            if unit == 'mm':
                pass
            elif unit == 'um':
                factor = 0.001
            elif unit == 'cm':
                factor = 10.0
            elif unit == 'm':
                factor = 1000.0

            # Angles --> rad
            elif unit == 'rad':
                pass
            elif unit == 'deg':
                factor = numpy.pi / 180.0

            # Time --> ms
            elif unit == 'ms':
                pass
            elif unit == 's':
                factor = 1000.0
            elif unit == 'us':
                factor = 0.001

            # Energy --> keV
            elif unit == 'kev':
                pass
            elif unit == 'mev':
                factor = 1000.0
            elif unit == 'ev':
                factor = 0.001

            # Voltage --> kV
            elif unit == 'kv':
                pass
            elif unit == 'mv':
                factor = 1000.0
            elif unit == 'v':
                factor = 0.001

            # Current --> uA
            elif unit == 'ua':
                pass
            elif unit == 'ma':
                factor = 1000.0
            elif unit == 'a':
                factor = 1000000.0
            
            # Dimensionless units: pass
            elif unit == 'line':
                pass
            
            else:
                self._parent.warning('Unknown unit: ' + unit + '. Skipping.')

            return factor
        
    
    def read_skyscan_thermalshifts(self, path = ''):
        path = update_path(path,self)

        # Try to find the log file in the selected path
        fname = [x for x in os.listdir(path) if (os.path.isfile(os.path.join(path, x)) and os.path.splitext(os.path.join(path, x))[1] == '.csv')]
        if len(fname) == 0:
            self._parent.warning('XY shifts csv file not found in path: ' + path)
            return
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
                #self._parent.meta.geometry['thermal_shifts'].append([row['x'], row['y']])
                shifts.append([float(row['x']), float(row['y'])])
        
            self._parent.meta.geometry['thermal_shifts'] = numpy.array(shifts)
            
            

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
        geometry = self._parent.meta.geometry
        #physics = self._parent.meta.physics

        # Create a dictionary of keywords (skyscan -> our geometry definition):
        geom_dict = {'camera pixel size': 'det_pixel', 'image pixel size': 'img_pixel', 'object to source':'src2obj', 'camera to source':'src2det', 
        'optical axis':'optical_axis', 'rotation step':'rot_step', 'exposure':'exposure', 'source voltage':'voltage', 'source current':'current',
        'camera binning':'det_binning', 'image rotation':'det_tilt', 'number of rows':'det_rows', 'number of columns':'det_cols', 'postalignment':'det_offset', 'object bigger than fov':'object_bigger_than_fov'}
        
        
        
        with open(log_file, 'r') as logfile:
            for line in logfile:
                name, val = line.partition("=")[::2]
                name = name.strip().lower()
                
                # If name contains one of the keys (names can contain other stuff like units):
                geom_key = [geom_dict[key] for key in geom_dict.keys() if key in name]
                
                if geom_key != [] and (geom_key[0] != 'det_binning') and (geom_key[0] != 'det_offset') and (geom_key[0] != 'large_object') and (geom_key[0] != 'object_bigger_than_fov'):
                    factor = self._parse_unit(name)
                    geometry[geom_key[0]] = float(val)*factor
                elif geom_key != [] and geom_key[0] == 'det_binning':
                    # Parse with the 'x' separator
                    bin_x, bin_y = val.partition("x")[::2]
                    geometry[geom_key[0]] = [float(bin_x), float(bin_y)]
                elif geom_key != [] and geom_key[0] == 'det_offset':
                    geometry[geom_key[0]][0] = -float(val)
                elif geom_key != [] and geom_key[0] == 'object_bigger_than_fov':
                    if val.strip().lower() == 'off':
                        geometry[geom_key[0]] = False
                    else:
                        geometry[geom_key[0]] = True
                    
        # Convert optical axis into detector offset (skyscan measures lines from the bottom)
        if 'optical_axis' in geometry:
            geometry['det_offset'][1] = geometry['optical_axis'] - geometry['det_rows']/2.0
        
        
        # Convert detector tilt into radian units (degrees assumed)
        if 'det_tilt' in geometry:
            geometry['det_tilt'] = geometry['det_tilt'] * self._parse_unit('deg')
            
    
    
    
    def save(self, path = '', fname='data', fmt = 'tiff', slc_range = None, window = None, digits = 4, dtype = None):
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
            
            if slc_range is None:
                slc_range = range(0,self._parent.data._data.shape[0])
            
            if dtype is None:
                dtype = self._parent.data._data.dtype
            
            maxi = numpy.max(self._parent.data._data)
            mini = numpy.min(self._parent.data._data)
            
            if (not (window is None)):
                maxi = numpy.min([maxi, window[1]])
                mini = numpy.max([mini, window[0]])
            
            for i in slc_range:
                # Fix the file name
                fname_tmp = fname
                fname_tmp += '_'
                fname_tmp += str(i).zfill(digits)
                fname_tmp += '.' + fmt
                
                # Fix the windowing and output type
                slc = numpy.array(self._parent.data._data[i,:,:], dtype = numpy.float32)
                
                if (not (window is None)):
                    numpy.clip(a = slc, a_min = window[0], a_max = window[1], out = slc)
                
                # Rescale if integer type
                if not (numpy.issubdtype(dtype, numpy.floating)):
                    slc -= mini
                    if (maxi != mini):
                        slc /= (maxi - mini)
                    slc *= numpy.iinfo(dtype).max
                
                # Save the image
                im = Image.fromarray(numpy.asarray(slc, dtype=dtype))
                im.save(fname_tmp)
                
                
    
    def save_tiff(self, path = '', fname='data', axis = 0, digits = 4):
        '''
        Saves the data to tiff files
        '''
        from PIL import Image
        if self._parent.data._data is not None:
        # First check if digit is large enough, otherwise add a digit
            im_nb = self._parent.data._data.shape[axis]
            if digits <= numpy.log10(im_nb):
                digits = int(numpy.log10(im_nb)) + 1
                
            path = update_path(path, self)
            fname = os.path.join(path, fname)
            
            for i in range(0,self._parent.data._data.shape[axis]):
                fname_tmp = fname
                fname_tmp += '_'
                fname_tmp += str(i).zfill(digits)
                fname_tmp += '.tiff'
                im = Image.fromarray(self._parent.data._data[i,:,:])
                im.save(fname_tmp)
                #misc.imsave(name = os.path.join(path, fname_tmp), arr = self._parent.data._data[i,:,:])
        #dxchange.writer.write_tiff_stack(self._parent.data.get_data(),fname=os.path.join(path, fname), axis=axis,overwrite=True)

# **************************************************************
#           META class and subclasses
# **************************************************************

class meta(subclass):
    '''
    This object contains various properties of the imaging system and the history of pre-processing.
    '''
    geometry = {'src2obj': 0, 'det2obj': 0, 'det_pixel': 0, 'det_offset': [0,0], 'det_tilt': 0, 'det_binning': [1, 1]}
    theta =  numpy.linspace(0, 2*numpy.pi, 540)

    physics = {'voltage': 0, 'current':0, 'exposure': 0}
    history = {'object initialized': time.asctime()}

# **************************************************************
#           DISPLAY class and subclasses
# **************************************************************
class display(subclass):
    '''
    This is a collection of display tools for the raw and reconstructed data
    '''
    def __init__(self, parent = []):
        self._parent = parent
        self._cmap = 'gray'

    def _figure_maker_(self, fig_num):
        '''
        Make a new figure or use old one.
        '''
        if fig_num:
            plt.figure(fig_num)
        else:
            plt.figure()


    def slice(self, slice_num, dim_num, fig_num = [], mirror = False, upsidedown = False):
        '''
        Display a 2D slice of 3D volumel
        '''
        self._figure_maker_(fig_num)

        img = extract_2d_array(dim_num, slice_num, self._parent.data.get_data())

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
        img = extract_2d_array(dim_num, slice_num, self._parent.data.get_data())
        fig = plt.imshow(img, cmap = self._cmap)

        plt.colorbar()
        plt.show()

        for slice_num in range(1, self._parent.data.shape()[dim_num]):
            img = extract_2d_array(dim_num, slice_num, self._parent.data.get_data())
            fig.set_data(img)
            plt.show()
            plt.title(slice_num)
            plt.pause(0.0001)

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
        data = self._parent.data.get_data().copy()

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
        data = self._parent.data.get_data().copy()

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
    def __init__(self, parent = []):
        self._parent = parent

    def l2_norm(self):
        return numpy.sum(self._parent.data.get_data() ** 2)

    def mean(self):
        return numpy.mean(self._parent.data.get_data())

    def min(self):
        return numpy.min(self._parent.data.get_data())

    def max(self):
        return numpy.max(self._parent.data.get_data())

    def center_of_mass(self):
        return measurements.center_of_mass(self._parent.data.get_data().max(1))

    def histogram(self, nbin = 256, plot = True, log = False):

        mi = self.min()
        ma = self.max()

        a, b = numpy.histogram(self._parent.data.get_data(), bins = nbin, range = [mi, ma])

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
#def interpolate_in_mask(image, mask, kernel):
#    '''
#    Replace masked pixels with interpolated values
#    '''
from scipy import ndimage
from tomopy.recon import rotation
import scipy.ndimage.interpolation as interp

class process(subclass):
    '''
    Various preprocessing routines
    '''
    def __init__(self, parent = []):
        self._parent = parent

    def arbitrary_function(self, func):
        '''
        Apply an arbitrary function:
        '''
        print(func)
        self._parent.data._data = func(self._parent.data._data)

        # add a record to the history:
        self._parent.meta.history['process.arbitrary_function'] = func.__name__

        self._parent.message('Arbitrary function applied.')

    def pixel_calibration(self, kernel=5):
        '''
        Apply correction to miscalibrated pixels.
        '''
        # Compute mean image of intensity variations that are < 5x5 pixels
        res = self._parent.data._data - ndimage.filters.median_filter(self._parent.data._data, [kernel, 1, kernel])
        res = res.mean(1)
        self._parent.data._data -= res.reshape((res.shape[0], 1, res.shape[1]))
        self._parent.meta.history['Pixel calibration'] = 1
        self._parent.message('Pixel calibration correction applied.')

    def medipix_quadrant_shift(self):
        '''
        Expand the middle line
        '''
        self._parent.data._data[:,:, 0:self._parent.data.shape(2)//2 - 2] = self._parent.data._data[:,:, 2:self._parent.data.shape(2)//2]
        self._parent.data._data[:,:, self._parent.data.shape(2)//2 + 2:] = self._parent.data._data[:,:, self._parent.data.shape(2)//2:-2]

        # Fill in two extra pixels:
        for ii in range(-2,2):
            closest_offset = -3 if (numpy.abs(-3-ii) < numpy.abs(2-ii)) else 2
            self._parent.data._data[:,:, self._parent.data.shape(2)//2 - ii] = self._parent.data._data[:,:, self._parent.data.shape(2)//2 + closest_offset]


        # Then in columns
        self._parent.data._data[0:self._parent.data.shape(0)//2 - 2,:,:] = self._parent.data._data[2:self._parent.data.shape(0)//2,:,:]
        self._parent.data._data[self._parent.data.shape(0)//2 + 2:, :, :] = self._parent.data._data[self._parent.data.shape(0)//2:-2,:,:]

        # Fill in two extra pixels:
        for jj in range(-2,2):
            closest_offset = -3 if (numpy.abs(-3-jj) < numpy.abs(2-jj)) else 2
            self._parent.data._data[self._parent.data.shape(0)//2 - jj,:,:] = self._parent.data._data[self._parent.data.shape(0)//2 + closest_offset,:,:]

        self._parent.meta.history['Quadrant shift'] = 1
        self._parent.message('Medipix quadrant shift applied.')

    def flat_field(self, kind=''):
        '''
        Apply flat field correction.
        '''

        if (str.lower(kind) == 'skyscan'):
            if ('object_bigger_than_fov' in self._parent.meta.geometry) and (self._parent.meta.geometry['object_bigger_than_fov']):
                print(self._parent.meta.geometry['object_bigger_than_fov'])
                air_values = numpy.ones_like(self._parent.data._data[:,:,0]) * 2**16 - 1
            else:
                air_values = numpy.max(self._parent.data._data, axis = 2)
                
            air_values = air_values.reshape((air_values.shape[0],air_values.shape[1],1))
            self._parent.data._data = self._parent.data._data / air_values
            
            # add a record to the history: 
            self._parent.meta.history['process.flat_field'] = 1

            self._parent.message('Skyscan flat field correction applied.')

        else:
            if numpy.min(self._parent.data._ref) <= 0:
                self._parent.warning('Flat field reference image contains zero (or negative) values! Will replace those with little tiny numbers.')

                tiny = self._parent.data._ref[self._parent.data._ref > 0].min()
                self._parent.data._ref[self._parent.data._ref <= 0] = tiny

            # How many projections:
            n_proj = self._parent.data.shape(1)

            for ii in range(0, n_proj):
                self._parent.data._data[:, ii, :] = self._parent.data._data[:, ii, :] / self._parent.data._ref

            # add a record to the history:
            self._parent.meta.history['process.flat_field'] = 1

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
        sdd = self._parent.meta.geometry['src2det']
        for u in range(0,weights.shape[2]):
            weights[:,:,u] = u

        weights = weights - weights.shape[2]/2
        weights = self._parent.meta.geometry['det_pixel']*weights
        weights = numpy.arctan(weights/sdd)

        theta = self._parent.meta.theta
        for ang in range(0,theta.shape[0]):
            tet = theta[ang]
            for u in range(0, weights.shape[2]):
                weights[:,ang,u] = _Parker_window(theta = tet, gamma = weights[0,ang,u], fan=fan_angle)

        self._parent.data._data *= weights
        # add a record to the history:
        self._parent.meta.history['process.short_scan'] = 1

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
        #self._parent.data._data[self._parent.data._data > upper_bound] = upper_bound
        #self._parent.data._data[~numpy.isfinite(self._parent.data._data)] = upper_bound

        #self._parent.data._data = numpy.nan_to_num(self._parent.data._data)
        numpy.clip(self._parent.data._data, a_min = lower_bound, a_max = upper_bound, out = self._parent.data._data)

        self._parent.message('Logarithm is applied.')
        self._parent.meta.history['process.log(upper_bound)'] = upper_bound

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

        self._parent.meta.history['process.salt_pepper(kernel)'] = kernel

    def simple_tilt(self, tilt):
        '''
        Tilts the sinogram
        '''
        for ii in range(0, self._parent.data.shape()[1]):
            self._parent.data._data[:, ii, :] = interp.rotate(numpy.squeeze(self._parent.data._data[:, ii, :]), -tilt, reshape=False)
            
        self._parent.message('Tilt is applied.')


    def center_shift(self, offset=None, test_path=None, ind=None, cen_range=None):
        '''
        Find the center of the sinogram and apply the shift to corect for the offset
        '''
        sz = self._parent.data.shape(2) // 2

        if test_path is not None:
            rotation.write_center(self._parent.data._data, theta=self._parent.meta.theta, dpath=test_path, ind=ind, cen_range=cen_range, sinogram_order=True)

        else:
            if offset is None:
                offset = sz-rotation.find_center(self._parent.data._data, self._parent.meta.theta,  sinogram_order=True)[0]
                #offset = sz-rotation.find_center_pc(self._parent.data._data[:,0,:], self._parent.data._data[:,self._parent.data.shape(1)//2,:])p

            else:
                # Do nothing is the offset is less than 1 pixel
                if (numpy.abs(offset) >= 1):
                    self._parent.data._data = interp.shift(self._parent.data._data, (0,0,offset))
                    self._parent.meta.history['process.center_shift(offset)'] = offset

                    self._parent.message('Horizontal offset corrected.')
                else:
                    self._parent.warning("Center shift found an offset smaller than 1. Correction won't be applied")




    def bin_theta(self):
        '''
        Bin angles with a factor of two
        '''
        self._parent.data._data = (self._parent.data._data[:,0:-1:2,:] + self._parent.data._data[:,1::2,:]) / 2
        self._parent.meta.theta = (self._parent.meta.theta[:,0:-1:2,:] + self._parent.meta.theta[:,1::2,:]) / 2

    def bin_data(self):
        '''
        Bin data with a factor of two
        '''
        self._parent.data._data = (self._parent.data._data[:, :, 0:-1:2] + self._parent.data._data[:, :, 1::2]) / 2
        self._parent.data._data = (self._parent.data._data[0:-1:2, :, :] + self._parent.data._data[1::2, :, :]) / 2

        self._parent.meta.geometry['det_pixel'] *= 2


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

        self._parent.meta.history['process.ccrop(top_left, bottom_right)'] = [top_left, bottom_right]

        self._parent.message('Sinogram cropped.')

    def crop_centered(self, center, dimensions):
        '''
        Crop the sinogram
        '''
        self._parent.data._data = self._parent.data._data[center[0] - dimensions[0]//2:center[0] + dimensions[0]//2, :, center[1] - dimensions[1]//2:center[1] + dimensions[1]//2]
        self._parent.data._data = numpy.ascontiguousarray(self._parent.data._data, dtype=numpy.float32)
        gc.collect()

        self._parent.meta.history['process.crop_centered(center, dimensions)'] = [center, dimensions]

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

    def __init__(self, sino = []):
        self._parent = sino
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
        sz = self._parent.data.shape()

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
        src2obj = prnt.meta.geometry['src2obj']
        total = prnt.meta.geometry['src2obj'] + prnt.meta.geometry['det2obj']
        pixel = prnt.meta.geometry['det_pixel']

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
        #pixel_size = prnt.meta.geometry['det_pixel']
        #det2obj = prnt.meta.geometry['det2obj']
        #src2obj = prnt.meta.geometry['src2obj']
        theta = prnt.meta.theta

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
        pixel_size = prnt.meta.geometry['det_pixel']
        det2obj = prnt.meta.geometry['det2obj']
        src2obj = prnt.meta.geometry['src2obj']
        theta = prnt.meta.theta

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
        sz = self._parent.data.shape()

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

        sz = self._parent.data.shape()
        geom = prnt.meta.geometry

        # Discrete reconstruction space: discretized functions on the rectangle.
        dim = numpy.array([sz[0], sz[2], sz[2]])
        space = odl.uniform_discr(min_pt = -dim / 2 * geom['img_pixel'], max_pt = dim / 2 * geom['img_pixel'], shape=dim, dtype='float32')

        # Angles: uniformly spaced, n = 1000, min = 0, max = pi
        angle_partition = odl.uniform_partition(geom['first_angle'], geom['last_angle'], geom['nb_angle'])

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

    def FDK(self, short_scan = None, min_constraint = None):
        '''

        '''
        prnt = self._parent

        # Initialize ASTRA:
        self._initialize_astra()

        # Run the reconstruction:
        #epsilon = numpy.pi / 180.0 # 1 degree - I deleted a part of code here by accident...
        theta = self._parent.meta.theta
        if short_scan is None:
            short_scan = (theta.max() - theta.min()) < (numpy.pi * 1.99)
        
        vol = self._backproject(prnt.data._data, algorithm='FDK_CUDA', short_scan = short_scan, min_constraint = min_constraint)

        vol = volume(vol)
        #vol.history['FDK'] = 'generated in '
        
        self._parent.message('FDK reconstruction performed.')
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
        
        text = 'SIRT reconstruction performed with %d iterations.' % iterations
        self._parent.message(text)
        return volume(vol)
        # No need to make a history record - sinogram is not changed.
        
        
    def SIRT_CPU(self, proj_type = 'cuda3d', iterations = 10, relaxation = 1.0, min_constraint = None, max_constraint = None):
        '''

        '''
        prnt = self._parent
        # Initialize ASTRA:
        self._initialize_astra()
        out = numpy.zeros(astra.functions.geom_size(self.vol_geom), dtype=numpy.float32)
        cfg = {}
        proj_id = 0
        rec_id = 0
        sino_id = 0
        sirt = astra.plugins.SIRTPlugin()
        try:
            proj_id = astra.create_projector(proj_type = proj_type, proj_geom = self.proj_geom, vol_geom = self.vol_geom)
            rec_id = astra.data3d.link('-vol', self.vol_geom, out)
            sino_id = astra.data3d.link('-sino', self.proj_geom, prnt.data._data)
            cfg['ProjectorId'] = proj_id
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = sino_id
            sirt.initialize(cfg = cfg, Relaxation = relaxation, MinConstraint = min_constraint, MaxConstraint = max_constraint)
            sirt.run(its = iterations)
            
        finally:
            astra.projector.delete(proj_id)
            astra.data3d.delete([rec_id, sino_id])
        
        text = 'SIRT-CPU reconstruction performed with %d iterations.' % iterations
        self._parent.message(text)
        
        return volume(out)
        # No need to make a history record - sinogram is not changed.
        
    
        

    def SIRT_custom(self, iterations = 10, min_constraint = None):
        '''

        '''
        prnt = self._parent
        # Initialize ASTRA:
        self._initialize_astra()

        # Create a volume containing only ones for forward projection weights
        sz = self._parent.data.shape()
        theta = self._parent.meta.theta

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
        theta = self._parent.meta.theta

        vol_ones = numpy.ones((sz[0], sz[2], sz[2]), dtype=numpy.float32)
        vol = numpy.zeros_like(vol_ones, dtype=numpy.float32)
        theta = self._parent.meta.theta
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


    def CGLS_CPU(self, proj_type = 'cuda3d', iterations = 10):
        '''

        '''
        prnt = self._parent
        # Initialize ASTRA:
        self._initialize_astra()
        out = numpy.zeros(astra.functions.geom_size(self.vol_geom), dtype=numpy.float32)
        cfg = {}
        proj_id = 0
        rec_id = 0
        sino_id = 0
        cgls = astra.plugins.CGLSPlugin()
        try:
            proj_id = astra.create_projector(proj_type = proj_type, proj_geom = self.proj_geom, vol_geom = self.vol_geom)
            rec_id = astra.data3d.link('-vol', self.vol_geom, out)
            sino_id = astra.data3d.link('-sino', self.proj_geom, prnt.data._data)
            cfg['ProjectorId'] = proj_id
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = sino_id
            cgls.initialize(cfg)
            cgls.run(its = iterations)
            
        finally:
            astra.projector.delete(proj_id)
            astra.data3d.delete([rec_id, sino_id])
        
        text = 'CGLS-CPU reconstruction performed with %d iterations.' % iterations
        self._parent.message(text)
        
        return volume(out)
        # No need to make a history record - sinogram is not changed.
    
    #----- ASTRA Utilities
    def _apply_geometry_modifiers(self, vectors):
        '''
        Apply arbitrary geometrical modifiers
        '''        
        #Detector shift (V):
        vectors[:,3:6] += self.geom_modifier['det_tra_vrt'] * vectors[:,9:12]

        #Detector shift (H):
        vectors[:,3:6] += self.geom_modifier['det_tra_hrz'] * vectors[:,6:9]

        #Source shift (V):
        vectors[:,0:3] += self.geom_modifier['src_tra_vrt'] * vectors[:,9:12]   

        #Source shift (H):
        vectors[:,0:3] += self.geom_modifier['src_tra_hrz'] * vectors[:,6:9]
        
        
                
    def _initialize_astra(self, sz = None, det_pixel_size = None, 
                          det2obj = None, src2obj = None, theta = None, vec_geom = False):

        if sz is None: sz = self._parent.data.shape()
        if det_pixel_size is None: det_pixel_size = self._parent.meta.geometry['det_pixel'] * self._parent.meta.geometry['det_binning'][0]
        if det2obj is None: det2obj = self._parent.meta.geometry['det2obj']
        if src2obj is None: src2obj = self._parent.meta.geometry['src2obj']
        if theta is None: theta = self._parent.meta.theta

        # Initialize ASTRA (3D):
        det_count_x = sz[2]
        det_count_z = sz[0]
        
        # Make volume count x > detector count to include corneres of the object:
        vol_count_x = sz[2]
        vol_count_z = sz[0]

        tot_dist = det2obj + src2obj

        magnification = tot_dist / src2obj
        
        if self.vol_geom is None:
            self.vol_geom = astra.create_vol_geom(vol_count_x, vol_count_x, vol_count_z)
        
        if self.proj_geom is None:
            self.proj_geom = astra.create_proj_geom('cone', magnification, magnification, det_count_z, det_count_x, theta, (src2obj*magnification)/det_pixel_size, (det2obj*magnification)/det_pixel_size)

        if (self._parent.meta.geometry['det_offset'] != [0.0,0.0]) or (self._parent.meta.geometry['det_tilt'] != 0.0):
            # Use now vec projection geometry to gain degrees of freedom
            if (self.proj_geom['type'] == 'cone'):
                self.proj_geom = astra.functions.geom_2vec(self.proj_geom)
            
            ### Translations of source and detector
            # Shift the detector by the de-magnified amount horizontally and change the reconstruction window accordingly
            vectors = self.proj_geom['Vectors']
            if (self._parent.meta.geometry['det_offset'] != [0.0,0.0]):
                vectors[:,3:6] = vectors[:,3:6] - self._parent.meta.geometry['det_offset'][0]/magnification * vectors[:,6:9]
                #self.vol_geom['option']['WindowMinX'] = -self.vol_geom['GridColCount'] / 2.0 - self._parent.meta.geometry['det_offset'][0]/magnification - (self._parent.meta.geometry['det_offset'][0]/magnification)*(magnification-1.0)
                #self.vol_geom['option']['WindowMaxX'] = self.vol_geom['GridColCount'] / 2.0 - self._parent.meta.geometry['det_offset'][0]/magnification - (self._parent.meta.geometry['det_offset'][0]/magnification)*(magnification-1.0)
                #self.vol_geom['option']['WindowMinX'] = -self.vol_geom['GridColCount'] / 2.0 - self._parent.meta.geometry['det_offset'][0]
                #self.vol_geom['option']['WindowMaxX'] = self.vol_geom['GridColCount'] / 2.0 - self._parent.meta.geometry['det_offset'][0]
                
                
                # Shift the source by the de-magnified amount horizontally
                vectors[:,0:3] = vectors[:,0:3] - self._parent.meta.geometry['det_offset'][0]/magnification * vectors[:,6:9]
                
            
                # Shift the detector to vertically align the optical axis and change the reconstruction window accordingly
                #vectors[:,3:6] = vectors[:,3:6] - self._parent.meta.geometry['det_offset'][1] * vectors[:,9:12]
                vectors[:,0:3] = vectors[:,0:3] + self._parent.meta.geometry['det_offset'][1] * vectors[:,9:12]
                #self.vol_geom['option']['WindowMinZ'] = -self.vol_geom['GridSliceCount'] / 2.0 - self._parent.meta.geometry['det_offset'][1]
                self.vol_geom['option']['WindowMinZ'] = -self.vol_geom['GridSliceCount'] / 2.0 + self._parent.meta.geometry['det_offset'][1]*(magnification - 1)
                #self.vol_geom['option']['WindowMaxZ'] = self.vol_geom['GridSliceCount'] / 2.0 - self._parent.meta.geometry['det_offset'][1]
                self.vol_geom['option']['WindowMaxZ'] = self.vol_geom['GridSliceCount'] / 2.0 + self._parent.meta.geometry['det_offset'][1]*(magnification - 1)
            
            
            ### Random thermal movements
            if ('thermal_shifts' in self._parent.meta.geometry.keys()):
                thermal_shifts = self._parent.meta.geometry['thermal_shifts']
                
                # These shifts are apparent on the detector, but come from the source
                # Fix the geometry by moving the source by delta/(magnification - 1)
                vectors[:,0:3] = vectors[:,0:3] - numpy.reshape(thermal_shifts[:,0], (thermal_shifts.shape[0],1)) /(magnification - 1) * vectors[:,6:9]
                vectors[:,0:3] = vectors[:,0:3] - numpy.reshape(thermal_shifts[:,1], (thermal_shifts.shape[0],1)) /(magnification - 1) * vectors[:,9:12]
                

            
            ### In-plane rotation (tilt) of detector
            if (self._parent.meta.geometry['det_tilt'] != 0.0):
                for i in range(0,vectors.shape[0]):
                    # Define rotation axis by normal to the detector
                    tilt = - self._parent.meta.geometry['det_tilt']
                    rot_axis = numpy.cross(vectors[i,6:9], vectors[i,9:12])
                    rot_axis = rot_axis / numpy.sqrt(numpy.dot(rot_axis, rot_axis))
                    #rot_axis = rot_axis / numpy.sqrt(rot_axis[0]*rot_axis[0]+rot_axis[1]*rot_axis[1]+rot_axis[2]*rot_axis[2])
                    # Compute rotation matrix
                    c = numpy.cos(tilt)
                    s = numpy.sin(tilt)

                    rot_matrix = numpy.zeros((3,3))
                    rot_matrix[0,0] = rot_axis[0]*rot_axis[0]*(1.0-c)+c
                    rot_matrix[0,1] = rot_axis[0]*rot_axis[1]*(1.0-c)-rot_axis[2]*s
                    rot_matrix[0,2] = rot_axis[0]*rot_axis[2]*(1.0-c)+rot_axis[1]*s
                    rot_matrix[1,0] = rot_axis[1]*rot_axis[0]*(1.0-c)+rot_axis[2]*s
                    rot_matrix[1,1] = rot_axis[1]*rot_axis[1]*(1.0-c)+c
                    rot_matrix[1,2] = rot_axis[1]*rot_axis[2]*(1.0-c)-rot_axis[0]*s
                    rot_matrix[2,0] = rot_axis[2]*rot_axis[0]*(1.0-c)-rot_axis[1]*s
                    rot_matrix[2,1] = rot_axis[2]*rot_axis[1]*(1.0-c)+rot_axis[0]*s
                    rot_matrix[2,2] = rot_axis[2]*rot_axis[2]*(1.0-c)+c
                    
                    #Apply rotation matrix for each detector position  self.W = astra.OpTomo(cfg['ProjectorId'])    
                    vectors[i,6:9] = numpy.dot(rot_matrix, vectors[i,6:9])
                    vectors[i,9:12] = numpy.dot(rot_matrix, vectors[i,9:12])
            
                
            
            # self._apply_geometry_modifiers(vectors)


    def _initialize_ramp_filter(self, power = 1):
      sz = self._parent.data.shape()

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
    
              slice_proj_geom = astra.create_proj_geom('parallel', 1.0, sz[1], self._parent.meta.theta)
    
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
        pixel_size = prnt.meta.geometry['det_pixel']
        det2obj = prnt.meta.geometry['det2obj']
        src2obj = prnt.meta.geometry['src2obj']
        theta = prnt.meta.theta


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
#           DATA class and subclasses
# **************************************************************
import scipy.interpolate as interp_sc
import sys

class data(subclass):
    '''
    Memory allocation, reading and writing the data
    '''
    _data = []
    _ref = []
    _backup = []

    _isgpu = False

    def __init__(self, parent = []):
        self._parent = parent

    def interpolate_slice(self, target_theta)        :
        '''
        Get one slice at a given theta, use interpolation is needed.
        '''
        sz = self.shape()
        thetas = self._parent.meta.theta

        interp_grid = numpy.transpose(numpy.meshgrid(target_theta, numpy.arange(sz[0]), numpy.arange(sz[2])), (1,2,3,0))
        original_grid = (numpy.arange(sz[0]), thetas, numpy.arange(sz[2]))

        return interp_sc.interpn(original_grid, self._data, interp_grid)

    def get_data(self):
        '''
        Get sinogram data. Copies data from GPU if needed
        '''
        return self._data

    def set_data(self, data):
        '''
        Set sinogram data. Copies data to GPU if needed
        '''
        self._data = data

        self._parent.meta.history['data.set_data'] = 1

    def shape(self, dim = None):
        if dim is None:
            return self._data.shape
        else:
            return self._data.shape[dim]

    def size_mb(self):
        '''
        Get the size of the data object in MB.
        '''
        return sys.getsizeof(self)

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

        volume.data.set_data((volume.data.get_data() > threshold) * 1.0)

    def measure_thickness(self, volume, obj_intensity = None):
        '''
        Measure average thickness of an object.
        '''

        # Apply threshold:
        self.treshold(volume, obj_intensity)

        # Skeletonize:
        skeleton = morphology.skeletonize3d(volume.data.get_data())

        # Compute distance across the wall:
        distance = ndimage.distance_transform_bf(volume.data.get_data()) * 2

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

import random

class sinogram(object):
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
        self.meta.history['error'] = msg
        raise ValueError(msg)

    def warning(self, msg):
        '''
        Throw a warning. In their face!
        '''
        self.meta.history['warning'] = msg
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
            self.message((k in self.meta.history.keys()))

            if not(k in self.meta.history.keys()):
                self.message('You should use ' + k + ' as a next step')
                finished = False
                break

            if finished:
                self.message('All basic processing steps were done. Use "reconstruct.FDK" to compute filtered backprojection.')

    def _check_double_hist(self, new_key):
        '''
        Check if the operation was already done
        '''
        if new_key in self.meta.history.keys():
            self.error(new_key + ' is found in the history of operations! Aborting.')
