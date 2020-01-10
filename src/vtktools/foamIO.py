'''
Provides a wrapper for the OpenFOAM-reader in VTK, giving
a more intuitive interface to iterate over the timeseries. 
A helper function for writing OpenFOAM-data is also provided.
'''
from __future__ import print_function

import numpy as np
import vtk
import re
import os

# Abstraction for OpenFOAM dimensions
class _DataDimension:
    def __init__(self, dimList):
        self.dimList = np.array(dimList)

    def __mul__(self, rhs):
        return _DataDimension(self.dimList + rhs.dimList)

    def __div__(self, rhs):
        return _DataDimension(self.dimList - rhs.dimList)

    def __pow__(self, rhs):
        return _DataDimension(rhs*self.dimList)

    def __str__(self):
        return '[{} {} {} {} 0 0 0]'.format(self.dimList[0], self.dimList[1], self.dimList[2], self.dimList[3])

Dimensionless = _DataDimension([0, 0, 0, 0])
DimMass = _DataDimension([1, 0, 0, 0])
DimLength = _DataDimension([0, 1, 0, 0])
DimTime = _DataDimension([0, 0, 1, 0])
DimTemperature = _DataDimension([0, 0, 0, 1])

class FoamFileVersion:
    def __init__(self, major, minor):
        self.major = major
        self.minor = minor

    @staticmethod
    def fromString(versionString):
        parts = versionString.split('.')
        self.major = int(parts[0])
        self.minor = int(parts[1])

    def __str__(self):
        return str(self.major) + '.' + str(self.minor)

class FoamFileInformation:
    def __init__(self, **kwargs):
        if not 'version' in kwargs:
            self.version = FoamFileVersion(2, 0)
        else:
            self.version = FoamFileVersion.fromString(kwargs['version'])

        self.format = kwargs.get('format', 'ascii')
        if not self.format in ['ascii', 'binary']:
            raise ValueError('Unknown format type: ' + self.format)

        self.location = kwargs.get('location')
        self.className = kwargs.get('className')
        self.object = kwargs.get('object')

def _writeFoamFileHeader(f, fileInformation):
    f.write('/*--------------------------------*- C++ -*----------------------------------*\ \n')
    f.write('| =========                 |                                                 |\n')
    f.write('| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
    f.write('|  \\    /   O peration     | Version:  2.3.x                                 |\n')
    f.write('|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
    f.write('|    \\/     M anipulation  |                                                 |\n')
    f.write('\*---------------------------------------------------------------------------*/\n')
    f.write('FoamFile\n')
    f.write('{\n')
    f.write('	version     ' +str(fileInformation.version) +';\n')
    f.write('	format      ' +fileInformation.format       + ';\n')
    f.write('	class       ' +fileInformation.className    + ';\n')
    f.write('	location    "'+fileInformation.location     +'";\n')
    f.write('	object      ' +fileInformation.object       + ';\n')
    f.write('}\n')
    f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')

def readFoamDict(fileName):
    # Open file
    with open(fileName, 'r') as f:
        # Get lines
        text = '\n'.join(f.readlines())
    # Strip multiline comments
    text = re.sub(re.compile('/\*.*?\*/', re.DOTALL), '', text)	
    # Strip single line comments
    text = re.sub(re.compile('//.*?\n'), '', text)
    # Remove newlines
    text = text.replace('\r\n', '').replace('\n', '')

    # Intermediate represenation of the foam file
    class FoamEntry(dict):
        def __init__(self, parent=None):
            self.parent = parent

    # Recursively split the file into blocks defined by { }-clauses
    root = FoamEntry()
    currentNode = [root] # Put in list to allow for mutable access in the inline functions

    def onOpenNewBlock(scanner, token):
        newNode = FoamEntry(currentNode[0])
        currentNode[0][token.strip('{').strip()] = newNode
        currentNode[0] = newNode

    def onCloseBlock(scanner, token):
        currentNode[0] = currentNode[0].parent

    def onOtherBlock(scanner, token):
        # Remove ;, and trailing whitespace
        token = token.strip(';').strip()
        if not len(token) == 0:
            # Split into key and the rest
            key, sep, value = token.partition(' ')
            currentNode[0][key.strip()] = value.strip()

    scanner = re.Scanner([
        ('.+?\{', onOpenNewBlock),
        ('\}', onCloseBlock),
        (r"\s+", None), # skip whitespace
        (r".+?(?=(\;|\}|\{|$))", onOtherBlock)
    ])
    scanner.scan(text)
    return root    

class OpenFOAMReader:
    """Convenience wrapper around vtkOpenFOAMReader.
    Allows all reader options to be set in the constructor,
    and simplifies iteration over a timeseries.

    Args:
        :fileName (str): .foam-file to read (should be placed in the root folder of the case)
    
    Keyword args:
        :cellArrays = [str]: List of cell data array names to read
        :pointArrays = [str]: List of point data array names to read
        :lagrangianArrays = [str]: List of lagrangian array names to read
        :patchArrays = [str]: List of region names to read from
        :decomposePolyhedra = (True|False): Decompose polyhedral mesh elements to tetras and wedges
        :cellToPoint = (True|False): Convert cell data to point data
        :cacheMesh = (True|False): Cache or re-read the mesh at each timestep

    Returns:
        OpenFOAMReader

    """

    _currentIteration = -1
    _isInit = False
    _finishedReading = False

    def __init__(self, fileName, **kwargs):
        self.reader = vtk.vtkOpenFOAMReader()
        self.reader.SetFileName(fileName)
        self.reader.UpdateInformation()
        self.timeValues = self.reader.GetTimeValues()
        self.Nt = self.timeValues.GetNumberOfTuples()
        self.caseFolder = os.path.dirname(fileName)

        # Default options (read everything)
        self.reader.DecomposePolyhedraOn()
        self.reader.CacheMeshOn()
        self.reader.EnableAllCellArrays()
        self.reader.EnableAllPointArrays()
        self.reader.CreateCellToPointOff()
        self.reader.EnableAllPatchArrays()
        self.reader.EnableAllLagrangianArrays()

        # Time dictionary for the current timestep
        self.__timeDict = None # Lazy evaluation

        # Read options
        for k,v in kwargs.items():
            #print k, ' => ', v
            if k == 'cellArrays':
                self.reader.DisableAllCellArrays()
                if v:
                    for cellArray in v:
                        self.reader.SetCellArrayStatus(cellArray, 1)
            elif k == 'patchArrays':
                patches = self.getPatchNames()
                self.reader.DisableAllPatchArrays()
                if v:
                    for patchArray in v:
                        if not patchArray in patches:
                            print('Available patches')
                            for patch in patches:
                                print(' ' + patch)
                            raise RuntimeError("Unknown patch name " +patchArray)
                        self.reader.SetPatchArrayStatus(patchArray, 1)
            elif k == 'pointArrays':
                self.reader.DisableAllPointArrays()
                if v:
                    for pointArray in v:
                        self.reader.SetPointArrayStatus(pointArray, 1)
            elif k == 'lagrangianArrays':
                self.reader.DisableAllLagrangianArrays()
                if v:
                    for lagrangianArray in v:
                        self.reader.SetLagrangianArrayStatus(lagrangianArray, 1)
            elif k == 'decomposePolyhedra':
                if v:
                    self.reader.DecomposePolyhedraOn()
                else:
                    self.reader.DecomposePolyhedraOff()
            elif k == 'cellToPoint':
                if v:
                    self.reader.CreateCellToPointOn()
                else:
                    self.reader.CreateCellToPointOff()		
            elif k == 'cacheMesh':
                if v:
                    self.reader.CacheMeshOn()
                else:
                    self.reader.CacheMeshOff()
            else:
                raise ValueError('Invalid argument ' + k)

    def getDataSet(self):
        """Get the dataset at the current timestep
        
        Returns:
            vtkMultiBlockDataSet
        """
        if not self._isInit:
            raise RuntimeError('The reader has not read anything yet!')
        if self.finishedReading():
            raise RuntimeError('The reader has no more data to read')
        return self.reader.GetOutput()

    def getPatchNames(self):
        """Get a list of names of all regions/patches in the dataset
        
        Returns:
            [regionName1, regionName2, ...]
        """
        return [self.reader.GetPatchArrayName(i) for i in range(self.reader.GetNumberOfPatchArrays())]

    def readIteration(self, iteration):
        """Set the timestep of the reader, and read the data.
        
        Args:
            :iteration (int): Timestep number to read

        Returns:
            Nothing

        Raises:
            :ValueError: If the iteration number is out of range
        """
        if iteration < 0 or iteration >= self.Nt:
            raise ValueError('Iteration number out of range')
        self._currentIteration = iteration
        self.reader.SetTimeValue(self.timeValues.GetTuple(iteration)[0])
        self.reader.Modified()
        self.reader.Update()
        self._isInit = True

        # Invalidate the timeDict
        self.__timeDict = None

    def readTime(self, t):
        """Set the time (in seconds) of the reader, and read the data.
        
        Args:
            :t (float): Time to read

        Returns:
            Nothing

        Raises:
            :ValueError: If the time value was not found in the dataset
        """
        for i in range(self.Nt):
            if self.timeValues.GetTuple(i)[0] == t:
                it = i
                break
        else:
            raise ValueError('The time ' + str(t) + ' did not exist in the dataset')
        self.readIteration(i)

    def currentTime(self):
        """Get the current time (in seconds) of the reader

        Returns:
            float: Current time
        """
        return self.timeValues.GetValue(self._currentIteration)

    def currentIteration(self):
        """Get the current timestep index of the reader

        Returns:
            int: Current timestep
        """
        return self._currentIteration

    def finishedReading(self):
        """Check if the reader has reached the last timestep, can be used in combination with
        readNext() to iterate over the dataset::
            reader = OpenFOAMReader(filename)
            reader.startReading()
            while not reader.finishedReader():
                # do stuff
                reader.readNext()

        Returns:
            bool: True if end has been reached, otherwise False
        """
        
        return self._finishedReading

    def startReading(self):
        """Read the first timestep of the dataset

        Raises:
            :ValueError: If no data is found in the dataset

        Returns:
            Nothing
        """
        self._finishedReading = False
        self.readIteration(0)

    def skipAndRead(self, n):
        """Read the data at n timesteps from the current timestep. Can be used to iterate over the
        dataset::
            reader = OpenFOAMReader(filename)
            reader.startReading()
            while not reader.finishedReader():
                # do stuff
                reader.skipAndRead(10) # Skip 9 timesteps

        Params:
            :n (int): Number of timesteps to move forward in time

        Raises:
            :ValueError: If attempting to read of out range
            :RuntimeError: If no data has yet been read (no prior call has been made to startReading, readIteration or readTime)

        Returns:
            Nothing
        """
        if not self._isInit:
            raise RuntimeError('The reader has not been intialized, call startReading()/readIteration()/readTime() before readNext()/skipAndRead()')
        if (self._currentIteration+n) >= self.Nt:
            self._finishedReading = True
        else:
            self.readIteration(self._currentIteration+n)

    def readNext(self):
        """Read the next timestep of the dataset, can be used in combination with finishedReading to iterate
        over the dataset::
            reader = OpenFOAMReader(filename)
            reader.startReading()
            while not reader.finishedReader():
                # do stuff
                reader.readNext()

        Returns:
            Nothing
        """
        self.skipAndRead(1)

    def deltaT(self):
        return float(self.timeDict.get('deltaT'))

    def deltaT0(self):
        return float(self.timeDict.get('deltaT0'))

    @property
    def timeDict(self):
        if not self._isInit:
            raise RuntimeError('The reader has not been intialized, call startReading()/readIteration()/readTime() before this function')
        if self.__timeDict is None:
            self.__timeDict = readFoamDict(os.path.join(self.caseFolder, _foamTimeToString(self.currentTime()), 'uniform', 'time'))
        return self.__timeDict

# Convert a floating point time to a string
# OF represents integer times without decimal point
def _foamTimeToString(time):
    if int(time) == time:
        return str(int(time)) 
    else: 
        return str(time)

def _writeVector(f, vec):
    f.write('(')
    for i in range(vec.size):
        if i != 0:
            f.write(' ')
        f.write(str(vec[i]))
    f.write(')')	

def _writeField(f, fieldData, fieldType):
    if fieldType == 'volScalarField':
        if fieldData.ndim == 0:
            # Uniform data
            f.write('uniform ' + str(fieldData))
        else:
            # Non-uniform data
            f.write('nonuniform List<scalar> ' + str(fieldData.shape[0]) + '\n(\n')
            for i in range(0, fieldData.shape[0]):
                f.write(str(fieldData[i]) + '\n')
            f.write(')')
    else:
        if fieldData.ndim == 1:
            # Uniform data
            f.write('uniform ')
            _writeVector(f, fieldData)
        else:
            # Non-uniform data
            f.write('nonuniform List<')
            if fieldType == 'volVectorField':
                f.write('vector')
            elif fieldType == 'volSymmTensorField':
                f.write('symmTensor')
            elif fieldType == 'volTensorField':
                f.write('tensor')
            else:
                raise RuntimeError('Unknown field type ' + fieldType)
            f.write('> ' + str(fieldData.shape[0]) + '\n(\n')
            for i in range(fieldData.shape[0]):
                _writeVector(f, fieldData[i, :])
                f.write('\n')
            f.write(')')
    f.write(';\n')

def _determineDataType(fields):
    maxSize = [0, 0]
    for field in fields:
        for axis in range(field.ndim):
            maxSize[axis] = max(field.shape[axis], maxSize[axis])
    if maxSize[1] != 0:
        if maxSize[1] == 1:
            return 'volScalarField'
        if maxSize[1] == 3:
            return 'volVectorField'
        if maxSize[1] == 6:
            return 'volSymmTensorField'
        if maxSize[1] == 9:
            return 'volTensorField'
    else:
        if maxSize[0] != 0:
            return 'volScalarField'
    raise RuntimeError('Unable to determine data type')

def writeFoamData(caseFolder, fieldName, **kwargs):
    """Write numpy-arrays to OpenFOAM-files.
    Args:
        :caseFolder (str): root folder of the OF case to write to
        :fieldName (str): name of the field to write

    Keyword args:
        :internalField = np.array: Internal field data to write
        :boundaryField = {'boundaryName1'\: np.array, 'boundaryName2'\: np.array, ...}: Dictionary of boundary name-numpy data arrays of data to write
        :dimension (DataDimension): Physical dimension (e.g. length/time) of the data to write
        :time (float): Time value at which the data should be written (e.g. if t=0.1, the data will be written to caseFolder/0.1/)
        :fieldType (str): Type of the data to write (one of volScalarField / volVectorField / volSymmTensorField / volTensorField). If not provided,
            the function will make an attempt at deducing the type.

    Returns: 
        Nothing

    """
    internalField = kwargs.get('internalField')
    boundaryField = kwargs.get('boundaryField')
    dimension = kwargs.get('dimension', Dimensionless)
    time = kwargs.get('time', 0)
    fieldType = kwargs.get('fieldType', None)
    timeStr = _foamTimeToString(time)

    # Determine datatype if not provided
    if fieldType == None:
        fieldType = _determineDataType([internalField] + boundaryField.values())
    
    # Verify that all the fields conform to the provided data type
    fieldTests = dict()
    fieldTests['volTensorField'] = [lambda field: ((field.ndim == 1 and field.shape[0] == 9) or (field.ndim == 2 and field.shape[1] == 9)), '(9, ) or (x, 9)']
    fieldTests['volScalarField'] = [lambda field: (field.ndim == 0 or field.ndim == 1), '() or (x,)']
    fieldTests['volVectorField'] = [lambda field: ((field.ndim == 1 and field.shape[0] == 3) or (field.ndim == 2 and field.shape[1] == 3)), '(x, 3) or (3,)']
    fieldTests['volSymmTensorField'] = [lambda field: ((field.ndim == 1 and field.shape[0] == 6) or (field.ndim == 2 and field.shape[1] == 6)), '(x, 6) or (6,)']

    fieldTest = fieldTests[fieldType][0]
    expectedShape = fieldTests[fieldType][1]
    
    for field, domainName in zip([internalField] + boundaryField.values(), ['internalField'] + boundaryField.keys()):
        if not isinstance(field, basestring):
            if not fieldTest(field):
                raise RuntimeError('Invalid ' + fieldType +' shape ' + str(field.shape) + ' in ' + domainName + ' (expected ' + expectedShape + ')')

    with open(caseFolder + '/'+timeStr+'/' + fieldName, 'w') as f:
        _writeFoamFileHeader(f, FoamFileInformation(format='ascii', className = fieldType, location = timeStr, object = fieldName))
        f.write('dimensions ' + str(dimension) + ';\n')
        
        # Write the internal field
        if internalField != None:
            f.write('internalField ')
            _writeField(f, internalField, fieldType)

        # Write boundaries
        if boundaryField != None:
            f.write('boundaryField\n')
            f.write('{\n')
            for boundaryName, data in boundaryField.iteritems():
                f.write('	'+ boundaryName+'\n')
                f.write('	{\n')
                if isinstance(data, basestring):
                    f.write('		type ' + data + ';\n')
                else:
                    f.write('		type fixedValue;\n')
                    f.write('		value ')
                    _writeField(f, data, fieldType)
                f.write('	}\n')
            f.write('}\n')
