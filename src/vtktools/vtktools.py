import numpy as np
import vtk
import vtk.numpy_interface
import math
from vtk.numpy_interface import dataset_adapter as dsa

class PolyLineIterator:
	'''Iterator for point indices in a vtkPolyLine object::
		polyLine = vtk.vtkPolyLine()
		for pi in PolyLineIterator(pl):
			point = polyLine.GetPoint(pi)
	
	Args:
		:cell: Polyline object to iterate over
	'''

	def __init__(self, cell):
		if not cell.IsA('vtkPolyLine'):
			raise TypeError('Unable to create the cell iterator, only vtkPolyLines supported as cell type')
		self.i = 0
		self.n = cell.GetNumberOfPoints()
		self.pl = cell

	def __iter__(self):
		return self

	def next(self):
		'''Advance the iterator and return the current point index'''
		if self.i < self.n:
			i = self.i
			self.i += 1
			return self.pl.GetPointId(i)
		else:
			raise StopIteration()

class CellIterator:
	'''Iterator for cells in a vtk dataset::
		dataSet = vtk.vtkPolyData()
		for cell in CellIterator(dataSet):
			# Do something with the cell
	
	Args:
		:dataSet: non-composite vtk dataset, e.g. vtkPolyData or vtkUnstructuredGrid
	'''

	def __init__(self, dataSet):
		self.i = 0
		self.n = dataSet.GetNumberOfCells()
		self.ds = dataSet

	def __iter__(self):
		return self
	
	def next(self):
		'''Advance the iterator and return the current cell'''
		if self.i < self.n:
			i = self.i
			self.i += 1
			return self.ds.GetCell(i)
		else:
			raise StopIteration()

def _createMapper(dataSet):
	if dataSet.IsA('vtkPolyData'):
		mapper = vtk.vtkPolyDataMapper()
	elif dataSet.IsA('vtkUnstructuredGrid'):
		mapper = vtk.vtkDataSetMapper()
	else:
		raise RuntimeError('Unsupported dataset type')
	mapper.SetInputData(dataSet)
	return mapper

def renderDataSet(dataSet, **kwargs):
	""" Render a vtk dataset

	Args:
		:dataSet: vtkUnstructuredGrid or vtkPolyData

	Keyword args:
		:colorBy (str): constant or scalar
		:color (tuple): rgb value (if colorBy == constant)
	"""

	colorBy = kwargs.get('colorBy', 'constant')

	mapper = _createMapper(dataSet)

	actor = vtk.vtkActor()
	actor.SetMapper(mapper)

	renderer = vtk.vtkRenderer()
	renderer.SetBackground(1., 1., 1.)
	renderer.AddActor(actor)

	# Coloring
	if colorBy == 'constant':
		color = kwargs.get('color', (0.5, 0.5, 0.5))
		mapper.ScalarVisibilityOff()
		actor.GetProperty().SetColor(color[0], color[1], color[2])
	elif colorBy == 'scalar':
		# Create lookup table
		lut = vtk.vtkLookupTable()
		lut.SetTableRange(0, 1)
		lut.SetHueRange(0, 1)
		lut.SetSaturationRange(1, 1)
		lut.SetValueRange(1, 1)
		lut.Build()

		# Create colorbar
		colorbar = vtk.vtkScalarBarActor()
		colorbar.SetLookupTable(lut)
		colorbar.SetTitle('aaa')
		colorbar.SetNumberOfLabels(4)
		renderer.AddActor2D(colorbar)

		# Add colormap to the actor
		mapper.ScalarVisibilityOn()		
		mapper.SetLookupTable(lut)
	

	# Setup camera
	'''bounds = polyData.GetBounds()
	cx = (bounds[0] + bounds[1]) / 2.0
	cy = (bounds[2] + bounds[3]) / 2.0
	scale = max(math.fabs(bounds[0] - cx), math.fabs(bounds[1] - cx), math.fabs(bounds[2] - cy), math.fabs(bounds[3] - cy))
	camera = renderer.GetActiveCamera()
	camera.ParallelProjectionOn()
	camera.SetParallelScale(scale)
	camera.SetPosition(cx, cy, 1)
	camera.SetFocalPoint(cx, cy, 0)'''

	renderWindow = vtk.vtkRenderWindow()
	renderWindow.SetSize(600, 600)
	renderWindow.AddRenderer(renderer)
	#renderWindow.Render()

	interactor = vtk.vtkRenderWindowInteractor()
	interactor.SetRenderWindow(renderWindow)
	interactor.Initialize()
	#interactor.Render()
	interactor.Start()

def cutDataSet(dataSet, point, normal):
	plane = vtk.vtkPlane()
	plane.SetOrigin(point[0], point[1], point[2])
	plane.SetNormal(normal[0], normal[1], normal[2])
	
	cutter = vtk.vtkCutter()
	cutter.SetInputData(dataSet)
	cutter.SetCutFunction(plane)
	cutter.Update()
	return cutter.GetOutput()

def cutPolySurface(dataSet, point, normal):
	''' Cut a surface with a plane, and return an ordered list
	of points around the circumference of the resulting curve. The cut
	must result in a closed loop, and if the cut produces multiple sub-curves
	the closest one is returned.

	Args:
		:dataSet: (vtkPolyData): surface dataset
		:point: origin of the cutplane
		:normal: normal of the cutplane

	Returns:	
		:np.array: List of positions around the cicumference of the cut

	Raises:
		RuntimeError: If the cut results in a non-closed loop being formed
	'''

	# Generate surface cutcurve
	cutData = cutDataSet(dataSet, point, normal)
	
	edges = []
	cutLines = cutData.GetLines()
	cutLines.InitTraversal()
	idList = vtk.vtkIdList()
	while cutLines.GetNextCell(idList) == 1:
		edges.append((idList.GetId(0), idList.GetId(1)))	

	# Gather all points by traversing the edge graph starting
	# from the point closest to the centerline point
	locator = vtk.vtkPointLocator()
	locator.SetDataSet(cutData)
	locator.BuildLocator()
	startPtId = locator.FindClosestPoint(point)

	pointIds = [startPtId]
	try:
		while True:
			# Find the edge that starts at the latest point 
			pred = (v[1] for v in edges if v[0] == pointIds[-1])
			currentPtId = next(pred)

			# Check if we've returned to the start point
			if currentPtId == startPtId:
				break

			pointIds.append(currentPtId)
		else:	# if no break occured
			raise RuntimeError('The cut curve does not form a closed loop')
	except:
		# We reached the end of the edge graph without getting back to the beginning
		raise RuntimeError('The cut curve does not form a closed loop')
	cutCurve = dsa.WrapDataObject(cutData)
	return cutCurve.Points[pointIds]

def createVtkCylinder(**kwargs):
	''' Create a vtk cylinder
		Keyword arguments:
			:origin (tuple/list/np-array): Origin of the cylinder
			:axis (tuple/list/np-array): Cylinder axis
			:radius (float): Cylinder radius
	'''
	origin = np.array(kwargs.get('origin', [0, 0, 0]))
	axis = np.array(kwargs.get('axis', [0, 0, 1]))
	radius = np.array(kwargs.get('radius', 1))

	cylinder = vtk.vtkCylinder()
	cylinder.SetCenter(0, 0, 0)
	cylinder.SetRadius(radius)

	# The cylinder is (by default) aligned with the y-axis
	# Rotate the cylinder so that it is aligned with the tangent vector
	transform = vtk.vtkTransform()

	yDir = np.array([0, 1, 0])
	# If the tangent is in the y-direction, do nothing
	if np.abs(1. - np.abs(np.dot(yDir, axis))) > 1e-8:
		# Create a vector in the normal direction to the plane spanned by yDir and the tangent
		rotVec = np.cross(yDir, axis)
		rotVec /= np.linalg.norm(rotVec)
		
		# Evaluate rotation angle
		rotAngle = np.arccos(np.dot(yDir, axis))
		transform.RotateWXYZ(-180*rotAngle/np.pi, rotVec)

	transform.Translate(-origin)
	cylinder.SetTransform(transform)
	
	return cylinder

def cutPolyData(dataSet, **kwargs):
	# Read options
	pt = kwargs.get('point')
	normal = kwargs.get('normal')
	delta = kwargs.get('maxDist')

	if pt == None:
		raise RuntimeError('No point provided')
	if normal == None:
		raise RuntimeError('No normal provided')

	# Create plane
	plane = vtk.vtkPlane()
	plane.SetOrigin(pt[0], pt[1], pt[2])
	plane.SetNormal(normal[0], normal[1], normal[2])

	cutter = vtk.vtkCutter()
	cutter.SetCutFunction(plane)
	cutter.SetInputData(dataSet)

	if delta == None:
		cutter.Update()
		return cutter.GetOutput()
	else:
		# Create a box
		box = vtk.vtkBox()
		box.SetBounds(pt[0]-delta, pt[0]+delta, pt[1]-delta, pt[1]+delta, pt[2]-delta, pt[2]+delta)
	
		# Clip data with a box
		clipper = vtk.vtkClipDataSet()
		clipper.SetClipFunction(box)
		clipper.SetInputConnection(cutter.GetOutputPort())
		clipper.InsideOutOn()
		clipper.Update()
		return clipper.GetOutput()

def clearCellData(dataSet):
	for i in reversed(range(dataSet.GetCellData().GetNumberOfArrays())):
		dataSet.GetCellData().RemoveArray(i)

def clearPointData(dataSet):
	for i in reversed(range(dataSet.GetPointData().GetNumberOfArrays())):
		dataSet.GetPointData().RemoveArray(i)

def createQuadCells(Nx, Ny, **kwargs):
	''' Create quad cells corresponding to a Nx x Ny grid. 
	Parameters:
		:Nx (int): Number of points in the first dimension
		:Ny (int): Number of points in the second dimension
	Keyword arguments:
		:cutsectionIsClosed (bool or [bool]): Flag to indicate whether to wrap around the indices
		This is useful for creating a cell distribution for a cylinder
		Can also be an array of bools each indicating whether each of the sections in the x-direction should be wrapped 
	'''
	cells = vtk.vtkCellArray()
	cutsectionIsClosed = kwargs.get('cutsectionIsClosed', True)

	for i in range(1, Nx):
		if isinstance(cutsectionIsClosed, bool):
			wrapAround = cutsectionIsClosed
		else:
			# Assume array like
			wrapAround = cutsectionIsClosed[i] and cutsectionIsClosed[i-1]

		for j in range(0, Ny):
			quad = vtk.vtkQuad()

			if j == 0:
				if wrapAround:
					quad.GetPointIds().SetId(0, i*Ny)
					quad.GetPointIds().SetId(1, (i+1)*Ny-1)
					quad.GetPointIds().SetId(2, i*Ny-1)
					quad.GetPointIds().SetId(3, (i-1)*Ny)
			else:
				quad.GetPointIds().SetId(0, i*Ny+j)
				quad.GetPointIds().SetId(1, i*Ny+j-1)
				quad.GetPointIds().SetId(2, (i-1)*Ny + j-1)
				quad.GetPointIds().SetId(3, (i-1)*Ny + j)
			cells.InsertNextCell(quad)
	return cells

def createLineCells(Nx, Ny, **kwargs):
	''' Create lines corresponding to a Nx x Ny grid. 
	Parameters:
		:Nx (int): Number of points in the first dimension
		:Ny (int): Number of points in the second dimension
	Keyword arguments:
		:cutsectionIsClosed (bool or [bool]): Flag to indicate whether to wrap around the indices
		This is useful for creating a cell distribution for a cylinder
		Can also be an array of bools each indicating whether each of the sections in the x-direction should be wrapped 
	'''
	cells = vtk.vtkCellArray()
	cutsectionIsClosed = kwargs.get('cutsectionIsClosed', True)

	for i in range(Nx):
		if isinstance(cutsectionIsClosed, bool):
			wrapAround = cutsectionIsClosed
		else:
			# Assume array like
			wrapAround = cutsectionIsClosed[i] and cutsectionIsClosed[i-1]

		cell = vtk.vtkPolyLine()
		startId = i*Ny

		if wrapAround:
			cell.GetPointIds().SetNumberOfIds(Ny+1)
			cell.GetPointIds().SetId(Ny, startId)
		else:
			cell.GetPointIds().SetNumberOfIds(Ny)

		for j in range(Ny):
			cell.GetPointIds().SetId(j, startId+j)

		cells.InsertNextCell(cell)
	return cells

def computeCellVolumes(dataSet):
	vols = np.zeros(dataSet.GetNumberOfCells())
	for i, cell in enumerate(CellIterator(dataSet)):
		if cell.GetCellDimension() != 3:
			vols[i] = 0
		else:
			ptIds = vtk.vtkIdList()
			pts = vtk.vtkPoints()
			if cell.Triangulate(0, ptIds, pts) != 1:
				raise RuntimeError("Unable to triangulate the cell")
			for j in range(0, ptIds.GetNumberOfIds() // 4):
				vols[i] += vtk.vtkTetra.ComputeVolume(pts.GetPoint(4*j), pts.GetPoint(4*j+1), pts.GetPoint(4*j+2), pts.GetPoint(4*j+3))
	return vols

def computeCellCenters(dataSet):
	pCoords = [0, 0, 0]
	x = [0, 0, 0]
	subId = vtk.mutable(0)
	weights = []
	cellCenters = np.zeros((dataSet.GetNumberOfCells(), 3))
	for i in range(dataSet.GetNumberOfCells()):
		cell = dataSet.GetCell(i)

		# Get parametric coordinates for the cell center
		cell.GetParametricCenter(pCoords)

		if len(weights) < cell.GetNumberOfPoints():
			weights = [0 for j in range(cell.GetNumberOfPoints())]

		cell.EvaluateLocation(subId, pCoords, x, weights)
		
		cellCenters[i, :] = x
	return cellCenters
