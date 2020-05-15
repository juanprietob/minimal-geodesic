
import vtk
import numpy as np
import time
import LinearSubdivisionFilter as lsf

import argparse
import os

import math
import PNS

def normalize_points(poly, radius=1.0):
	polypoints = poly.GetPoints()
	for pid in range(polypoints.GetNumberOfPoints()):
		spoint = polypoints.GetPoint(pid)
		spoint = np.array(spoint)
		norm = np.linalg.norm(spoint)
		spoint = spoint/norm * radius
		polypoints.SetPoint(pid, spoint)
	poly.SetPoints(polypoints)
	return poly

def CreateIcosahedron(radius, sl):
	icosahedronsource = vtk.vtkPlatonicSolidSource()
	icosahedronsource.SetSolidTypeToIcosahedron()
	icosahedronsource.Update()
	icosahedron = icosahedronsource.GetOutput()
	
	subdivfilter = lsf.LinearSubdivisionFilter()
	subdivfilter.SetInputData(icosahedron)
	subdivfilter.SetNumberOfSubdivisions(sl)
	subdivfilter.Update()

	icosahedron = subdivfilter.GetOutput()
	icosahedron = normalize_points(icosahedron, radius)

	return icosahedron

pns = PNS.PNS()
# lets create dummy {x,y,z} coordinates
rand_points = np.abs([pns.NormalizeVector(r) for r in np.random.normal(size=[20, 4096])])


pns.Fit(rand_points)

# build points & polydata from numpy_to_vtk
points = vtk.vtkPoints()
vertices = vtk.vtkCellArray()

for p in pns.GetPoints(3):
	pid = points.InsertNextPoint(p)
	vertices.InsertNextCell(1)
	vertices.InsertCellPoint(pid)

poly_rand = vtk.vtkPolyData()
poly_rand.SetPoints(points)
poly_rand.SetVerts(vertices)

circle_center_v1 = pns.GetCircleCenter(3)
angle_r1 = pns.GetAngleR1(3)

rot_mat = pns.GetRotationMatrix(3)

projected_points = vtk.vtkPoints()
projected_cells = vtk.vtkCellArray()

for p, prp in zip(pns.GetPoints(3), pns.GetProjectedPoints(3)):

	pid0 = projected_points.InsertNextPoint(p)
	pid1 = projected_points.InsertNextPoint(prp)
	
	vtk_line = vtk.vtkLine()
	vtk_line.GetPointIds().SetId(0, pid0)
	vtk_line.GetPointIds().SetId(1, pid1)
	projected_cells.InsertNextCell(vtk_line)


projected_poly = vtk.vtkPolyData()
projected_poly.SetPoints(projected_points)
projected_poly.SetLines(projected_cells)

circle_points = vtk.vtkPoints()
circle_cells = vtk.vtkCellArray()
circle_radius = np.sin(angle_r1)
lastpid = -1

for t in np.arange(0., 2*np.pi, np.pi/180):

	pcircle = np.array([circle_radius*np.cos(t), circle_radius*np.sin(t), np.cos(angle_r1)])
	
	pcircle = np.matmul(pcircle, np.transpose(rot_mat))

	pid = circle_points.InsertNextPoint(pcircle)
	
	if(lastpid != -1):
		vtk_line = vtk.vtkLine()
		vtk_line.GetPointIds().SetId(0, lastpid)
		vtk_line.GetPointIds().SetId(1, pid)
		lastpid = pid

		circle_cells.InsertNextCell(vtk_line);
		
	else:
		lastpid = pid


poly_circle = vtk.vtkPolyData()
poly_circle.SetPoints(circle_points)
poly_circle.SetLines(circle_cells)


poly_rand_mapper = vtk.vtkPolyDataMapper()
poly_rand_mapper.SetInputData(poly_rand)
poly_rand_actor = vtk.vtkActor()
poly_rand_actor.SetMapper(poly_rand_mapper)
# poly_rand_actor.GetProperty().SetRepresentationToPoints()
poly_rand_actor.GetProperty().SetColor(1.0, 0, 0)
poly_rand_actor.GetProperty().SetPointSize(20)


poly_circle_mapper = vtk.vtkPolyDataMapper()
poly_circle_mapper.SetInputData(poly_circle)
poly_circle_actor = vtk.vtkActor()
poly_circle_actor.SetMapper(poly_circle_mapper)


circle_center_source = vtk.vtkLineSource()
circle_center_source.SetPoint1([0, 0, 0])
circle_center_source.SetPoint2(circle_center_v1)

circle_center_source_mapper = vtk.vtkPolyDataMapper()
circle_center_source_mapper.SetInputConnection(circle_center_source.GetOutputPort())
circle_center_source_actor = vtk.vtkActor()
circle_center_source_actor.SetMapper(circle_center_source_mapper)


north_source = vtk.vtkLineSource()
north_source.SetPoint1([0, 0, 0])
north_source.SetPoint2([0, 0, 1])

north_source_mapper = vtk.vtkPolyDataMapper()
north_source_mapper.SetInputConnection(north_source.GetOutputPort())
north_source_actor = vtk.vtkActor()
north_source_actor.SetMapper(north_source_mapper)



projected_points_mapper = vtk.vtkPolyDataMapper()
projected_points_mapper.SetInputData(projected_poly)
projected_actor = vtk.vtkActor()
projected_actor.SetMapper(projected_points_mapper)


icosahedron = CreateIcosahedron(1, 4)

icosahedronmapper = vtk.vtkPolyDataMapper()
icosahedronmapper.SetInputData(icosahedron)
icosahedronactor = vtk.vtkActor()
icosahedronactor.SetMapper(icosahedronmapper)
icosahedronactor.GetProperty().SetRepresentationToWireframe()

ren1 = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren1)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

ren1.AddActor(icosahedronactor)
ren1.AddActor(poly_rand_actor)
ren1.AddActor(poly_circle_actor)
ren1.AddActor(circle_center_source_actor)
ren1.AddActor(north_source_actor)
ren1.AddActor(projected_actor)

ren1.SetBackground(0,.5,1)
ren1.ResetCamera()
ren1.GetActiveCamera().Dolly(1.2)
ren1.ResetCameraClippingRange()
renWin.SetSize(1080, 960)
renWin.Render()
iren.Start()




