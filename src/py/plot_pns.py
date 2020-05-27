
import vtk
import numpy as np
import time
import LinearSubdivisionFilter as lsf

import argparse
import os

import PNS
import json
import csv

import tensorflow as tf

print("Tensorflow version:", tf.__version__)

parser = argparse.ArgumentParser(description='Plot PNS results', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--json', type=str, help='JSON file created by PNS', required=True)
parser.add_argument('--labels', type=str, help='CSV with label info', default=None)
parser.add_argument('--color', type=str, help='JSON with color info for labels', default=None)

args = parser.parse_args()

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

 
with open(args.json, 'r') as jsf:
	pns_obj = json.load(jsf)

color_obj = {}
if args.color is not None:
	with open(args.color, 'r') as jsf:
		color_obj = json.load(jsf)

points_colors = None
dict_filenames = {}
if args.labels is not None:

	points_colors = vtk.vtkUnsignedCharArray()
	points_colors.SetNumberOfComponents(3)
	points_colors.SetName("Colors")

	with open(args.labels) as csvfile:
		csv_reader = csv.reader(csvfile)
		for row in csv_reader:
			prop = {}
			if color_obj is not None and color_obj[row[1]] is not None:
				prop["color"] = color_obj[row[1]]
			else:
				color_obj[row[1]] = np.random.rand(3)
				prop["color"] = color_obj[row[1]]

			prop["label"] = row[1]
			dict_filenames[row[0]] = prop

			points_colors.InsertNextTuple(prop["color"])
	

points = pns_obj["points"]
projected = pns_obj["projected"]
circle_center_v1 = pns_obj["circle_center_v1"]
angle_r1 = pns_obj["angle_r1"]
rot_mat = pns_obj["rot_mat"]


# build points & polydata from numpy_to_vtk
vtk_points = vtk.vtkPoints()
vertices = vtk.vtkCellArray()

for p in points:
	pid = vtk_points.InsertNextPoint(p)
	vertices.InsertNextCell(1)
	vertices.InsertCellPoint(pid)

poly_points = vtk.vtkPolyData()
poly_points.SetPoints(vtk_points)
poly_points.SetVerts(vertices)

if points_colors is not None:
	poly_points.GetPointData().SetScalars(points_colors)

projected_points = vtk.vtkPoints()
projected_cells = vtk.vtkCellArray()

for p, prp in zip(points, projected):

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


poly_points_mapper = vtk.vtkPolyDataMapper()
poly_points_mapper.SetInputData(poly_points)
poly_points_actor = vtk.vtkActor()
poly_points_actor.SetMapper(poly_points_mapper)
poly_points_actor.GetProperty().SetRepresentationToPoints()
poly_points_actor.GetProperty().SetColor(1.0, 0, 0)
poly_points_actor.GetProperty().SetPointSize(20)


poly_circle_mapper = vtk.vtkPolyDataMapper()
poly_circle_mapper.SetInputData(poly_circle)
poly_circle_actor = vtk.vtkActor()
poly_circle_actor.SetMapper(poly_circle_mapper)
poly_circle_actor.GetProperty().SetColor(1.0, 0.5, 0)
poly_circle_actor.GetProperty().SetLineWidth(20)


circle_center_source = vtk.vtkLineSource()
circle_center_source.SetPoint1([0, 0, 0])
circle_center_source.SetPoint2(circle_center_v1)

circle_center_source_mapper = vtk.vtkPolyDataMapper()
circle_center_source_mapper.SetInputConnection(circle_center_source.GetOutputPort())
circle_center_source_actor = vtk.vtkActor()
circle_center_source_actor.SetMapper(circle_center_source_mapper)
circle_center_source_actor.GetProperty().SetColor(0, 0, 1.0)
circle_center_source_actor.GetProperty().SetLineWidth(5)


north_source = vtk.vtkLineSource()
north_source.SetPoint1([0, 0, 0])
north_source.SetPoint2([0, 0, 1])

north_source_mapper = vtk.vtkPolyDataMapper()
north_source_mapper.SetInputConnection(north_source.GetOutputPort())
north_source_actor = vtk.vtkActor()
north_source_actor.SetMapper(north_source_mapper)
north_source_actor.GetProperty().SetColor(0.0, 1.0, 0)
north_source_actor.GetProperty().SetLineWidth(10)



projected_points_mapper = vtk.vtkPolyDataMapper()
projected_points_mapper.SetInputData(projected_poly)
projected_actor = vtk.vtkActor()
projected_actor.SetMapper(projected_points_mapper)
projected_actor.GetProperty().SetColor(1.0, 1.0, 0)
projected_actor.GetProperty().SetLineWidth(10)


icosahedron = CreateIcosahedron(1, 10)

icosahedronmapper = vtk.vtkPolyDataMapper()
icosahedronmapper.SetInputData(icosahedron)
icosahedronactor = vtk.vtkActor()
icosahedronactor.SetMapper(icosahedronmapper)
# icosahedronactor.GetProperty().SetRepresentationToWireframe()

ren1 = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren1)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

ren1.AddActor(icosahedronactor)
ren1.AddActor(poly_points_actor)
ren1.AddActor(poly_circle_actor)
ren1.AddActor(circle_center_source_actor)
ren1.AddActor(north_source_actor)
# ren1.AddActor(projected_actor)

ren1.SetBackground(0,.5,1)
ren1.ResetCamera()
ren1.GetActiveCamera().Dolly(1.2)
ren1.ResetCameraClippingRange()
renWin.SetSize(1080, 960)
renWin.Render()
iren.Start()




