
#include "pnsCLP.h"

#include <vtkLinearSubdivisionFilter.h>
#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkPoints.h>
#include <vtkIdList.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkLine.h>
#include <vtkLineSource.h>
#include <vtkExtractCells.h>
#include <vtkOBBTree.h>
#include <vtkCellLocator.h>
#include <vtkPlatonicSolidSource.h>
#include <vtkVertex.h>
#include <vtkPlaneSource.h>
#include <vtkPolyDataNormals.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataMapper.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>

#include <vnl/vnl_vector.h>
#include <vnl/vnl_cross.h>

// #include <itkVectorImage.h>
// #include <itkImageFileWriter.h>
// #include <itkImageRegionIterator.h>
// #include <itkComposeImageFilter.h>

#include "vtkbestsphere.h"

#include <random>
#include <math.h>

// typedef double VectorImagePixelType;
// typedef itk::VectorImage<VectorImagePixelType, 2> VectorImageType;  
// typedef VectorImageType::Pointer VectorImagePointerType;  
// typedef itk::ImageRegionIterator<VectorImageType> VectorImageIteratorType;
// typedef itk::ImageFileWriter<VectorImageType> VectorImageFileWriterType;

// typedef itk::VectorImage<VectorImagePixelType, 3> VectorImageComposeType; 
// typedef itk::ImageRegionIterator<VectorImageComposeType> VectorImageComposeIteratorType;
// typedef itk::ImageFileWriter<VectorImageComposeType> VectorImageComposeFileWriterType; 

using namespace std;

int main(int argc, char * argv[])
{
  
	PARSE_ARGS;

	vtkSmartPointer<vtkPolyData> sphere;
	
	vtkSmartPointer<vtkPlatonicSolidSource> icosahedron_source = vtkSmartPointer<vtkPlatonicSolidSource>::New();
	icosahedron_source->SetSolidTypeToIcosahedron();
	icosahedron_source->Update();

	vtkSmartPointer<vtkLinearSubdivisionFilter> subdivision = vtkSmartPointer<vtkLinearSubdivisionFilter>::New();
	subdivision->SetInputData(icosahedron_source->GetOutput());
	subdivision->SetNumberOfSubdivisions(4);
	subdivision->Update();
	sphere = subdivision->GetOutput();

	for(unsigned i = 0; i < sphere->GetNumberOfPoints(); i++){
	  double point[3];
	  sphere->GetPoints()->GetPoint(i, point);
	  vnl_vector<double> v = vnl_vector<double>(point, 3);
	  v = v.normalize();
	  sphere->GetPoints()->SetPoint(i, v.data_block());
	}
  

	// cout<<"Reading: "<<inputSurface<<endl;
	// vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
	// reader->SetFileName(inputSurface.c_str());
	// reader->Update();
	// vtkSmartPointer<vtkPolyData> input_mesh = reader->GetOutput();

	// vtkSmartPointer<vtkPolyDataNormals> normalGenerator = vtkSmartPointer<vtkPolyDataNormals>::New();
	// normalGenerator->SetInputData(input_mesh);
	// normalGenerator->ComputePointNormalsOn();
	// normalGenerator->ComputeCellNormalsOn();
	// normalGenerator->Update();

	// input_mesh = normalGenerator->GetOutput();

	vtkSmartPointer<vtkPolyData> random_mesh = vtkSmartPointer<vtkPolyData>::New();
	vtkSmartPointer<vtkPoints> random_points = vtkSmartPointer<vtkPoints>::New();
	vtkSmartPointer<vtkCellArray> random_cells = vtkSmartPointer<vtkCellArray>::New();

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0, 1.0);

	vector< vnl_vector<double> > vect_points;

	for (int i = 0; i < 100; i++){
		vnl_vector<double> p(3);
		for(int j = 0; j < 3; j++){
			double number = distribution(generator);
			p[j] = abs(number);
		}
		p.normalize();
		vect_points.push_back(p);
		vtkIdType pid = random_points->InsertNextPoint(p.data_block());
		vtkSmartPointer<vtkVertex> vertex = vtkSmartPointer<vtkVertex>::New();
		vertex->GetPointIds()->SetId(0, pid);
		random_cells->InsertNextCell(vertex);
	}

	random_mesh->SetPoints(random_points);
	random_mesh->SetVerts(random_cells);


	vtkSmartPointer<vtkBestSphere> bestsphere = vtkSmartPointer<vtkBestSphere>::New();
    bestsphere->SetInputData(random_mesh);
    bestsphere->SetPoints(vect_points);
    bestsphere->Update();

    vtkSmartPointer<vtkPolyData> best_circle = bestsphere->GetOutput();

    // m_S0 = bestsphere->GetAvg()*r0;

	vtkSmartPointer<vtkRenderer> renderer;
	vtkSmartPointer<vtkRenderWindow> renderWindow;
	vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor;

	vtkSmartPointer<vtkPolyDataMapper> random_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	random_mapper->SetInputData(random_mesh);
	vtkSmartPointer<vtkActor> random_actor = vtkSmartPointer<vtkActor>::New();
	random_actor->SetMapper(random_mapper);
	random_actor->GetProperty()->SetRepresentationToPoints();
	random_actor->GetProperty()->SetColor(0, 1, 0);
	random_actor->GetProperty()->SetPointSize(10);

	vtkSmartPointer<vtkPolyDataMapper> best_circle_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	best_circle_mapper->SetInputData(best_circle);
	vtkSmartPointer<vtkActor> best_circle_actor = vtkSmartPointer<vtkActor>::New();
	best_circle_actor->SetMapper(best_circle_mapper);

	vtkSmartPointer<vtkPolyDataMapper> sphereMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	sphereMapper->SetInputData(sphere);
	vtkSmartPointer<vtkActor> sphereActor = vtkSmartPointer<vtkActor>::New();
	sphereActor->SetMapper(sphereMapper);
	// sphereActor->GetProperty()->SetRepresentationToWireframe();
	sphereActor->GetProperty()->SetColor(0.89,0.81,0.34);

	renderer = vtkSmartPointer<vtkRenderer>::New();
	renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->AddRenderer(renderer);
	renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	renderWindowInteractor->SetRenderWindow(renderWindow);

	renderer->AddActor(sphereActor);
	renderer->AddActor(random_actor);
	renderer->AddActor(best_circle_actor);

	renderer->SetBackground(.4, .5, .6);
	renderWindow->Render();
	renderWindowInteractor->Start();


	// vtkSmartPointer<vtkLineSource> lineSource = vtkSmartPointer<vtkLineSource>::New();
	// lineSource->SetPoint1(point_plane_v.data_block());
	// lineSource->SetPoint2(point_end_v.data_block());

	// vtkSmartPointer<vtkPolyDataMapper> lineMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	// lineMapper->SetInputConnection(lineSource->GetOutputPort());
	// vtkSmartPointer<vtkActor> lineActor = vtkSmartPointer<vtkActor>::New();
	// lineActor->SetMapper(lineMapper);

	// vtkSmartPointer<vtkPolyDataMapper> planeMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	// planeMapper->SetInputData(planeMesh);
	// vtkSmartPointer<vtkActor> planeActor = vtkSmartPointer<vtkActor>::New();
	// planeActor->SetMapper(planeMapper);
	// planeActor->GetProperty()->SetRepresentationToWireframe();

	// renderer->AddActor(lineActor);
	// renderer->AddActor(planeActor);

	// renderWindowInteractor->Start();
	// renderer->RemoveActor(planeActor);
	// renderer->RemoveActor(lineActor);

	// vtkSmartPointer< vtkBestCircle > findbestcircle = vtkSmartPointer< vtkBestCircle >::New();
	// findbestcircle->SetInputData(spherereg);
	// findbestcircle->SetLabels(vectlabel);
	// findbestcircle->SetLabel(corpuslabel);
	// findbestcircle->Update();




    

	return EXIT_SUCCESS;
}