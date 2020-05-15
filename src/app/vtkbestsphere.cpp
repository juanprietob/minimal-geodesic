#include "vtkbestsphere.h"
#include "circleMean.h"

#include "vtkSmartPointer.h"


#include "vtkCellArray.h"
#include "vtkVertex.h"
#include "vtkLine.h"

#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vnl/vnl_least_squares_function.h"

#include "vnl/vnl_cross.h"
#include "vnl/vnl_inverse.h"

#include "vnl/algo/vnl_levenberg_marquardt.h"

#ifndef PI
#define PI 3.14159265
#endif

#include "vtkObjectFactory.h"
vtkStandardNewMacro(vtkBestSphere);

vtkBestSphere::vtkBestSphere()
{
}

vtkBestSphere::~vtkBestSphere()
{
}

// Superclass method to update the pipeline
int vtkBestSphere::RequestData(vtkInformation* request,
                        vtkInformationVector** inputVector,
                        vtkInformationVector* outputVector){

    vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
    vtkPolyData* input = dynamic_cast<vtkPolyData*>(vtkPolyData::SafeDownCast(inInfo->Get(vtkPolyData::DATA_OBJECT())));


    if(m_Points.size() > 0){
        BestCircleFit bestfit(m_Points.size());
        bestfit.SetPoints(m_Points);

        vnl_levenberg_marquardt levenberg(bestfit);

        vnl_vector<double> v1a(4);
        v1a.fill(1);
        v1a.normalize();

        levenberg.minimize(v1a);

        vnl_vector<double> v1(v1a.data_block(), 3);
        v1.normalize();
        double r1 = v1a[v1a.size()-1];


        vnl_vector< double > v2(3);
        v2.fill(0);
        v2[2] = 1;
        // vnl_matrix<double> rotmat = GetRotationMatrix_v1(v3, acos(dot_product(v1, v2)));
        
        vnl_matrix<double> rotmat = GetRotationMatrix(v1, v2);

        //cout<<rotmat<<endl;
        //rotmat = rotmat.transpose();

        vector< vnl_vector< double > > vnlprojected;

        for(unsigned i = 0; i < m_Points.size(); i++){
            vnl_vector< double > temp = m_Points[i] * sin(r1);
            double t1 = bestfit.GeodesicDistance(m_Points[i], v1);
            vnl_vector< double > temp1 = v1 * sin(t1  - r1);

            //cout<<"geodist= "<<bestfit.GeodesicDistance(labelpoint[i], v1)<<endl;
            //cout<<"temp= "<<temp<<endl;
            //cout<<"temp1= "<<temp1<<endl;

            vnl_vector< double > projected = (temp + temp1)/sin(bestfit.GeodesicDistance(m_Points[i], v1));
            //cout<<"gd = "<<bestfit.GeodesicDistance(labelpoint[i], v1)<<endl;
            //projected.normalize();
            projected = projected;
            vnl_vector< double > projectedrot = (projected*rotmat).normalize();
            //cout<<projectedrot<<endl;
            vnlprojected.push_back(projectedrot);

        }

        CircleMean circlemean(vnlprojected.size());
        circlemean.SetPoints(vnlprojected);

        vnl_levenberg_marquardt levenberg0(circlemean);

        vnl_vector<double> xymean(3);
        xymean.fill(1);
        xymean.normalize();

        levenberg0.minimize(xymean);

        double geodavg = 0;
        if(xymean[0] == 0){
            cout<<"multiple means?"<<endl;
            geodavg = PI/2.0;
        }else{
            geodavg = atan2(xymean[1],xymean[0]);
        }

        vnl_vector< double > avgp(3);
        avgp[0] = cos(geodavg)*sin(r1);
        avgp[1] = sin(geodavg)*sin(r1);
        avgp[2] = cos(r1);
        avgp *= rotmat.transpose();
        //cout<<"avgp "<<avgp<<endl;
        m_Avg = avgp;

        vtkInformation *outInfo = outputVector->GetInformationObject(0);
        vtkPolyData *output_circle = vtkPolyData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));
        
        vtkSmartPointer<vtkPoints> circlepoints = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkCellArray> circlecellarray = vtkSmartPointer<vtkCellArray>::New();

        double stepangle = PI/180;
        double radiuscircle = sin(r1);
        vtkIdType lastindex = -1;
        for(double t = 0; t < 2*PI; t+=stepangle){
            vnl_vector<double> pcircle(3);
            pcircle[0] = radiuscircle*cos(t);
            pcircle[1] = radiuscircle*sin(t);
            pcircle[2] = cos(r1);
            
            pcircle = pcircle*rotmat.transpose();
            // pcircle.normalize();

            vtkIdType id = circlepoints->InsertNextPoint(pcircle[0], pcircle[1], pcircle[2]);

            if(lastindex != -1){
                vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
                line->GetPointIds()->SetId(0, lastindex);
                line->GetPointIds()->SetId(1, id);
                lastindex = id;
                circlecellarray->InsertNextCell(line);
            }else{
                lastindex = id;
            }

        }

        vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
        line->GetPointIds()->SetId(0, lastindex);
        line->GetPointIds()->SetId(1, 0);
        circlecellarray->InsertNextCell(line);

        output_circle->SetPoints(circlepoints);
        output_circle->SetLines(circlecellarray);
        
    }
}

vnl_matrix< double > vtkBestSphere::GetRotationMatrix(vnl_vector< double > aa, vnl_vector< double > bb){

    vnl_matrix< double > a(aa.data_block(), aa.size(), 1);
    vnl_matrix< double > b(bb.data_block(), bb.size(), 1);

    vnl_matrix< double > c = (b - a*(a.transpose()*b))/(b - a*(a.transpose()*b)).flatten_row_major().magnitude();


    vnl_matrix< double > A = a*c.transpose() - c*a.transpose();

    double theta = acos((a.transpose()*b).data_block()[0]);

    vnl_matrix< double > I(A);
    I.set_identity();

    return I + sin(theta)*A + (cos(theta) - 1)*(a*a.transpose() + c*c.transpose());
}


// vnl_matrix< double > vtkBestSphere::GetRotationMatrix_v1(vnl_vector< double > rotvect, double theta){
//     vnl_matrix< double > mat(3, 3);

//     double ux = rotvect[0];
//     double uy = rotvect[1];
//     double uz = rotvect[2];

//     double costheta = cos(theta);
//     double sintheta = sin(theta);

//     mat(0,0) = costheta + (ux*ux)*(1-costheta);
//     mat(0,1) = ux*uy*(1-costheta)-uz*sintheta;
//     mat(0,2) = ux*uz*(1-costheta)+uy*sintheta;
//     mat(1,0) = uy*ux*(1-costheta)+uz*sintheta;
//     mat(1,1) = costheta+(uy*uy)*(1-costheta);
//     mat(1,2) = uy*uz*(1-costheta)-ux*sintheta;
//     mat(2,0) = uz*ux*(1-costheta)-uy*sintheta;
//     mat(2,1) = uz*uy*(1-costheta)+ux*sintheta;
//     mat(2,2) = costheta+(uz*uz)*(1-costheta);

//     return mat;
// }
