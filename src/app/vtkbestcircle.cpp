#include "vtkbestcircle.h"
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
vtkStandardNewMacro(vtkBestCircle);

vtkBestCircle::vtkBestCircle()
{
    m_NumLabel = -1;
}

vtkBestCircle::~vtkBestCircle()
{
}

// Superclass method to update the pipeline
int vtkBestCircle::RequestData(vtkInformation* request,
                        vtkInformationVector** inputVector,
                        vtkInformationVector* outputVector){


    // get the info objects
    vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
    // get the input and output
    vtkPolyData* input = dynamic_cast<vtkPolyData*>(vtkPolyData::SafeDownCast(inInfo->Get(vtkPolyData::DATA_OBJECT())));

    vtkInformation *outInfo = outputVector->GetInformationObject(0);
    vtkPolyData *output = vtkPolyData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));


    vtkSmartPointer<vtkCellArray> interpolatedcellarray = vtkSmartPointer<vtkCellArray>::New();
    vtkSmartPointer<vtkPoints> interpolatedpoints = vtkSmartPointer<vtkPoints>::New();

    vector< vnl_vector< double > > labelpoint;

    vector< vnl_vector< double > > labelpointborder;

    double radius = 1;

    for(unsigned i = 0; i < m_Labels.size();i++){
        double val[1];
        val[0] = m_Labels[i];

        if(val[0]==m_NumLabel){

            double temp[3];
            input->GetPoint(i, temp);

            vnl_vector< double > vnlpoint(temp, 3);

            radius = vnlpoint.magnitude();

            labelpoint.push_back(vnlpoint.normalize());

            vtkSmartPointer<vtkIdList> cellsid = vtkSmartPointer<vtkIdList>::New();
            input->GetPointCells(i, cellsid);

            for(unsigned j = 0; j < cellsid->GetNumberOfIds(); j++){
                vtkSmartPointer<vtkIdList> pointsid = vtkSmartPointer<vtkIdList>::New();
                input->GetCellPoints(cellsid->GetId(j), pointsid);
                for(unsigned k = 0; k < pointsid->GetNumberOfIds(); k++){
                    if(m_Labels[pointsid->GetId(k)] != m_NumLabel){
                        double temp[3];
                        input->GetPoint(i, temp);
                        vnl_vector< double > vnlpoint(temp, 3);

                        labelpointborder.push_back(vnlpoint.normalize());


                        k = pointsid->GetNumberOfIds();
                    }
                }
            }

            //cout<<point<<endl;
        }
    }

    BestCircleFit bestfit(labelpoint.size());
    bestfit.SetPoints(labelpoint);

    vnl_levenberg_marquardt levenberg(bestfit);

    vnl_vector<double> v1a(4);
    v1a.fill(1);
    v1a.normalize();

    levenberg.minimize(v1a);

    vnl_vector<double> v1(v1a.data_block(), 3);
    v1.normalize();
    double r1 = v1a[v1a.size()-1];

    //vector< vnl_vector< double > > labelpointprojected;


    vnl_vector< double > v2(3);
    v2.fill(0);
    v2[2] = 1;
    vnl_vector< double > v3 = vnl_cross_3d(v2, v1);
    v3.normalize();
    vnl_matrix<double> rotmat = GetRotationMatrix(v3, acos(dot_product(v1, v2)));
    //cout<<rotmat<<endl;
    //rotmat = rotmat.transpose();

    vector< vnl_vector< double > > vnlprojected;

    for(unsigned i = 0; i < labelpoint.size(); i++){
        vnl_vector< double > temp = labelpoint[i] * sin(r1);
        double t1 = bestfit.GeodesicDistance(labelpoint[i], v1);
        vnl_vector< double > temp1 = v1 * sin(t1  - r1);

        //cout<<"geodist= "<<bestfit.GeodesicDistance(labelpoint[i], v1)<<endl;
        //cout<<"temp= "<<temp<<endl;
        //cout<<"temp1= "<<temp1<<endl;

        vnl_vector< double > projected = (temp + temp1)/sin(bestfit.GeodesicDistance(labelpoint[i], v1));        
        //cout<<"gd = "<<bestfit.GeodesicDistance(labelpoint[i], v1)<<endl;
        //projected.normalize();
        projected = projected*radius;
        vtkSmartPointer< vtkVertex > vertex = vtkSmartPointer< vtkVertex >::New();
        vertex->GetPointIds()->SetId(0, interpolatedpoints->InsertNextPoint(projected[0], projected[1], projected[2]));

        interpolatedcellarray->InsertNextCell(vertex);
        vnl_vector< double > projectedrot = (projected*rotmat).normalize();
        //cout<<projectedrot<<endl;
        vnlprojected.push_back(projectedrot);

    }

    output->SetPoints(interpolatedpoints);
    output->SetVerts(interpolatedcellarray);

    /*vnl_vector<double> xymean(2);
    xymean.fill(0);

    vnl_vector<double> xymin(2);
    xymin.fill(999999);

    vnl_vector<double> xymax(2);
    xymax.fill(-999999);
    for(unsigned i = 0; i < vnlprojected.size(); i++){
        //geodavg = geodavg + bestfit.GeodesicDistance(vnlprojected[i], vnlprojected[i+1]);
        xymean[0] += vnlprojected[i][0];
        xymean[1] += vnlprojected[i][1];

        if(xymin[0] > vnlprojected[i][0]){
            xymin[0] = vnlprojected[i][0];
        }
        if(xymin[1] > vnlprojected[i][1]){
            xymin[1] = vnlprojected[i][1];
        }

        if(xymax[0] < vnlprojected[i][0]){
            xymax[0] = vnlprojected[i][0];
        }
        if(xymax[1] < vnlprojected[i][1]){
            xymax[1] = vnlprojected[i][1];
        }
    }*/
    //xymean /= vnlprojected.size();
    //xymean = (xymax + xymin)/2.0;

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


    m_CircleCenter = -v1;//*rotmat.transpose();
    m_CircleCenter.normalize();
    //cout<<m_Avg<<endl;
    //get the derivative while is projected in the plane
    m_DerAvg.set_size(3);
    m_DerAvg[0] = -sin(geodavg);
    m_DerAvg[1] = cos(geodavg);
    m_DerAvg[2] = 0;

    vnl_vector<double> xDir(3);
    xDir.fill(0);
    xDir[0] = 1;
    m_DerAvg.normalize();
    m_DerAvg = m_DerAvg*rotmat.transpose();    

    //m_Avg = m_Avg*radius;

    vtkSmartPointer<vtkPoints> tangentpoints = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> tangentlinecellarray = vtkSmartPointer<vtkCellArray>::New();
    vtkSmartPointer<vtkLine> tangentline = vtkSmartPointer<vtkLine>::New();

    vtkIdType t0 = tangentpoints->InsertNextPoint(m_Avg[0]*radius, m_Avg[1]*radius, m_Avg[2]*radius);
    tangentline->GetPointIds()->SetId(0, t0);
    tangentline->GetPointIds()->SetId(1, tangentpoints->InsertNextPoint(m_Avg[0]*radius + m_DerAvg[0]*20, m_Avg[1]*radius + m_DerAvg[1]*20, m_Avg[2]*radius + m_DerAvg[2]*20));
    tangentlinecellarray->InsertNextCell(tangentline);

    vtkSmartPointer<vtkCellArray> tangentlinevertcellarray = vtkSmartPointer<vtkCellArray>::New();
    vtkSmartPointer<vtkVertex> tangentvert = vtkSmartPointer<vtkVertex>::New();
    tangentlinevertcellarray->InsertNextCell(tangentvert);

    m_Tangent = vtkSmartPointer<vtkPolyData>::New();
    m_Tangent->SetPoints(tangentpoints);
    m_Tangent->SetLines(tangentlinecellarray);
    m_Tangent->SetVerts(tangentlinevertcellarray);



    //cout<<m_DerAvg<<endl;
    //cout<<m_DerAvg<<endl;
    double northangle = bestfit.GeodesicDistance(v2, m_CircleCenter);
    vnl_matrix< double > rotnorth = GetRotationMatrix(vnl_cross_3d(v2, m_CircleCenter).normalize(),northangle);
    m_DerAvg = m_DerAvg*rotnorth;
    m_AnglePlane = atan2(m_DerAvg[1],m_DerAvg[0]);


    vnlprojected.clear();
    labelpoint.clear();


    //Draw circle

    m_Circle = vtkSmartPointer<vtkPolyData>::New();
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

        pcircle *= radius;
        pcircle = pcircle*rotmat.transpose();

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

    m_Circle->SetPoints(circlepoints);
    m_Circle->SetLines(circlecellarray);




    //end bestcircle







    vector< vnl_vector< double > > vnlprojectedrotborder;
    vector< vnl_vector< double > > vnlrotborder;

    double minangle = VTK_DOUBLE_MAX;
    double maxangle = VTK_DOUBLE_MIN;

    for(unsigned i = 0; i < labelpointborder.size(); i++){

        vnl_vector< double > temp = labelpointborder[i] * sin(r1);
        double t1 = bestfit.GeodesicDistance(labelpointborder[i], v1);
        vnl_vector< double > temp1 = v1 * sin(t1  - r1);
        vnl_vector< double > projected = (temp + temp1)/sin(bestfit.GeodesicDistance(labelpoint[i], v1));
        projected = projected*radius;
        vnl_vector< double > projectedrot = (projected*rotmat).normalize();

        double tempangle = atan2(projectedrot[1], projectedrot[0]);
        if(minangle > tempangle){
            minangle = tempangle;
        }
        if(maxangle < tempangle){
            maxangle = tempangle;
        }

        vnlprojectedrotborder.push_back(projectedrot);
        vnlrotborder.push_back((labelpointborder[i]*rotmat).normalize());
    }

    vector< vnl_vector< double > > closestinside;
    vector< vnl_vector< double > > closestoutside;

    vnl_vector<double> cross = vnl_cross_3d(v2, m_Avg);
    double angle = acos(dot_product(v2, m_Avg));

    vnl_matrix<double> rotmatnorth = GetRotationMatrix(cross, angle);
    vnl_matrix<double> rotmatplane = GetRotationMatrix(v2, m_AnglePlane);

    for(double rad = minangle; rad < maxangle; rad += PI/64.0){

        double currentinside = VTK_DOUBLE_MAX;
        double currentoutside = VTK_DOUBLE_MAX;
        unsigned indexinside = 0;
        unsigned indexoutside = 0;

        for(unsigned i = 0; i < vnlprojectedrotborder.size(); i++){
            double tempangle = atan2(vnlprojectedrotborder[i][1], vnlprojectedrotborder[i][0]);

            //cout<<vnlrotborder[i]<<endl;
            //cout<<vnlprojectedrotborder[i]<<endl;
            double magproj = sqrt(pow(vnlprojectedrotborder[i][0], 2) + pow(vnlprojectedrotborder[i][1], 2));
            double magrot = sqrt(pow(vnlrotborder[i][0], 2) + pow(vnlrotborder[i][1], 2));

            if(fabs(tempangle - rad) < currentinside && magrot < magproj){
                currentinside = fabs(tempangle - rad);
                indexinside = i;
            }
            if(fabs(tempangle - rad) < currentoutside && magrot > magproj){
                currentoutside = fabs(tempangle - rad);
                indexoutside = i;
            }
        }

        closestinside.push_back(((vnlrotborder[indexinside]*rotmat.transpose())*rotmatnorth)*rotmatplane*100);
        closestoutside.push_back(((vnlrotborder[indexoutside]*rotmat.transpose())*rotmatnorth)*rotmatplane*100);
    }


    if(this->GetDebug()){
        for(unsigned i = 0; i < closestinside.size(); i++){
            cout<<closestinside[i]<<endl;
        }
        cout<<"out"<<endl;
        for(unsigned i = 0; i < closestoutside.size(); i++){
            cout<<closestoutside[i]<<endl;
        }
    }

    m_SampleBorderPoints.insert(m_SampleBorderPoints.begin(), closestoutside.begin(), closestoutside.end());
    m_SampleBorderPoints.insert(m_SampleBorderPoints.end(), closestinside.rbegin(), closestinside.rend());

    return 1;

}


vnl_matrix< double > vtkBestCircle::GetRotationMatrix(vnl_vector< double > rotvect, double theta){
    vnl_matrix< double > mat(3, 3);

    double ux = rotvect[0];
    double uy = rotvect[1];
    double uz = rotvect[2];

    double costheta = cos(theta);
    double sintheta = sin(theta);

    mat(0,0) = costheta + (ux*ux)*(1-costheta);
    mat(0,1) = ux*uy*(1-costheta)-uz*sintheta;
    mat(0,2) = ux*uz*(1-costheta)+uy*sintheta;
    mat(1,0) = uy*ux*(1-costheta)+uz*sintheta;
    mat(1,1) = costheta+(uy*uy)*(1-costheta);
    mat(1,2) = uy*uz*(1-costheta)-ux*sintheta;
    mat(2,0) = uz*ux*(1-costheta)-uy*sintheta;
    mat(2,1) = uz*uy*(1-costheta)+ux*sintheta;
    mat(2,2) = costheta+(uz*uz)*(1-costheta);

    return mat;
}
