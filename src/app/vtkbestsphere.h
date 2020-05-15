#ifndef vtkBestSphere_H
#define vtkBestSphere_H

#include "vtkPolyDataAlgorithm.h"
#include "vtkSmartPointer.h"

#include "bestCircleFit.h"


class vtkBestSphere
        :public vtkPolyDataAlgorithm
{
public:
    static vtkBestSphere *New();

    vnl_vector< double > GetAvg(){
        return m_Avg;
    }

    void SetPoints(vector< vnl_vector< double > > points){
        m_Points = points;
    }

protected:
    vtkBestSphere();

    ~vtkBestSphere();


    // Superclass method to update the pipeline
    virtual int RequestData(vtkInformation* request,
                            vtkInformationVector** inputVector,
                            vtkInformationVector* outputVector);


    // vnl_matrix< double > GetRotationMatrix_v1(vnl_vector< double > rotvect, double theta);
    vnl_matrix< double > GetRotationMatrix(vnl_vector< double > a, vnl_vector< double > b);

    vector< vnl_vector< double > > m_Points;
    vnl_vector< double > m_Avg;
};

#endif // vtkBestSphere_H
