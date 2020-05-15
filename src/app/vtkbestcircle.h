#ifndef VTKBESTCIRCLE_H
#define VTKBESTCIRCLE_H

#include "vtkPolyDataAlgorithm.h"
#include "vtkSmartPointer.h"

#include "bestCircleFit.h"


class vtkBestCircle
        :public vtkPolyDataAlgorithm
{
public:
    static vtkBestCircle *New();

    void SetLabels(vnl_vector< int > labels){
        m_Labels = labels;
    }

    void SetLabel(int numlabel){
        m_NumLabel = numlabel;
    }

    vnl_vector< double > GetCircleCenter(){
        return m_CircleCenter;
    }

    vnl_vector< double > GetAvg(){
        return m_Avg;
    }

    vnl_vector< double > GetDerAvg(){
        return m_DerAvg;
    }

    double* GetAvgd(){
        return m_Avg.data_block();
    }

    double GetAnglePlane(){
        return m_AnglePlane;
    }

    vector< vnl_vector< double > > GetSampleBorderPoints(){
        return m_SampleBorderPoints;
    }

    vtkSmartPointer<vtkPolyData> GetOutputCircle(){
        return m_Circle;
    }

    vtkPolyData* GetOutputTangent(){
        return m_Tangent;
    }

protected:
    vtkBestCircle();

    ~vtkBestCircle();

    // Superclass method to update the pipeline
    virtual int RequestData(vtkInformation* request,
                            vtkInformationVector** inputVector,
                            vtkInformationVector* outputVector);

    vnl_vector< int > m_Labels;
    int m_NumLabel;
    vnl_vector< double > m_CircleCenter;
    vnl_vector< double > m_Avg;
    vnl_vector< double > m_DerAvg;
    double m_AnglePlane;

    vector< vnl_vector< double > > m_SampleBorderPoints;

    vnl_matrix< double > GetRotationMatrix(vnl_vector< double > rotvect, double theta);

    vtkSmartPointer<vtkPolyData> m_Circle;
    vtkSmartPointer<vtkPolyData> m_Tangent;
};

#endif // VTKBESTCIRCLE_H
