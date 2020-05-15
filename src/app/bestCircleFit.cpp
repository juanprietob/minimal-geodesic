#include "bestCircleFit.h"

#include "iostream"

#include "vtkIdList.h"

using namespace std;

BestCircleFit::BestCircleFit(int npoints)
    : vnl_least_squares_function(4, npoints,no_gradient)
{

}

BestCircleFit::~BestCircleFit()
{
    m_Points.clear();
}

void BestCircleFit::f(vnl_vector< double > const &x, vnl_vector< double > &fx){


    vnl_vector< double > xp(x.data_block(), x.size() - 1);
    double r1 = x[x.size()-1];
    for(unsigned i = 0; i < m_Points.size(); i++){
        fx[i] = GeodesicDistance(m_Points[i], xp) - r1;
    }

}

double BestCircleFit::GeodesicDistance(vnl_vector< double > x, vnl_vector< double > y){

    x.normalize();
    y.normalize();

    return acos(dot_product(x, y));

}
