#include "circleMean.h"

#include "iostream"

#include "vtkIdList.h"

using namespace std;

CircleMean::CircleMean(int npoints)
    : vnl_least_squares_function(3, npoints,no_gradient)
{

}

CircleMean::~CircleMean()
{
    m_Points.clear();
}

void CircleMean::f(vnl_vector< double > const &x, vnl_vector< double > &fx){


    vnl_vector< double > xp(x.data_block(), x.size());
    for(unsigned i = 0; i < m_Points.size(); i++){
        fx[i] = GeodesicDistance(m_Points[i], xp);
    }

}

double CircleMean::GeodesicDistance(vnl_vector< double > x, vnl_vector< double > y){

    x.normalize();
    y.normalize();

    return acos(dot_product(x, y));

}
