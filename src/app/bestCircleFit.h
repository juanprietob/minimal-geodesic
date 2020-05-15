#ifndef BestCircleFit_H
#define BestCircleFit_H

#include "vnl/vnl_least_squares_function.h"

#include <vector>

using namespace std;

class BestCircleFit : public vnl_least_squares_function
{
public:    
    BestCircleFit(int npoints);
    ~BestCircleFit();

    virtual void f(vnl_vector< double > const &x, vnl_vector< double > &fx);


    void SetPoints(vector< vnl_vector< double > > points){
        m_Points = points;
    }

    double GeodesicDistance(vnl_vector< double > x, vnl_vector< double > y);

private:

    vector< vnl_vector<double> > m_Points;

};

#endif // BestCircleFit_H
