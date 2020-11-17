/* franc.h */

/* Some common functions used in franc
 * (must be included in a main function file no more than one time)
 * 
 * Jichao Sun (js87@njit.edu)
 *
 * Oct 4, 2013
 */

#ifndef FRANC_H
#define FRANC_H

#include <vector>
#include <sys/time.h>
#include <cstdio>
#include <cstdlib>
#include <cfloat>    /* for FLT_MAX */
using namespace std;

#include "DenMatSin.h"
#include "global.h"
using namespace kprop;

#include "nndes/nndes.h"
#include "nndes/nndes-data.h"
using namespace nndes;

/*************** compute correctness ********************/
/* for true KNN graph */
float compCorrectness(const DenMatSin& knns, const int* lab,
                      int nr_data, int K,
                      DenMatSin& nrCorrNeighbors, int iter)
{
    int nr_corr = 0;
    for (int n = 0; n < nr_data; ++n)
    {
        int i = n;
        for (int k = 1; k <= K; ++k)
        {
            int j = (int) knns(i, k);
            if (lab[i] == lab[j])
            {
                nr_corr ++;
                nrCorrNeighbors(iter, n) += 1.0F;
            }
        }
    }
    return (float) nr_corr / (float) (nr_data*K);
}
/* for estimated KNN graph */
float compCorrectness(const vector<KNN>& knns, const int* lab,
                      int nr_data, int K,
                      DenMatSin& nrCorrNeighbors, int iter)
{
    int nr_corr = 0;
    for (int n = 0; n < nr_data; ++n)
    {
        int i = n;
        for (int k = 0; k < K; ++k)
        {
            int j = knns[i][k].key;
            if (lab[i] == lab[j])
            {
                nr_corr ++;
                nrCorrNeighbors(iter, n) += 1.0F;
            }
        }
    }

    return (float) nr_corr / (float) (nr_data*K);
}
                      
/*************** compute edge precision ********************/
/* for true KNN graph */
float compEdgePrecision(const DenMatSin& knns,
        const int* lab, int nr_data, int K)
{
    int nr_edges = 0;
    int nr_corr_edges = 0;
    DenMatSin comp_flag(nr_data, nr_data); /* all zero: unchecked */

    for (int n = 0; n < nr_data; ++n)
    {
        int i = n;
        int lab_i = lab[i];
        for (int k = 1; k <= K; ++k)
        {
            int j = (int) knns(i,k);
            printf("%d\n", j);
            if (! comp_flag.isSet(i,j)) /* edge not checked */
            {
                nr_edges++;
                int lab_j = lab[j];
                if (lab_i == lab_j)
                {
                    nr_corr_edges++;
                }
                //printf("%d\n", i);
                //printf("%d\n", j);
                comp_flag.set(i,j);
                comp_flag.set(j,i);
            }
        }
    }

    return (float) nr_corr_edges / (float) nr_edges;
}
/* for estimated KNN graph */
float compEdgePrecision(const vector<KNN>& knns, 
        const int* lab, int nr_data, int K)
{
    int nr_edges = 0;
    int nr_corr_edges = 0;
    DenMatSin comp_flag(nr_data, nr_data); /* all zero: unchecked */

    for (int n = 0; n < nr_data; ++n)
    {
        int i = n;
        int lab_i = lab[i];
        for (int k = 0; k < K; ++k)
        {
            int j = knns[i][k].key;
            if (! comp_flag.isSet(i,j)) /* edge not checked */
            {
                nr_edges++;
                int lab_j = lab[j];
                if (lab_i == lab_j)
                {
                    nr_corr_edges++;
                }
                comp_flag.set(i,j);
                comp_flag.set(j,i);
            }
        }
    }

    return (float) nr_corr_edges / (float) nr_edges;
}

/***********************************************************/

bool validateKnn(const vector<KNN>& knns)
{
    int nr_data = (int) knns.size();
    int K = (int) knns[0].size();

    for (int n = 0; n < nr_data; ++n)
    {
        if ( (int) knns[n].size() != K )
            return false;
        int i = n;
        for (int k = 0; k < K; ++k)
        {
            int j = knns[n][k].key;
            if (j == KNN::Element::BAD || j == i)
                return false;
            for (int k2 = k+1; k2 < K; ++k2)
            {
                int j2 = knns[n][k2].key;
                if (j2 == KNN::Element::BAD || j2 == j)
                    return false;
            }
        }
    }
    return true;
}

float compMeanEdgeDist(const vector<KNN>& knns)
{
    int nr_data = (int) knns.size();
    int K       = (int) knns[0].size();
    assert(nr_data > 0);
    /* counted flag */
    DenMatSin counted_flag(nr_data, nr_data);
    
    int nr_edges = 0;
    float sum_dist = 0.0F;
    for (int n = 0; n < nr_data; ++n)
    {
        assert(K == (int) knns[n].size());
        for (int k = 0; k < K; ++k)
        {
            int i = n;
            int j = knns[n][k].key;
            float dist = knns[n][k].dist;
            if (! counted_flag.isSet(i,j))
            {
                nr_edges ++;
                sum_dist += dist;
                counted_flag.set(i,j);
                counted_flag.set(j,i);
            }
        }
    }
    return sum_dist / (float)nr_edges;
}

float compMeanEdgeDist(const DenMatSin& knns1, const DenMatSin& knns2, int K)
{
    int nr_data = knns1.getM();
    assert(knns2.getM() == nr_data);
    assert(nr_data > 0);
    /* counted flag */
    DenMatSin counted_flag(nr_data, nr_data);
    
    int nr_edges = 0;
    float sum_dist = 0.0F;
    for (int n = 0; n < nr_data; ++n)
    {
        for (int k = 1; k <= K; ++k)
        {
            int i = n;
            int j = (int) knns1(n,k);
            float dist = knns2(n,k);
            if (! counted_flag.isSet(i,j))
            {
                nr_edges ++;
                sum_dist += dist;
                counted_flag.set(i,j);
                counted_flag.set(j,i);
            }
        }
    }
    return sum_dist / (float)nr_edges;
}

void compMaxMinEdgeDist(const vector<KNN>& knns, float* max, float* min)
{
    int nr_data = (int) knns.size();
    int K       = (int) knns[0].size();
    assert(nr_data > 0);

    (*max) = -FLT_MAX;
    (*min) = FLT_MAX;
    for (int n = 0; n < nr_data; ++n)
    {
        for (int k = 0; k < K; ++k)
        {
            float dist = knns[n][k].dist;
            if (dist > (*max))
                (*max) = dist;
            if (dist < (*min))
                (*min) = dist;
        }
    }
    assert( *max > *min);
}

/* compute a LLS for dimension 'dim' of an item 'pivot'
 * uses non-weighted variance
 * user provides t value, e.g., 1.0F, 2*sigma*sigma, where
 * sigma is the global mean of graph edge distances
 * for simplicity, regard var = 1.0F always
 */
float llsLite(int pivot, int dim,
                  const KNN & knn, const Dataset<float> & data,
                  float t)
{
    float var = 1.0F;
    int K = (int) knn.size();
    float sum = 0.0F; /* numerator */
    int i = pivot;

    for (int k = 0; k < K; ++k) 
    {
        int       j = knn[k].key;
        float   Sij = (float) exp (- knn[k].dist * knn[k].dist / t);
        float fi_fj = data[i][dim] - data[j][dim]; /* f_i - f_j */

        sum += fi_fj * fi_fj * Sij;
    }
    return sum / var;
}


float llsLite(int pivot, int dim,
                  const KNN & knn, const Dataset<float> & data,
                  const DenMatSin& dim_stats,
                  float t)
{
    float var = dim_stats(dim, 1);
    
    if ( (float)fabs(var) < TOLERANCE_VALUE ) /* var == 0, means all features
                                                 in this dim are the same,
                                                 a meaningless feature and
                                                 should be removed/sparsified */
    {
        assert(0); /*  I think should not be here */
        return FLT_MAX;
    }

    int K = (int) knn.size();
    float sum = 0.0F; /* numerator */
    int i = pivot;

    for (int k = 0; k < K; ++k) 
    {
        int       j = knn[k].key;
        float   Sij = (float) exp (- knn[k].dist * knn[k].dist / t);
        float fi_fj = data[i][dim] - data[j][dim]; /* f_i - f_j */

        sum += fi_fj * fi_fj * Sij;
    }
    return sum / var;
}

float llsLite(int pivot, int dim,
                  const KNN & knn, const Dataset<float> & data,
                  float max_dist, float min_dist)
{
    float var = 1.0F;
    int K = (int) knn.size();
    float sum = 0.0F; /* numerator */
    int i = pivot;

    for (int k = 0; k < K; ++k) 
    {
        int       j = knn[k].key;
        // float   Sij = (float) exp (- knn[k].dist * knn[k].dist / t);
        float Sij = 1.0F - (knn[k].dist - min_dist) / (max_dist - min_dist);
        float fi_fj = data[i][dim] - data[j][dim]; /* f_i - f_j */

        sum += fi_fj * fi_fj * Sij;
    }
    return sum / var;
}

class Timer
{
    struct  timeval start; 
public:
    Timer () {}
    /// Start timing.
    void tick ()
    {
        gettimeofday(&start, 0); 
    }
    /// Stop timing, return the time passed (in second).
    float tuck (const char *msg) const
    {
        struct timeval end;
        float   diff; 
        gettimeofday(&end, 0); 

        diff = (end.tv_sec - start.tv_sec) 
                + (end.tv_usec - start.tv_usec) * 0.000001; 
        if (msg != 0) 
        {
            std::cout << msg << ':' <<  diff << std::endl;
        }
        return diff;
    }
};

/* for random shuffle */
int myrandom(int i)
{
    return rand()%i;
}

#endif /* #ifndef FRANC_H */
