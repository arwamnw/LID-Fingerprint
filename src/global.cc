/* global.cc */

/* Implements global.h.
 * Jichao Sun (js87@njit.edu)
 *
 * June 5, 2013 Initialized based on global.cc (last modified on Mar 14,
 * 2013) from kprop.6
 */

#include "global.h"
#include <cstdlib>  /* for qsort() and exit() */
#include <cmath>    /* for acos() and sqrt() */
#include <cfloat>   /* for FLT_MAX */
#include <string>
using std::string;
#include <boost/dynamic_bitset.hpp>
using boost::dynamic_bitset;

namespace kprop
{
    const float TOLERANCE_VALUE = 1e-6F;   
    const char  WHITESPACES[]   = " \n\t\v\r\f";
}

int kprop::cmp_int_i(void const* a, void const* b)
{
    return KPROP_COMPARE(*(int*)a, *(int*)b);
}
int kprop::cmp_int_d(void const* a, void const* b)
{
    return (-1 * cmp_int_i(a, b));
}
int kprop::cmp_fl_i(void const* a, void const* b)
{
    return KPROP_COMPARE(*(float*)a, *(float*)b);
}
int kprop::cmp_fl_d(void const* a, void const* b)
{
    return (-1 * cmp_fl_i(a, b));
}
int kprop::cmp_i2pairs_i(void const* a, void const* b)
{
    i2pair* pa = (i2pair*)a;
    i2pair* pb = (i2pair*)b;
    int d = KPROP_COMPARE(pa->weight, pb->weight);
    if (d != 0) return d;
    return KPROP_COMPARE(pa->index, pb->index);
}
int kprop::cmp_i2pairs_d(void const* a, void const* b)
{
    return (-1 * cmp_i2pairs_i(a, b));
}
int kprop::cmp_i3tuples_i(void const* a, void const* b)
{
    i3tuple* pa = (i3tuple*)a;
    i3tuple* pb = (i3tuple*)b;
    int d = KPROP_COMPARE(pa->weight1, pb->weight1);
    if (d != 0) return d;
    d = KPROP_COMPARE(pa->weight2, pb->weight2);
    if (d != 0) return d;
    return KPROP_COMPARE(pa->index, pb->index);
}
int kprop::cmp_i3tuples_d(void const* a, void const* b)
{
    return (-1 * cmp_i3tuples_i(a, b));
}
int kprop::cmp_i4tuples_i(void const* a, void const* b)
{
    i4tuple* pa = (i4tuple*)a;
    i4tuple* pb = (i4tuple*)b;
    int d = KPROP_COMPARE(pa->weight1, pb->weight1);
    if (d != 0) return d;
    d = KPROP_COMPARE(pa->weight2, pb->weight2);
    if (d != 0) return d;
    d = KPROP_COMPARE(pa->weight3, pb->weight3);
    if (d != 0) return d;
    return KPROP_COMPARE(pa->index, pb->index);
}
int kprop::cmp_i4tuples_d(void const* a, void const* b)
{
    return (-1 * cmp_i4tuples_i(a, b));
}
int kprop::cmp_ifpairs_i(void const* a, void const* b)
{
    ifpair* pa = (ifpair*)a;
    ifpair* pb = (ifpair*)b;
    int d = KPROP_COMPARE(pa->weight, pb->weight);
    if (d != 0) return d;
    return KPROP_COMPARE(pa->index, pb->index);
}
int kprop::cmp_ifpairs_d(void const* a, void const* b)
{
    return (-1 * cmp_ifpairs_i(a, b));
}
int kprop::cmp_i2ftuples_i(void const* a, void const* b)
{
    i2ftuple* pa = (i2ftuple*)a;
    i2ftuple* pb = (i2ftuple*)b;
    int d = KPROP_COMPARE(pa->weight1, pb->weight1);
    if (d != 0) return d;
    d = KPROP_COMPARE(pa->weight2, pb->weight2);
    if (d != 0) return d;
    return KPROP_COMPARE(pa->index, pb->index);
}
int kprop::cmp_i2ftuples_d(void const* a, void const* b)
{
    return (-1 * cmp_i2ftuples_i(a, b));
}
int kprop::cmp_i3ftuples_i(void const* a, void const* b)
{
    i3ftuple* pa = (i3ftuple*)a;
    i3ftuple* pb = (i3ftuple*)b;
    int d = KPROP_COMPARE(pa->weight1, pb->weight1);
    if (d != 0) return d;
    d = KPROP_COMPARE(pa->weight2, pb->weight2);
    if (d != 0) return d;
    d = KPROP_COMPARE(pa->weight3, pb->weight3);
    if (d != 0) return d;
    return KPROP_COMPARE(pa->index, pb->index);
}
int kprop::cmp_i3ftuples_d(void const* a, void const* b)
{
    return (-1 * cmp_i3ftuples_i(a, b));
}
void kprop::trim(string& s)
{
    s.erase(0U, s.find_first_not_of(WHITESPACES));
    s.erase(s.find_last_not_of(WHITESPACES) + 1U);
}
float kprop::l1Dist(const float* pa, const float* pb, int size)
{
    if (pa == NULL || pb == NULL || size <= 0)
    {
        KPROP_ERROR("invalid vector(s) or size.\n");
        exit(EXIT_FAILURE);
    }
    double sum = 0.0;
    for (int i = 0; i < size; i++)
    {
        double diff = (double)pa[i] - (double)pb[i];
        sum += (diff < 0.0) ? (-diff) : diff;
    }
    return (float)sum;
}
float kprop::l2Dist(const float* pa, const float* pb, int size)
{
    if (pa == NULL || pb == NULL || size <= 0)
    {
        KPROP_ERROR("invalid vector(s) or size.\n");
        exit(EXIT_FAILURE);
    }
    double sum = 0.0;
    for (int i = 0; i < size; i++)
    {
        double diff = (double)pa[i] - (double)pb[i];
        sum += diff * diff;
    }
    return (float)sqrt(sum);
}
float kprop::l2DistPar(const float* pa, const float* pb,
                       const int* pi,   int size)
{
    if (pa == NULL || pb == NULL || pi == NULL || size <= 0)
    {
        KPROP_ERROR("invalid vector(s), indices or size.\n");
        exit(EXIT_FAILURE);
    }
    double sum = 0.0;
    for (int i = 0; i < size; i++)
    {
        int ind = pi[i];
        if (ind < 0)
        {
            KPROP_WARN("invalid index.\n");
            exit(EXIT_FAILURE);
        }
        double diff = (double)pa[ind] - (double)pb[ind];
        sum += diff * diff;
    }
    return (float)sqrt(sum);
}
float kprop::vaDist(const float* pa, const float* pb, int size)
{
    if (pa == NULL || pb == NULL || size <= 0)
    {
        KPROP_ERROR("invalid vector(s) or size.\n");
        exit(EXIT_FAILURE);
    }
    double normPa = 0.0;
    double normPb = 0.0;
    for (int i = 0; i < size; i++)
    {
        normPa += (double)pa[i] * (double)pa[i];
        normPb += (double)pb[i] * (double)pb[i];
    }
    normPa = sqrt(normPa);
    normPb = sqrt(normPb);
    double cosine = 0.0;
    for (int i = 0; i < size; i++)
    {
        cosine += (double)pa[i] * (double)pb[i];
    }
    cosine /= (normPa * normPb);
    if (cosine >= 1.0)
    {
        return 0.0F;
    }
    else if (cosine <= -1.0F)
    {
        return (float)acos(-1.0);
    }
    else
    {
        return (float)acos(cosine);
    }
}
float kprop::snnSim(const float* pa, const float* pb, int size, int length)
{
    if (pa == NULL || pb == NULL || size <= 0)
    {
        KPROP_ERROR("invalid vector(s) or size.\n");
        exit(EXIT_FAILURE);
    }
    if (length < size)
    {
        KPROP_ERROR("invalid size of the vector space.\n");
        exit(EXIT_FAILURE);
    }
    dynamic_bitset<> b1(length);
    dynamic_bitset<> b2(length);
    for (int i = 0; i < size; i++)
    {
        b1.set((int)pa[i]);
        b2.set((int)pb[i]);
    }
    b1 &= b2;
    return (float)b1.count() / (float)size;
}
