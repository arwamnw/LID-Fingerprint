/* global.h */

/* Defines common macros, data types, constants and functions.
 * Jichao Sun (js87@njit.edu)
 *
 * June 5, 2013 Initialized based on global.h (last modified on Mar 14,
 * 2013) from kprop.6
 */

#ifndef KPROP_GLOBAL_H_
#define KPROP_GLOBAL_H_

#include <cstdio>
#include <map>
using std::map;
#include <string>
using std::string;
#include <sstream>
using std::ostringstream;
using std::istringstream;

/* macros */
/* compares 2 numeric values
 * returns 1 when a>b, returns 0 when a==b and returns -1 when a<b
 */
#define KPROP_COMPARE(a,b)  (((a) > (b)) - ((a) < (b)))
/* returns the max of two numeric values */
#define KPROP_MAX(a,b)      ((a) > (b) ? (a) : (b))
/* returns the min of two numeric values */
#define KPROP_MIN(a,b)      ((a) > (b) ? (b) : (a))
/* prints an error message */
#define KPROP_ERROR(...)    fprintf(stderr, "Error: %s: %d: %s(): ",    \
                                    __FILE__, __LINE__, __FUNCTION__);  \
                            fprintf(stderr, __VA_ARGS__);
/* prints a warning message */
#define KPROP_WARN(...)     fprintf(stderr, "Warning: %s: %d: %s(): ",  \
                                    __FILE__, __LINE__, __FUNCTION__);  \
                            fprintf(stderr, __VA_ARGS__);

namespace kprop
{

/* constants */
/* tolerance value to compare 2 float numbers */
extern const float TOLERANCE_VALUE;
/* whitespace characters */
extern const char  WHITESPACES[];

/* data types */
/* string-string map */
typedef map<string, string> ssmap;
/* iterator of a string-string map */
typedef map<string, string>::iterator ssmap_it;
/* const iterator of a string-string map */
typedef map<string, string>::const_iterator ssmap_cit;
/* string-int map */
typedef map<string, int> simap;
/* iterator of a string-int map */
typedef map<string, int>::iterator simap_it;
/* const iterator of a string-int map */
typedef map<string, int>::const_iterator simap_cit;
/* int-float map */
typedef map<int, float> ifmap;
/* iterator of a string-int map */
typedef map<int, float>::iterator ifmap_it;
/* const iterator of a string-int map */
typedef map<int, float>::const_iterator ifmap_cit;
/* <int, int> pair */
typedef struct
{
    int index;
    int weight;
} i2pair;
/* <int, int, int> tuple */
typedef struct
{
    int index;
    int weight1;
    int weight2;
} i3tuple;
/* <int, int, int, int> tuple */
typedef struct
{
    int index;
    int weight1;
    int weight2;
    int weight3;
} i4tuple;
/* <int, float> pair */
typedef struct
{
    int   index;
    float weight;
} ifpair;
/* <int, int, float> tuple */
typedef struct
{
    int   index;
    int   weight1;
    float weight2;
} i2ftuple;
/* <int, int, int, float> tuple */
typedef struct
{
    int   index;
    int   weight1;
    int   weight2;
    float weight3;
} i3ftuple;

/* functions */
/* comparison function for qsort (int, increasing) */
int cmp_int_i(void const* a, void const* b);
/* comparison function for qsort (int, decreasing) */
int cmp_int_d(void const* a, void const* b);
/* comparison function for qsort (float, increasing) */
int cmp_fl_i(void const* a, void const* b);
/* comparison function for qsort (float, decreasing) */
int cmp_fl_d(void const* a, void const* b);
/* comparison function for qsort (<int, int>, increasing) */
int cmp_i2pairs_i(void const* a, void const* b);
/* comparison function for qsort (<int, int>, decreasing) */
int cmp_i2pairs_d(void const* a, void const* b);
/* comparison function for qsort (<int, int, int>, increasing) */
int cmp_i3tuples_i(void const* a, void const* b);
/* comparison function for qsort (<int, int, int>, decreasing) */
int cmp_i3tuples_d(void const* a, void const* b);
/* comparison function for qsort (<int, int, int, int>, increasing) */
int cmp_i4tuples_i(void const* a, void const* b);
/* comparison function for qsort (<int, int, int, int>, decreasing) */
int cmp_i4tuples_d(void const* a, void const* b);
/* comparison function for qsort (<int, float>, increasing) */
int cmp_ifpairs_i(void const* a, void const* b);
/* comparison function for qsort (<int, float>, decreasing) */
int cmp_ifpairs_d(void const* a, void const* b);
/* comparison function for qsort (<int, int, float>, increasing) */
int cmp_i2ftuples_i(void const* a, void const* b);
/* comparison function for qsort (<int, int, float>, decreasing) */
int cmp_i2ftuples_d(void const* a, void const* b);
/* comparison function for qsort (<int, int, int, float>, increasing) */
int cmp_i3ftuples_i(void const* a, void const* b);
/* comparison function for qsort (<int, int, int, float>, decreasing) */
int cmp_i3ftuples_d(void const* a, void const* b);
/* removes leading and trailing whitespace(s) of a c++ string */
void trim(string& s);
/* T to string conversion
 * type T must support the "<<" operator
 */
template<class T> string T_as_string(const T& t)
{
    ostringstream ost;
    ost << t;
    return ost.str();
}
/* string to T conversion
 * type T must support the ">>" operator
 */
template<class T> T string_as_T(const string& s)
{
    T t;
    istringstream ist(s);
    ist >> t;
    return t;
}
/* string to string conversion (dummy function) */
template<> inline string string_as_T<string>(const string& s) { return s; }
/* string to bool conversion */
template<> inline bool string_as_T<bool>(const string& s)
{
    /* "false", "f", "no", "n", "na", "none", "0" and ""
     * will be interpreted as false; all others as true
     */
    string sup = s;
    for(string::iterator p = sup.begin(); p != sup.end(); ++p)
        *p = toupper(*p); /* case insensitive */
    if( sup==string("FALSE") || sup==string("F") ||
        sup==string("NO") || sup==string("N") ||
        sup==string("NA") || sup==string("NONE") ||
        sup==string("0") || sup==string("") )
        return false;
    return true;
}
/* Manhattan (L1) distance of two float vectors */
float l1Dist(const float* pa, const float* pb, int size);
/* Euclidean (L2) distance of two float vectors */
float l2Dist(const float* pa, const float* pb, int size);
/* partial Euclidean (L2) distance of two float vectors,
 * uses indices supplied */
float l2DistPar(const float* pa, const float* pb, const int* pi, int size);
/* vector angle distance of two float vectors */
float vaDist(const float* pa, const float* pb, int size);
/* shared nearest neighbor (snn) similarity of two float vectors
 * the two vectors are sorted nearest neighbor lists
 * the parameter length is usually the size of the vector space
 */
float snnSim(const float* pa, const float* pb, int size, int length);

} /* namespace kprop */

#endif /* KPROP_GLOBAL_H_ */
