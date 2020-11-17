/* Derived from the modified nndes-data.h as of June 18, 2013, see comments
 * below
 *
 * Jichao Sun (js87@njit.edu)
 *
 * July 16, 2013
 *   now includes nndes-data.h, nndes-data-avx.h and nndes-data-sse2.h
 */

/* modified by Jichao Sun on June 18, 2013
 *   added void load(const DenMatSin *) to class Dataset
 */

/* 
    Copyright (C) 2010,2011 Wei Dong <wdong.pku@gmail.com>. All Rights Reserved.

    DISTRIBUTION OF THIS PROGRAM IN EITHER BINARY OR SOURCE CODE FORM MUST BE
    PERMITTED BY THE AUTHOR.
*/

#ifndef __NNDES_DATA__
#define __NNDES_DATA__
#include <malloc.h>
#include <fstream>
#include <cmath>

#include <cstdio>
#include "../DenMatSin.h" /* to read dvf file */
using namespace kprop;

namespace nndes
{

template <typename T>
T sqr(T a)
{
    return a * a;
}

/* usually, float has 4 bytes and double has 8 bytes */

// Dataset
// T: element type
// A: alignment, default = 128 bits = 16 bytes
#ifdef __GNUC__
#ifdef __AVX__
#define NNDES_MATRIX_ALIGN 32
#else
#ifdef __SSE2__
#define NNDES_MATRIX_ALIGN 16
#else
#define NNDES_MATRIX_ALIGN 1
#endif
#endif
#endif

template <typename T, int A = NNDES_MATRIX_ALIGN> 
class Dataset
{
    int dim;
    int N;
    size_t stride;
    char *dims;
public:
    typedef T value_type;
    static const int ALIGN = A;

    void reset (int _dim, int _N)
    {
        BOOST_ASSERT((ALIGN % sizeof(T)) == 0);
        dim = _dim;
        N = _N;
        stride = dim * sizeof(T) + ALIGN - 1;
        stride = stride / ALIGN * ALIGN;
        if (dims != NULL) delete[] dims;
        dims = (char *)memalign(ALIGN, N * stride); // SSE instruction needs data to be aligned
        std::fill(dims, dims + N * stride, 0);
    }

    void free (void)
    {
        dim = N = stride = 0;
        if (dims != NULL) free(dims);
        dims = NULL;
    }
    
    Dataset () :dim(0), N(0), dims(NULL) {}
    Dataset (int _dim, int _N) : dims(NULL) { reset(_dim, _N); }
    ~Dataset () { if (dims != NULL) delete[] dims; }

    /* free? delete[]? for memalign? ~Jichao */

    /// Access the ith vector.
    const T *operator [] (int i) const 
    {
        return (const T *)(dims + i * stride);
    }

    /// Access the ith vector.
    T *operator [] (int i) 
    {
        return (T *)(dims + i * stride);
    }

    int getDim () const {return dim; }
    int size () const {return N; }

    void load (const std::string &path) /* load a binary file. ~Jichao */
    {
        std::ifstream is(path.c_str(), std::ios::binary);
        int header[3]; /* entry size, row, col */
        assert(sizeof header == 3*4);
        is.read((char *)header, sizeof header);
        BOOST_VERIFY(is);
        BOOST_VERIFY(header[0] == sizeof(T));
        reset(header[2], header[1]);
        char *off = dims;
        for (int i = 0; i < N; ++i) {
            is.read(off, sizeof(T) * dim);
            off += stride;
        }
        BOOST_VERIFY(is);
    }

    // initialize from a file
    /* load a binary file. ~Jichao */
    void load (const std::string &path, int _dim, int skip = 0, int gap = 0) 
    {
        std::ifstream is(path.c_str(), std::ios::binary);
        BOOST_VERIFY(is);
        is.seekg(0, std::ios::end);
        size_t size = is.tellg();
        size -= skip;
        int line = sizeof(float) * _dim + gap;
        BOOST_VERIFY(size % line == 0);
        int _N =  size / line;
        reset(_dim, _N);
        is.seekg(skip, std::ios::beg);
        char *off = dims;
        for (int i = 0; i < N; ++i) {
            is.read(off, sizeof(T) * dim);
            is.seekg(gap, std::ios::cur);
            off += stride;
        }
        BOOST_VERIFY(is);
    }

    /* load from DenMatSin (float only). ~Jichao */
    void load (const DenMatSin* desc)
    {
        assert(false); /* not allowed for non-float ~Jichao */
    }

    Dataset (const std::string &path, int _dim, int skip = 0, int gap = 0)
        : dims(NULL) 
    {
        load(path, _dim, skip, gap);
    }

    /* What's this for? ~Jichao */
    float operator () (int i, int j) const __attribute__ ((noinline));
};

template <>
void Dataset<float, 16>::load(const DenMatSin* desc)
{
    int _dim = desc->getN();
    int _N   = desc->getM();
    reset(_dim, _N);
    char* off = dims;
    char* in  = (char*) (&(*desc)(0,0)); /* read byte by byte */
    for (int i = 0; i < N; ++i)
    {
        int vec_width = sizeof(float) * dim; /* number of bytes each vec */
        // strncpy(out, in, vec_width); /* this won't work, it stops at '\0' */
        memcpy(off, in, vec_width);
        in += vec_width;
        off += stride;
    }
}

// L1 distance oracle on a dense dataset
/* ?? ~Jichao */
class OracleDirect 
{
    const Dataset<float> &m;
public:
    OracleDirect (const Dataset<float> &m_): m(m_) { }
    float operator () (int i, int j) const { return m[i][j]; }
};

// L1 distance oracle on a dense dataset
template <typename M>
class OracleL1 
{
    const M &m;
public:
    OracleL1 (const M &m_): m(m_) { }
    float operator () (int i, int j) const 
    {
        const typename M::value_type *first1 = m[i];
        const typename M::value_type *first2 = m[j];
        float r = 0.0;
        for (int i = 0; i < m.getDim(); ++i)
        {
            r += fabs(first1[i] - first2[i]);
        }
        return r;
    }
};

// L2 distance oracle on a dense dataset
// special SSE optimization is implemented for float data
template <typename M>
class OracleL2 
{
    const M &m;
public:
    OracleL2 (const M &m_): m(m_) {}
    float operator () (int i, int j) const __attribute__ ((noinline));
};

template <typename M>
float OracleL2<M>::operator () (int i, int j) const 
{
    const typename M::value_type *first1 = m[i];
    const typename M::value_type *first2 = m[j];
    float r = 0.0;
    for (int i = 0; i < m.getDim(); ++i)
    {
        float v = first1[i] - first2[i];
        r += v * v;
    }
    return sqrt(r);
}

typedef Dataset<float> FloatDataset;

}

// !!! If problems happen here, the following lines can be removed
// without affecting the correctness of the library.
#ifdef __GNUC__
#ifdef __AVX__
/* content from the original nndes-data-avx.h */
#include <immintrin.h>
namespace nndes 
{
#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm256_load_ps(addr1);\
    tmp2 = _mm256_load_ps(addr2);\
    tmp1 = _mm256_sub_ps(tmp1, tmp2); \
    tmp1 = _mm256_mul_ps(tmp1, tmp1); \
    dest = _mm256_add_ps(dest, tmp1); 
    template <>
    float OracleL2<Dataset<float, 32> >::operator () (int i, int j) const 
    {
        __attribute__ ((aligned (32))) __m256 sum;
        __attribute__ ((aligned (32))) __m256 l0, l1, l2, l3;
        __attribute__ ((aligned (32))) __m256 r0, r1, r2, r3;
        int D = (m.getDim() + 7) & ~7U; // # dim aligned up to 256 bits, or 8 floats
        int DR = D % 32;
        int DD = D - DR;
        const float *l = m[i];
        const float *r = m[j];
        const float *e_l = l + DD;
        const float *e_r = r + DD;
        float unpack[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        float ret = 0.0;
        sum = _mm256_loadu_ps(unpack);
        switch (DR) 
        {
            case 24:
                AVX_L2SQR(e_l+16, e_r+16, sum, l2, r2);
            case 16:
                AVX_L2SQR(e_l+8, e_r+8, sum, l1, r1);
            case 8:
                AVX_L2SQR(e_l, e_r, sum, l0, r0);
        }
        for (int i = 0; i < DD; i += 32, l += 32, r += 32) 
        {
            AVX_L2SQR(l, r, sum, l0, r0);
            AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
            AVX_L2SQR(l + 16, r + 16, sum, l2, r2);
            AVX_L2SQR(l + 24, r + 24, sum, l3, r3);
        }
        _mm256_storeu_ps(unpack, sum);
        ret = unpack[0] + unpack[1] + unpack[2] + unpack[3]
            + unpack[4] + unpack[5] + unpack[6] + unpack[7];
        return sqrt(ret);
    }
}
#else
#ifdef __SSE2__
/* content from the original nndes-data-ssh2.h */
#include <xmmintrin.h>
namespace nndes 
{
#define SSE_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm_load_ps(addr1);\
    tmp2 = _mm_load_ps(addr2);\
    tmp1 = _mm_sub_ps(tmp1, tmp2); \
    tmp1 = _mm_mul_ps(tmp1, tmp1); \
    dest = _mm_add_ps(dest, tmp1); 
    template <>
    float OracleL2<Dataset<float, 16> >::operator () (int i, int j) const 
    {
        __m128 sum;
        __m128 l0, l1, l2, l3;
        __m128 r0, r1, r2, r3;
        int D = (m.getDim() + 3) & ~3U;
        int DR = D % 16;
        int DD = D - DR;
        const float *l = m[i];
        const float *r = m[j];
        const float *e_l = l + DD;
        const float *e_r = r + DD;
        float unpack[4] = {0, 0, 0, 0};
        float ret = 0.0;
        sum = _mm_loadu_ps(unpack);
        switch (DR) 
        {
            case 12:
                SSE_L2SQR(e_l+8, e_r+8, sum, l2, r2);
            case 8:
                SSE_L2SQR(e_l+4, e_r+4, sum, l1, r1);
            case 4:
                SSE_L2SQR(e_l, e_r, sum, l0, r0);
        }
        for (int i = 0; i < DD; i += 16, l += 16, r += 16) 
        {
            SSE_L2SQR(l, r, sum, l0, r0);
            SSE_L2SQR(l + 4, r + 4, sum, l1, r1);
            SSE_L2SQR(l + 8, r + 8, sum, l2, r2);
            SSE_L2SQR(l + 12, r + 12, sum, l3, r3);
        }
        _mm_storeu_ps(unpack, sum);
        ret = unpack[0] + unpack[1] + unpack[2] + unpack[3];
        return sqrt(ret);
    }
}
#endif /* #ifdef __SSE2__ */
#endif /* #ifdef __AVX__  */
#endif /* #ifdef __GNUC__ */

#endif /* #ifndef __NNDES_DATA__ */

