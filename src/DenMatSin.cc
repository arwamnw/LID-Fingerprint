/* DenMatSin.cc */

/* Implements DenMatSin.h.
 * Jichao Sun (js87@njit.edu)
 *
 * July 30, 2013
 *   added set(), unset() and isSet() functions
 *
 * June 5, 2013 Initialized based on DenMatSin.cc (last modified on Mar
 * 14, 2013) from kprop.6 
 */

#include "DenMatSin.h"
#include <cstdio>  /* for io */
#include <cstdlib> /* for qsort() and exit() */
//  #include <cblas.h> /* for cblas_segmm(), requires -lblas or -lcblas
//                      * an alternative is <gsl/gsl_cblas.h> with -lgslcblas
//                      */
#include <iostream>
using std::endl;
#include <fstream>
using std::ostream;
using std::ifstream;
#include <string>
using std::string;
#include "nist_spblas/nist_spblas.h" /* for sparse matrix multiplication */
#include "global.h"

kprop::DenMatSin::DenMatSin(): m_(0), n_(0), data_(NULL) {}
kprop::DenMatSin::DenMatSin(int m, int n) /* m-by-n */
{
    if (m <= 0 || n <= 0)
    {
        KPROP_WARN("the desired #rows or #columns is smaller than 1, "
                "an empty matrix will be created.\n");
    }
    m_ = KPROP_MAX(m, 0);
    n_ = KPROP_MAX(n, 0);
    if (isEmpty())
    {
        //printf("I am here empty");
        data_ = NULL;
    }
    else
    {
        int nr_items = m_ * n_;
        data_ = new float[nr_items];

	/* added by Arwa Wali to fix segmentation fault problem */
        data_=(float*) calloc(nr_items,sizeof(float));
        /* *********************************************** */
        for (int i = 0; i < nr_items; i++)
            data_[i] = .0F;
        printf("I am here full");
        //printf(data_);
        
    }
}
kprop::DenMatSin::DenMatSin(const string& filename, bool ascii)
    : m_(0), n_(0), data_(NULL)
{
    FILE* fp = NULL;
    int m, n, nr_items;
    if (ascii) /* read from an ascii file */
    {
        fp = fopen(filename.c_str(), "rt");
        if (fp == NULL)
        {
            KPROP_ERROR("could not open matrix file '%s' to read.\n",
                        filename.c_str());
            exit(EXIT_FAILURE);
        }
        if (fscanf(fp, "%d %d", &m, &n) != 2
            || m < 0 || n < 0)
        {
            KPROP_ERROR("corrupted header of matrix file '%s'.\n",
                        filename.c_str());
            exit(EXIT_FAILURE);
        }
        if (m == 0 || n == 0)
        {
            fclose(fp);
            m_ = m;
            n_ = n;
            return;
        }
        nr_items = m * n;
        data_ = new float[nr_items];
        for (int i = 0; i < nr_items; i++)
        {
            if (fscanf(fp, "%f", data_+i) != 1)
            {
                printf("%d", data_+i);  /* Edited by Arwa Wali to check the problem of corrupted matrix */
                KPROP_ERROR("corrupted matrix file '%s'.\n",
                            filename.c_str());
                exit(EXIT_FAILURE);
            }
        }
        fclose(fp);
        m_ = m;
        n_ = n;
    }
    else /* read from a bin file */
    {
        fp = fopen(filename.c_str(), "rb");
        if (fp == NULL)
        {
            KPROP_ERROR("could not open matrix file '%s' to read.\n",
                        filename.c_str());
            exit(EXIT_FAILURE);
        }
        if (fread(&m, sizeof(int), 1, fp) != 1
            || m < 0
            || fread(&n, sizeof(int), 1, fp) != 1
            || n < 0)
        {
            KPROP_ERROR("corrupted header of matrix file '%s'.\n",
                        filename.c_str());
            exit(EXIT_FAILURE);
        }
        if (m == 0 || n == 0)
        {
            fclose(fp);
            m_ = m;
            n_ = n;
            return;
        }
        nr_items = m * n;
        data_ = new float[nr_items];
        if (fread(data_, sizeof(float), nr_items, fp)
            != (unsigned int) nr_items)
        {
            KPROP_ERROR("corrupted matrix file '%s'.\n", filename.c_str());
            exit(EXIT_FAILURE);
        }
        fclose(fp);
        m_ = m;
        n_ = n;
    }
}
kprop::DenMatSin::~DenMatSin()
{
    if (data_ != NULL)
    {
        delete[] data_;
        data_ = NULL;
    }
}
int kprop::DenMatSin::getM() const { return m_; }
int kprop::DenMatSin::getN() const { return n_; }
void kprop::DenMatSin::save(const string& filename, bool ascii, int prec)
    const
{
    FILE* fp = NULL;
    if (ascii)
    {
        fp = fopen(filename.c_str(), "wt");
        if (fp == NULL)
        {
            KPROP_ERROR("could not open matrix file '%s' to write.\n",
                    filename.c_str());
            exit(EXIT_FAILURE);
        }
        fprintf(fp, "%d %d\n", m_, n_);
        /* prec means precision --- number of digits after decimal points
         * if prep < 0, print all */
        for (int i = 0; i < m_; ++i)
            for (int j = 0; j < n_; j++)
                fprintf(fp, "%.*f%c", prec,
                        data_[i*n_+j], (j==n_-1)?'\n':' ');
        fclose(fp);
    }
    else
    {
        fp = fopen(filename.c_str(), "wb");
        if (fp == NULL)
        {
            KPROP_ERROR("could not open matrix file '%s' to write.\n",
                        filename.c_str());
            exit(EXIT_FAILURE);
        }
        if (fwrite(&m_, sizeof(int), 1, fp) != 1
            || fwrite(&n_, sizeof(int), 1, fp) != 1)
        {
            KPROP_ERROR("could not write matrix file '%s'.\n",
                        filename.c_str());
            exit(EXIT_FAILURE);
        }
        int nr_items = m_ * n_;
        if (fwrite(data_, sizeof(float), nr_items, fp)
            != (unsigned int) nr_items)
        {
            KPROP_ERROR("could not write matrix file '%s'.\n",
                        filename.c_str());
            exit(EXIT_FAILURE);
        }
        fclose(fp);
    }
}
bool kprop::DenMatSin::isEmpty() const { return (m_ == 0 || n_ == 0); }
float kprop::DenMatSin::max() const
{
    if (isEmpty())
    {
        KPROP_WARN("no maximum value in an empty matrix (0 will be returned).\n");
        return 0.0F;
    }
    float max = data_[0];
    for (int i = 1; i < m_ * n_; i++)
        if (data_[i] > max) max = data_[i];
    return max;
}
float kprop::DenMatSin::min() const
{
    if (isEmpty())
    {
        KPROP_WARN("no minimum value in an empty matrix (0 will be returned).\n");
        return 0.0F;
    }
    float min = data_[0];
    for (int i = 1; i < m_ * n_; i++)
        if (data_[i] < min) min = data_[i];
    return min;
}
void kprop::DenMatSin::reinit()
{
    for (int i = 0; i < m_ * n_; i++)
        data_[i] = .0F;
}
void kprop::DenMatSin::copy(const DenMatSin& mat)
{
    if (m_ != mat.m_ || n_ != mat.n_)
    {
        KPROP_ERROR("matrices' dimensions do not agree.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < m_ * n_; i++)
        data_[i] = mat.data_[i];
}
void kprop::DenMatSin::print(int x, int y) const
{
    printf("%d-by-%d\n", m_, n_);
    if (isEmpty())
    {
        printf("[Empty]\n");
        return;
    }
    int nr_rows_print    = (x <= 0) ? m_ : KPROP_MIN(x, m_);
    int nr_columns_print = (x <= 0) ? n_ : KPROP_MIN(x, n_);
    for (int i = 0; i < nr_rows_print - 1; i++)
    {
        for (int j = 0; j < nr_columns_print - 1; j++)
        {
            printf("%.*f ", y, data_[i*n_+j]);
        }
        if (nr_columns_print < n_)
        {
            printf("... ");
        }
        printf("%.*f\n", y, data_[i*n_+n_-1]);
    }
    if (nr_rows_print != m_)
    {
        printf("...\n");
    }
    for (int j = 0; j < nr_columns_print - 1; j++)
    {
        printf("%.*f ", y, data_[(m_-1)*n_+j]);
    }
    if (nr_columns_print < n_)
    {
            printf("... ");
    }
    printf("%.*f\n", y, data_[(m_-1)*n_+(n_-1)]);
}
float& kprop::DenMatSin::operator () (int i, int j)
{
    checkBounds(i,j);
    return data_[i*n_ + j];
}
const float& kprop::DenMatSin::operator () (int i, int j) const
{
    checkBounds(i,j);
    return data_[i*n_ + j];
}
void kprop::DenMatSin::transpose(const DenMatSin& mat)
{
    if (m_ != mat.getN() || n_ != mat.getM())
    {
        KPROP_ERROR("disqualified matrix dimension(s).\n");
        exit(EXIT_FAILURE);
    }
    if (mat.isEmpty()) return;
    for (int i = 0; i < m_; i++)
        for (int j = 0; j < n_; j++)
            (*this)(i,j) = mat(j,i); /* might not be very efficient */
            // data_[i*n_+j] = mat.data_[j*m_+i]; /* efficient one */
}
void kprop::DenMatSin::set(int i, int j)
{
    checkBounds(i,j);
    (*this)(i,j) = 1.0F;
}
void kprop::DenMatSin::unset(int i, int j)
{
    checkBounds(i,j);
    (*this)(i,j) = 0.0F;
}
bool kprop::DenMatSin::isSet(int i, int j) const
{
    checkBounds(i,j);
    if ((float) fabs((*this)(i,j)) < TOLERANCE_VALUE)
        return false;
    else 
        return true;
}
ostream& kprop::operator << (ostream& os, const DenMatSin& mat)
{
    os << mat.m_ << " " << mat.n_ << endl;
    if (mat.isEmpty()) return os;
    for (int i = 0; i < mat.m_; i++)
        for (int j = 0; j < mat.n_; j++)
            os << mat.data_[i*mat.n_+j] << ((j==mat.n_-1)? "\n" : " ");
    return os;
}
void kprop::DenMatSin::checkBounds(int x, int y) const
{
    if (x < 0 || x >= m_ || y < 0 || y >= n_)
    {
        KPROP_ERROR("matrix index or indices out of bound.\n");
        exit(EXIT_FAILURE);
    }
}
//  void kprop::MatMultiply(const DenMatSin& A,
//                          const DenMatSin& B,
//                          DenMatSin& C)
//  {
//      int am = A.getM();
//      int an = A.getN();
//      int bm = B.getM();
//      int bn = B.getN();
//      int cm = C.getM();
//      int cn = C.getN();
//  
//      if (A.isEmpty() || B.isEmpty() || an != bm || am != cm || bn != cn)
//      {
//          KPROP_WARN("invalid matrices for multiplication, terminated.\n");
//          return;
//      }
//      /* requires cblas */
//      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//                  am, bn, an,
//                  1.0,
//                  &(A(0,0)), an,
//                  &(B(0,0)), bn,
//                  0.0,
//                  &(C(0,0)), cn);
//  }
void kprop::MatMultiply2(const NIST_SPBLAS::FSp_mat& A,
                         const DenMatSin& B,
                         DenMatSin& C)
{
    int am = A.num_rows();
    int an = A.num_cols();
    int bm = B.getM();
    int bn = B.getN();
    int cm = C.getM();
    int cn = C.getN();

    if (am == 0 || an == 0 || B.isEmpty()
        || an != bm || am != cm || bn != cn)
    {
        KPROP_ERROR("invalid matrices for multiplication.\n");
        exit(EXIT_FAILURE);
    }
    /* C must be reinitialized first */
    C.reinit();
    /* use spblas */
    // A.NIST_SPBLAS::FSp_mat::usmm(blas_rowmajor, blas_no_trans,
    A.usmm(blas_rowmajor, blas_no_trans,
           bn, 1.0F,
           &(B(0,0)), bn,
           &(C(0,0)), cn);
}
void kprop::MatAdd(const DenMatSin& A,
		   const DenMatSin& B,
		   DenMatSin& C)
{
    int am = A.getM();
    int an = A.getN();
    int bm = B.getM();
    int bn = B.getN();
    int cm = C.getM();
    int cn = C.getN();

    if (A.isEmpty() || B.isEmpty()
        || am != bm || am != cm || an != bn || an != cn)
    {
        KPROP_ERROR("invalid matrices for addition.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < cm * cn; i++)
        *(&(C(0,0))+i) = *(&(A(0,0))+i) + *(&(B(0,0))+i);
}

