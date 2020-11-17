/* nist_spblas.h */

/* Modified from Sparse Basic Linear Algebra Subprograms (SPBLAS) library
 * (http://math.nist.gov/spblas/).
 * This file combines blas_enum.h, blas_sparse_proto.h and (part of)
 * nist_spblas.h.
 * The original description is included below.
 *
 * Jichao Sun (js87@njit.edu)
 * Last modified: Feb 23, 2012
 */

/* ORIGINAL DESCRIPTION
 * 
 * Sparse BLAS (Basic Linear Algebra Subprograms) Library
 *
 * A C++ implementation of the routines specified by the ANSI C 
 * interface specification of the Sparse BLAS in the BLAS Technical 
 * Forum Standard[1].   For details, see [2].
 *
 * Mathematical and Computational Sciences Division
 * National Institute of Technology,
 * Gaithersburg, MD USA
 *
 *
 * [1] BLAS Technical Forum: www.netlib.org/blas/blast-forum/
 * [2] I. S. Duff, M. A. Heroux, R. Pozo, "An Overview of the Sparse Basic
 *     Linear Algebra Subprograms: The new standard of the BLAS Techincal
 *     Forum,"  Vol. 28, No. 2, pp. 239-267,ACM Transactions on Mathematical 
 *     Software (TOMS), 2002.
 *
 *
 * DISCLAIMER:
 *
 * This software was developed at the National Institute of Standards and
 * Technology (NIST) by employees of the Federal Government in the course
 * of their official duties. Pursuant to title 17 Section 105 of the
 * United States Code, this software is not subject to copyright protection
 * and is in the public domain. NIST assumes no responsibility whatsoever for
 * its use by other parties, and makes no guarantees, expressed or implied,
 * about its quality, reliability, or any other characteristic.
 */

#ifndef NIST_SPBLAS_H_
#define NIST_SPBLAS_H_

// #ifndef _BLAS_ENUM_H
// #define _BLAS_ENUM_H

  /* Enumerated types */
enum blas_order_type {
            blas_rowmajor = 101,
            blas_colmajor = 102 };

enum blas_trans_type {
            blas_no_trans   = 111,
            blas_trans      = 112,
            blas_conj_trans = 113 };

enum blas_uplo_type  {
            blas_upper = 121,
            blas_lower = 122 };

enum blas_diag_type {
            blas_non_unit_diag = 131,
            blas_unit_diag     = 132 };

enum blas_side_type {
            blas_left_side  = 141,
            blas_right_side = 142 };

enum blas_cmach_type {
            blas_base      = 151,
            blas_t         = 152,
            blas_rnd       = 153,
            blas_ieee      = 154,
            blas_emin      = 155,
            blas_emax      = 156,
            blas_eps       = 157,
            blas_prec      = 158,
            blas_underflow = 159,
            blas_overflow  = 160,
            blas_sfmin     = 161};

enum blas_norm_type {
            blas_one_norm       = 171,
            blas_real_one_norm  = 172,
            blas_two_norm       = 173,
            blas_frobenius_norm = 174,
            blas_inf_norm       = 175,
            blas_real_inf_norm  = 176,
            blas_max_norm       = 177,
            blas_real_max_norm  = 178 };

enum blas_sort_type {
            blas_increasing_order = 181,
            blas_decreasing_order = 182 };

enum blas_conj_type {
            blas_conj    = 191,
            blas_no_conj = 192 };

enum blas_jrot_type {
            blas_jrot_inner  = 201,
            blas_jrot_outer  = 202,
            blas_jrot_sorted = 203 };

enum blas_prec_type {
            blas_prec_single     = 211,
            blas_prec_double     = 212,
            blas_prec_indigenous = 213,
            blas_prec_extra      = 214 };

enum blas_base_type {
            blas_zero_base = 221,
            blas_one_base  = 222 };

enum blas_symmetry_type {
            blas_general          = 231,
            blas_symmetric        = 232,
            blas_hermitian        = 233,
            blas_triangular       = 234,
            blas_lower_triangular = 235,
            blas_upper_triangular = 236,
            blas_lower_symmetric  = 237,
            blas_upper_symmetric  = 238,
            blas_lower_hermitian  = 239,
            blas_upper_hermitian  = 240  };

enum blas_field_type {
            blas_complex          = 241,
            blas_real             = 242,
            blas_double_precision = 243,
            blas_single_precision = 244  };

enum blas_size_type {
            blas_num_rows      = 251,
            blas_num_cols      = 252,
            blas_num_nonzeros  = 253  };

enum blas_handle_type{
            blas_invalid_handle = 261,
			blas_new_handle     = 262,
			blas_open_handle    = 263,
			blas_valid_handle   = 264};

enum blas_sparsity_optimization_type {
            blas_regular       = 271,
            blas_irregular     = 272,
            blas_block         = 273,
            blas_unassembled   = 274 };

// #endif /* _BLAS_ENUM_H */

// #ifndef _BLAS_SPARSE_PROTO_H
// #define _BLAS_SPARSE_PROTO_H

typedef int blas_sparse_matrix;

  /* Level 1 Computational Routines */
void BLAS_susdot( enum blas_conj_type conj, int nz, const float *x, 
                  const int *indx, const float *y, int incy, float *r,
                  enum blas_base_type index_base );
void BLAS_dusdot( enum blas_conj_type conj, int nz, const double *x, 
                  const int *indx, const double *y, int incy, double *r,
                  enum blas_base_type index_base );
void BLAS_cusdot( enum blas_conj_type conj, int nz, const void *x, 
                  const int *indx, const void *y, int incy, void *r,
                  enum blas_base_type index_base );
void BLAS_zusdot( enum blas_conj_type conj, int nz, const void *x, 
                  const int *indx, const void *y, int incy, void *r,
                  enum blas_base_type index_base );

void BLAS_susaxpy( int nz, float alpha, const float *x, const int *indx,
                 float *y, int incy, enum blas_base_type index_base );
void BLAS_dusaxpy( int nz, double alpha, const double *x, const int *indx,
                 double *y, int incy, enum blas_base_type index_base );
void BLAS_cusaxpy( int nz, const void *alpha, const void *x, const int *indx,
                 void *y, int incy, enum blas_base_type index_base );
void BLAS_zusaxpy( int nz, const void *alpha, const void *x, const int *indx,
                 void *y, int incy, enum blas_base_type index_base );

void BLAS_susga( int nz, const float *y, int incy, float *x, const int *indx,
              enum blas_base_type index_base );
void BLAS_dusga( int nz, const double *y, int incy, double *x, const int *indx,
              enum blas_base_type index_base );
void BLAS_cusga( int nz, const void *y, int incy, void *x, const int *indx,
              enum blas_base_type index_base );
void BLAS_zusga( int nz, const void *y, int incy, void *x, const int *indx,
              enum blas_base_type index_base );

void BLAS_susgz( int nz, float *y, int incy, float *x, const int *indx,
              enum blas_base_type index_base );
void BLAS_dusgz( int nz, double *y, int incy, double *x, const int *indx,
              enum blas_base_type index_base );
void BLAS_cusgz( int nz, void *y, int incy, void *x, const int *indx,
              enum blas_base_type index_base );
void BLAS_zusgz( int nz, void *y, int incy, void *x, const int *indx,
              enum blas_base_type index_base );

void BLAS_sussc( int nz, const float *x, float *y, int incy, const int *indx,
              enum blas_base_type index_base );
void BLAS_dussc( int nz, const double *x, double *y, int incy, const int *indx,
              enum blas_base_type index_base );
void BLAS_cussc( int nz, const void *x, void *y, int incy, const int *indx,
              enum blas_base_type index_base );
void BLAS_zussc( int nz, const void *x, void *y, int incy, const int *indx,
              enum blas_base_type index_base );

               /* Level 2 Computational Routines */
int BLAS_susmv( enum blas_trans_type transa, float alpha, 
    blas_sparse_matrix A, const float *x, int incx, float *y, int incy );
int BLAS_dusmv( enum blas_trans_type transa, double alpha, 
    blas_sparse_matrix A, const double *x, int incx, double *y, int incy );
int BLAS_cusmv( enum blas_trans_type transa, const void *alpha, 
    blas_sparse_matrix A, const void *x, int incx, void *y, int incy );
int BLAS_zusmv( enum blas_trans_type transa, const void *alpha, 
    blas_sparse_matrix A, const void *x, int incx, void *y, int incy );

int BLAS_sussv( enum blas_trans_type transt, float alpha, 
    blas_sparse_matrix T, float *x, int incx );
int BLAS_dussv( enum blas_trans_type transt, double alpha, 
    blas_sparse_matrix T, double *x, int incx );
int BLAS_cussv( enum blas_trans_type transt, const void *alpha, 
    blas_sparse_matrix T, void *x, int incx );
int BLAS_zussv( enum blas_trans_type transt, const void *alpha, 
    blas_sparse_matrix T, void *x, int incx );

               /* Level 3 Computational Routines */
int BLAS_susmm( enum blas_order_type order, enum blas_trans_type transa,
    int nrhs, float alpha, blas_sparse_matrix A, const float *b, int ldb,
        float *c, int ldc );
int BLAS_dusmm( enum blas_order_type order, enum blas_trans_type transa,
        int nrhs, double alpha, blas_sparse_matrix A, const double *b,
        int ldb, double *c, int ldc );
int BLAS_cusmm( enum blas_order_type order, enum blas_trans_type transa,
         int nrhs, const void *alpha, blas_sparse_matrix A, const void *b, 
     int ldb, void *c, int ldc );
int BLAS_zusmm( enum blas_order_type order, enum blas_trans_type transa,
         int nrhs, const void *alpha, blas_sparse_matrix A, const void *b, 
     int ldb, void *c, int ldc );

int BLAS_sussm( enum blas_order_type order, enum blas_trans_type transt,
              int nrhs, float alpha, int t, float *b, int ldb );
int BLAS_dussm( enum blas_order_type order, enum blas_trans_type transt,
              int nrhs, double alpha, int t, double *b, int ldb );
int BLAS_cussm( enum blas_order_type order, enum blas_trans_type transt,
              int nrhs, const void *alpha, int t, void *b, int ldb );
int BLAS_zussm( enum blas_order_type order, enum blas_trans_type transt,
              int nrhs, const void *alpha, int t, void *b, int ldb );

               /* Handle Management Routines */

               /* Creation Routines */
blas_sparse_matrix BLAS_suscr_begin( int m, int n );
blas_sparse_matrix BLAS_duscr_begin( int m, int n );
blas_sparse_matrix BLAS_cuscr_begin( int m, int n );
blas_sparse_matrix BLAS_zuscr_begin( int m, int n );


blas_sparse_matrix BLAS_suscr_block_begin( int Mb, int Nb, int k, int l );
blas_sparse_matrix BLAS_duscr_block_begin( int Mb, int Nb, int k, int l );
blas_sparse_matrix BLAS_cuscr_block_begin( int Mb, int Nb, int k, int l );
blas_sparse_matrix BLAS_zuscr_block_begin( int Mb, int Nb, int k, int l );

blas_sparse_matrix BLAS_suscr_variable_block_begin( int Mb, int Nb, 
		const int *k, const int *l );
blas_sparse_matrix BLAS_duscr_variable_block_begin( int Mb, int Nb, 
		const int *k, const int *l );
blas_sparse_matrix BLAS_cuscr_variable_block_begin( int Mb, int Nb, 
		const int *k, const int *l );
blas_sparse_matrix BLAS_zuscr_variable_block_begin( int Mb, int Nb, 
		const int *k, const int *l );

               /* Insertion Routines */
int BLAS_suscr_insert_entry( blas_sparse_matrix A, float val, int i, int j );
int BLAS_duscr_insert_entry( blas_sparse_matrix A, double val, int i, int j );
int BLAS_cuscr_insert_entry( blas_sparse_matrix A, const void *val, int i, int j );
int BLAS_zuscr_insert_entry( blas_sparse_matrix A, const void *val, int i, int j );

int BLAS_suscr_insert_entries( blas_sparse_matrix A, int nz, const float *val,
                            const int *indx, const int *jndx );
int BLAS_duscr_insert_entries( blas_sparse_matrix A, int nz, const double *val,
                            const int *indx, const int *jndx );
int BLAS_cuscr_insert_entries( blas_sparse_matrix A, int nz, const void *val,
                            const int *indx, const int *jndx );
int BLAS_zuscr_insert_entries( blas_sparse_matrix A, int nz, const void *val,
                            const int *indx, const int *jndx );

int BLAS_suscr_insert_col( blas_sparse_matrix A, int j, int nz,
                           const float *val, const int *indx );
int BLAS_duscr_insert_col( blas_sparse_matrix A, int j, int nz,
                           const double *val, const int *indx );
int BLAS_cuscr_insert_col( blas_sparse_matrix A, int j, int nz,
                           const void *val, const int *indx );
int BLAS_zuscr_insert_col( blas_sparse_matrix A, int j, int nz,
                           const void *val, const int *indx );

int BLAS_suscr_insert_row( blas_sparse_matrix A, int i, int nz,
                           const float *val, const int *indx );
int BLAS_duscr_insert_row( blas_sparse_matrix A, int i, int nz,
                           const double *val, const int *indx );
int BLAS_cuscr_insert_row( blas_sparse_matrix A, int i, int nz,
                           const void *val, const int *indx );
int BLAS_zuscr_insert_row( blas_sparse_matrix A, int i, int nz,
                           const void *val, const int *indx );

int BLAS_suscr_insert_clique( blas_sparse_matrix A, const int k, const int l, 
                        const float *val, const int row_stride, 
                        const int col_stride, const int *indx, 
                        const int *jndx );
int BLAS_duscr_insert_clique( blas_sparse_matrix A, const int k, const int l, 
                        const double *val, const int row_stride, 
                        const int col_stride, const int *indx, 
                        const int *jndx );
int BLAS_cuscr_insert_clique( blas_sparse_matrix A, const int k, const int l, 
                        const void *val, const int row_stride, 
                        const int col_stride, const int *indx, 
                        const int *jndx );
int BLAS_zuscr_insert_clique( blas_sparse_matrix A, const int k, const int l, 
                        const void *val, const int row_stride, 
                        const int col_stride, const int *indx, 
                        const int *jndx );

int BLAS_suscr_insert_block( blas_sparse_matrix A, const float *val, 
                        int row_stride, int col_stride, int i, int j );
int BLAS_duscr_insert_block( blas_sparse_matrix A, const double *val, 
                        int row_stride, int col_stride, int i, int j );
int BLAS_cuscr_insert_block( blas_sparse_matrix A, const void *val, 
                        int row_stride, int col_stride, int i, int j );
int BLAS_zuscr_insert_block( blas_sparse_matrix A, const void *val, 
                        int row_stride, int col_stride, int i, int j );

               /* Completion of Construction Routines */
int BLAS_suscr_end( blas_sparse_matrix A );
int BLAS_duscr_end( blas_sparse_matrix A );
int BLAS_cuscr_end( blas_sparse_matrix A );
int BLAS_zuscr_end( blas_sparse_matrix A );

               /* Matrix Property Routines */
int BLAS_usgp( blas_sparse_matrix A, int pname );
int BLAS_ussp( blas_sparse_matrix A, int pname );

               /* Destruction Routine */
int BLAS_usds( blas_sparse_matrix A );

// #endif /* BLAS_SPARSE_PROTO_H */

// #ifndef _NIST_SPBLAS_H
// #define _NIST_SPBLAS_H

#include <iostream>
#include <vector>
#include <complex>
using namespace std;

// #include "blas_enum.h"
// #include "blas_sparse_proto.h"

#ifdef SPBLAS_ERROR_FATAL
#include <cassert>
#define ASSERT_RETURN(x, ret_val) assert(x)
#define ERROR_RETURN(ret_val)  assert(0)
#else
#define ASSERT_RETURN(x, ret_val) {if (!(x)) return ret_val;}
#define ERROR_RETURN(ret_val) return ret_val
#endif

/* Level 1 */
/* The two functions below were moved here for compiling.
 * Jichao Sun
 * Last modified: Dec 29, 2011
 */
/* dummy routines for real version of usdot to compile. */
inline const double& conj(const double &x)
{ 
  return x;
}

inline const float& conj(const float &x)
{ 
  return x;
}

/* these macros are useful for creating some consistency between the 
   various precisions and floating point types.
*/
typedef float    FLOAT;
typedef double   DOUBLE;
typedef complex<float> COMPLEX_FLOAT;
typedef complex<double> COMPLEX_DOUBLE;

typedef float          SPBLAS_FLOAT_IN;
typedef double         SPBLAS_DOUBLE_IN;
typedef const void *   SPBLAS_COMPLEX_FLOAT_IN;
typedef const void *   SPBLAS_COMPLEX_DOUBLE_IN;

typedef float *  SPBLAS_FLOAT_OUT;
typedef double * SPBLAS_DOUBLE_OUT;
typedef void *   SPBLAS_COMPLEX_FLOAT_OUT;
typedef void *   SPBLAS_COMPLEX_DOUBLE_OUT;

typedef float *  SPBLAS_FLOAT_IN_OUT;
typedef double * SPBLAS_DOUBLE_IN_OUT;
typedef void *   SPBLAS_COMPLEX_FLOAT_IN_OUT;
typedef void *   SPBLAS_COMPLEX_DOUBLE_IN_OUT;

typedef const float *  SPBLAS_VECTOR_FLOAT_IN;
typedef const double * SPBLAS_VECTOR_DOUBLE_IN;
typedef const void *   SPBLAS_VECTOR_COMPLEX_FLOAT_IN;
typedef const void *   SPBLAS_VECTOR_COMPLEX_DOUBLE_IN;

typedef float *  SPBLAS_VECTOR_FLOAT_OUT;
typedef double * SPBLAS_VECTOR_DOUBLE_OUT;
typedef void *   SPBLAS_VECTOR_COMPLEX_FLOAT_OUT;
typedef void *   SPBLAS_VECTOR_COMPLEX_DOUBLE_OUT;

typedef float *  SPBLAS_VECTOR_FLOAT_IN_OUT;
typedef double * SPBLAS_VECTOR_DOUBLE_IN_OUT;
typedef void *   SPBLAS_VECTOR_COMPLEX_FLOAT_IN_OUT;
typedef void *   SPBLAS_VECTOR_COMPLEX_DOUBLE_IN_OUT;

#define SPBLAS_TO_FLOAT_IN(x)   x
#define SPBLAS_TO_DOUBLE_IN(x)  x
#define SPBLAS_TO_COMPLEX_FLOAT_IN(x) \
        (* reinterpret_cast<const complex<float> *>(x))
#define SPBLAS_TO_COMPLEX_DOUBLE_IN(x)  \
        (* reinterpret_cast<const complex<double> *>(x))

#define SPBLAS_TO_FLOAT_OUT(x)  x
#define SPBLAS_TO_DOUBLE_OUT(x) x
#define SPBLAS_TO_COMPLEX_FLOAT_OUT(x)  reinterpret_cast<complex<float> *>(x)
#define SPBLAS_TO_COMPLEX_DOUBLE_OUT(x) reinterpret_cast<complex<double> *>(x)  

#define SPBLAS_TO_FLOAT_IN_OUT(x)   x
#define SPBLAS_TO_DOUBLE_IN_OUT(x)  x
#define SPBLAS_TO_COMPLEX_FLOAT_IN_OUT(x)  reinterpret_cast<complex<float> *>(x)
#define SPBLAS_TO_COMPLEX_DOUBLE_IN_OUT(x) reinterpret_cast<complex<double>*>(x)  

#define SPBLAS_TO_VECTOR_DOUBLE_IN(x)   x 
#define SPBLAS_TO_VECTOR_FLOAT_IN(x)  x 
#define SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN(x) \
                          reinterpret_cast<const complex<float>*>(x)
#define SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN(x) \
                          reinterpret_cast<const complex<double>*>(x)

#define SPBLAS_TO_VECTOR_DOUBLE_OUT(x)  x 
#define SPBLAS_TO_VECTOR_FLOAT_OUT(x)   x 
#define SPBLAS_TO_VECTOR_COMPLEX_FLOAT_OUT(x) \
                          reinterpret_cast<complex<float>*>(x)
#define SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_OUT(x) \
                          reinterpret_cast<complex<double>*>(x)

#define SPBLAS_TO_VECTOR_DOUBLE_IN_OUT(x)   x 
#define SPBLAS_TO_VECTOR_FLOAT_IN_OUT(x)  x 
#define SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN_OUT(x) \
                          reinterpret_cast<complex<float>*>(x)
#define SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN_OUT(x) \
                          reinterpret_cast<complex<double>*>(x)

#define BLAS_FLOAT_NAME(routine_name) BLAS_s##routine_name
#define BLAS_DOUBLE_NAME(routine_name) BLAS_d##routine_name
#define BLAS_COMPLEX_FLOAT_NAME(routine_name) BLAS_c##routine_name
#define BLAS_COMPLEX_DOUBLE_NAME(routine_name) BLAS_z##routine_name

#define TSp_MAT_SET_FLOAT(A) {A->set_single_precision(); A->set_real();}
#define TSp_MAT_SET_DOUBLE(A) {A->set_double_precision(); A->set_real();}
#define TSp_MAT_SET_COMPLEX_FLOAT(A) {A->set_single_precision(); A->set_complex();}
#define TSp_MAT_SET_COMPLEX_DOUBLE(A) {A->set_double_precision(); A->set_complex();}

namespace NIST_SPBLAS
{
/**
   Generic sparse matrix (base) class: defines only the structure 
   (size, symmetry, etc.) and maintains state during construction, 
   but does not specify the actual nonzero values, or their type. 

*/
class Sp_mat
{
  private:
    int num_rows_;
    int num_cols_;
    int num_nonzeros_;

    /* ... */

    int void_;
    int nnew_;      /* avoid using "new" since it is a C++ keyword */
    int open_;
    int valid_;

    int unit_diag_ ;
    int complex_;
    int real_;
    int single_precision_;
    int double_precision_;
    int upper_triangular_;
    int lower_triangular_;
    int upper_symmetric_;
    int lower_symmetric_;
    int upper_hermitian_;
    int lower_hermitian_;
    int general_;

    int one_base_;

        /* optional block information */
    int Mb_;                /* matrix is partitioned into Mb x Nb blocks    */
    int Nb_;                /* otherwise 0, if regular (non-blocked) matrix */
    int k_;                 /* for constant blocks, each block is k x l     */
    int l_;                 /* otherwise 0, if variable blocks are used.   */

    int rowmajor_;          /* 1,if block storage is rowm major.  */
    int colmajor_;          /* 1,if block storage is column major. */

    /* unused optimization paramters */
    int opt_regular_;
    int opt_irregular_;
    int opt_block_;
    int opt_unassembled_;

    vector<int> K_; /* these are GLOBAL index of starting point of block     */
    vector<int> L_; /* i.e. block(i,j) starts at global location (K[i],L[i]) */
                    /* and of size (K[i+1]-K[i] x L[i+1]-L[i])               */

  public:
    Sp_mat(int M, int N) : 
      num_rows_(M),         /* default construction */
      num_cols_(N),
      num_nonzeros_(0),

      void_(0),
      nnew_(1),
      open_(0),
      valid_(0),

      unit_diag_(0),
      complex_(0),
      real_(0),
      single_precision_(0),
      double_precision_(0),
      upper_triangular_(0),
      lower_triangular_(0),
      upper_symmetric_(0),
      lower_symmetric_(0),
      upper_hermitian_(0),
      lower_hermitian_(0),
      general_(0),
      one_base_(0),
      Mb_(0),
      Nb_(0),
      k_(0),
      l_(0),
      rowmajor_(0),
      colmajor_(0),
      opt_regular_(0),
      opt_irregular_(1),
      opt_block_(0),
      opt_unassembled_(0),
      K_(),
      L_()
      {}

    int& num_rows()           { return num_rows_; }
    int& num_cols()           { return num_cols_; }
    int& num_nonzeros()         { return num_nonzeros_;}

    int num_rows() const        { return num_rows_; }
    int num_cols() const        { return num_cols_; }
    int num_nonzeros() const      { return num_nonzeros_;}

    int is_one_base() const     { return (one_base_ ? 1 : 0); }
    int is_zero_base() const    { return (one_base_ ? 0 : 1); }
    int is_void() const         { return void_; }
    int is_new() const          { return nnew_; }
    int is_open() const         { return open_; }
    int is_valid() const        { return valid_; }

    int is_unit_diag() const    { return unit_diag_; }
    int is_complex() const        { return complex_;}
    int is_real() const         { return real_;}
    int is_single_precision() const   { return single_precision_;}
    int is_double_precision() const   { return double_precision_;}
    int is_upper_triangular() const   { return upper_triangular_;}
    int is_lower_triangular() const   { return lower_triangular_;}
    int is_triangular() const     { return upper_triangular_ ||
                           lower_triangular_; }


    int is_lower_symmetric() const    { return lower_symmetric_; }
    int is_upper_symmetric() const    { return upper_symmetric_; }
    int is_symmetric() const      { return upper_symmetric_ ||
                           lower_symmetric_; }

    int is_lower_hermitian() const    { return lower_hermitian_; }
    int is_upper_hermitian() const    { return upper_hermitian_; }
    int is_hermitian() const  { return lower_hermitian_ || 
                                       upper_hermitian_; }
    int is_general() const { return !( is_hermitian() || is_symmetric()) ; }

    int is_lower_storage() const { return is_lower_triangular() ||
                                          is_lower_symmetric()  ||
                                          is_lower_hermitian() ; }

    int is_upper_storage() const { return is_upper_triangular() ||
                                          is_upper_symmetric()  ||
                                          is_upper_hermitian() ; }

    int is_opt_regular() const { return opt_regular_; }
    int is_opt_irregular() const { return opt_irregular_; }
    int is_opt_block() const { return opt_block_;} 
    int is_opt_unassembled() const { return opt_unassembled_;}

    int K(int i) const { return (k_ ? i*k_ : K_[i] ); }
    int L(int i) const { return (l_ ? i*l_ : L_[i] ); }

    int is_rowmajor() const { return rowmajor_; }
    int is_colmajor() const { return colmajor_; }

    void set_one_base()   { one_base_ = 1; }
    void set_zero_base()  { one_base_ = 0; }

    void set_void()       { void_ = 1;  nnew_ = open_ =  valid_ = 0;}
    void set_new()        { nnew_ = 1;  void_ = open_ =  valid_ = 0;}
    void set_open()       { open_ = 1;  void_ = nnew_  = valid_ = 0;}
    void set_valid()      { valid_ = 1; void_ = nnew_ =  open_ = 0; }

    void set_unit_diag()    { unit_diag_ = 1;}
    void set_complex()        {complex_ = 1; }
    void set_real()         { real_ = 1; }
    void set_single_precision()   { single_precision_ = 1; }
    void set_double_precision()   { double_precision_ = 1; }
    void set_upper_triangular()   { upper_triangular_ = 1; }
    void set_lower_triangular()   { lower_triangular_ = 1; }
    void set_upper_symmetric()  { upper_symmetric_ = 1; }
    void set_lower_symmetric()  { lower_symmetric_ = 1; }
    void set_upper_hermitian()  { upper_hermitian_ = 1; }
    void set_lower_hermitian()  { lower_hermitian_ = 1; }
  
    void set_const_block_parameters(int Mb, int Nb, int k, int l)
    {
      Mb_ = Mb;
      Nb_ = Nb;
      k_ = k;
      l_ = l;
    }

    void set_var_block_parameters(int Mb, int Nb, const int *k, const int *l)
    {
      Mb_ = Mb;
      Nb_ = Nb;
      k_ = 0;
      l_ = 0;

      K_.resize(Mb+1);
      K_[0] = 0;
      for (int i=0; i<Mb; i++)
        K_[i+1] = k[i] + K_[i];
      
      L_.resize(Nb+1);
      L_[0] = 0;
      for (int j=0; j<Mb; j++)
        K_[j+1] = k[j] + K_[j];
    }

    virtual int end_construction()
    {
      if (is_open() || is_new())
      {
        set_valid();

        return 0;
      }
      else
        ERROR_RETURN(1);
    }

    virtual void print() const;

    virtual void destroy() {};

    virtual ~Sp_mat() {};
};

template <class T>
class TSp_mat : public Sp_mat
{
  private:
    vector< vector< pair<T, int> > > S;
    vector<T> diag;                 /* optional diag if matrix is
                        triangular. Created
                        at end_construction() phase */
  private:
    inline T sp_dot_product( const vector< pair<T, int> > &r, 
        const T* x, int incx ) const
    {
        T sum(0);

        if (incx == 1)
        {
          for ( typename vector< pair<T,int> >::const_iterator p = r.begin(); 
            p < r.end(); p++)
          {
            //sum = sum + p->first * x[p->second];
            sum += p->first * x[p->second];
          }
        }
        else /* incx != 1 */
        {
          for ( typename vector< pair<T,int> >::const_iterator p = r.begin(); 
            p < r.end(); p++)
           {
            //sum = sum + p->first * x[p->second * incx];
            sum += p->first * x[p->second * incx];
            }
        }
    
        return sum;
  }

    inline T sp_conj_dot_product( const vector< pair<T, int> > &r, 
        const T* x, int incx ) const
    {
        T sum(0);

        if (incx == 1)
        {
          for ( typename vector< pair<T,int> >::const_iterator p = r.begin(); 
            p < r.end(); p++)
          {
            sum += conj(p->first) * x[p->second];
          }
        }
        else /* incx != 1 */
        {
          for ( typename vector< pair<T,int> >::const_iterator p = r.begin(); 
            p < r.end(); p++)
           {
            //sum = sum + p->first * x[p->second * incx];
            sum += conj(p->first) * x[p->second * incx];
            }
        }
    
        return sum;
  }

  inline void sp_axpy( const T& alpha, const vector< pair<T,int> > &r, 
      T*  y, int incy) const
  {
    if (incy == 1)
    {
      for (typename vector< pair<T,int> >::const_iterator p = r.begin(); 
          p < r.end(); p++)
       y[p->second] += alpha * p->first;  

    }
    else /* incy != 1 */
    {
    for (typename vector< pair<T,int> >::const_iterator p = r.begin(); 
        p < r.end(); p++)
      y[incy * p->second] += alpha * p->first;  
    } 
  } 

  inline void sp_conj_axpy( const T& alpha, const vector< pair<T,int> > &r, 
      T*  y, int incy) const
  {
    if (incy == 1)
    {
      for (typename vector< pair<T,int> >::const_iterator p = r.begin(); 
          p < r.end(); p++)
       y[p->second] += alpha * conj(p->first);  

    }
    else /* incy != 1 */
    {
    for (typename vector< pair<T,int> >::const_iterator p = r.begin(); 
        p < r.end(); p++)
      y[incy * p->second] += alpha * conj(p->first);  
    } 
  } 

  void mult_diag(const T& alpha, const T* x, int incx, T* y, int incy) 
      const
  {
    const T* X = x;
    T* Y = y;
    typename vector<T>::const_iterator d= diag.begin();
    for ( ; d < diag.end(); X+=incx, d++, Y+=incy)
    {
      *Y += alpha * *d * *X;
    }
  }

  void mult_conj_diag(const T& alpha, const T* x, int incx, T* y, int incy) 
      const
  {
    const T* X = x;
    T* Y = y;
    typename vector<T>::const_iterator d= diag.begin();
    for ( ; d < diag.end(); X+=incx, d++, Y+=incy)
    {
      *Y += alpha * conj(*d) * *X;
    }
  }

  void nondiag_mult_vec(const T& alpha, const T* x, int incx, 
      T* y, int incy) const
  {

    int M = num_rows();

    if (incy == 1)
    {
      for (int i=0; i<M; i++)
        y[i] += alpha * sp_dot_product(S[i], x, incx);
    }
    else
    {
      for (int i=0; i<M; i++)
        y[i * incy] += alpha * sp_dot_product(S[i], x, incx);
    }
  }

  void nondiag_mult_vec_conj(const T& alpha, const T* x, int incx, 
      T* y, int incy) const
  {

    int M = num_rows();

    if (incy == 1)
    {
      for (int i=0; i<M; i++)
        y[i] += alpha * sp_conj_dot_product(S[i], x, incx);
    }
    else
    {
      for (int i=0; i<M; i++)
        y[i * incy] += alpha * sp_conj_dot_product(S[i], x, incx);
    }
  }

  void nondiag_mult_vec_transpose(const T& alpha, const T* x, int incx, 
      T* y, int incy) const
  {
    /* saxpy: y += (alpha * x[i]) row[i]  */

    int M = num_rows();
    const T* X = x;
    for (int i=0; i<M; i++, X += incx)
      sp_axpy( alpha * *X, S[i], y, incy);
  }

  void nondiag_mult_vec_conj_transpose(const T& alpha, const T* x, int incx, 
      T* y, int incy) const
  {
    /* saxpy: y += (alpha * x[i]) row[i]  */

    int M = num_rows();
    const T* X = x;
    for (int i=0; i<M; i++, X += incx)
      sp_conj_axpy( alpha * *X, S[i], y, incy);
  }

  void mult_vec(const T& alpha, const T* x, int incx, T* y, int incy) 
      const
  {
    nondiag_mult_vec(alpha, x, incx, y, incy);

    if (is_triangular() || is_symmetric())
      mult_diag(alpha, x, incx, y, incy);

    if (is_symmetric())
      nondiag_mult_vec_transpose(alpha, x, incx, y, incy);
  }


  void mult_vec_transpose(const T& alpha, const T* x, int incx, T* y, 
      int incy) const
  {

    nondiag_mult_vec_transpose(alpha, x, incx, y, incy);

    if (is_triangular() || is_symmetric())
      mult_diag(alpha, x, incx, y, incy);

    if (is_symmetric())
      nondiag_mult_vec(alpha, x, incx, y, incy);
  }

  void mult_vec_conj_transpose(const T& alpha, const T* x, int incx, T* y, 
      int incy) const
  {

    nondiag_mult_vec_conj_transpose(alpha, x, incx, y, incy);

    if (is_triangular() || is_symmetric())
      mult_conj_diag(alpha, x, incx, y, incy);

    if (is_symmetric())
      nondiag_mult_vec_conj(alpha, x, incx, y, incy);
  }
      
  int triangular_solve(T alpha, T* x, int incx ) const
  {
    if (alpha == (T) 0.0)
      ERROR_RETURN(1);

    if ( ! is_triangular() )
      ERROR_RETURN(1);

    int N = num_rows();

    if (is_lower_triangular())
    {
        for (int i=0, ii=0; i<N; i++, ii += incx)
        {
            x[ii] = (x[ii] - sp_dot_product(S[i], x, incx)) / diag[i];
        }
       if (alpha != (T) 1.0)
       {
        for (int i=0, ii=0; i<N; i++, ii += incx)
            x[ii] /= alpha; 
       }
    }
    else if (is_upper_triangular())
    {

      for (int i=N-1, ii=(N-1)*incx ;   0<=i ;    i--, ii-=incx)
      {
         x[ii] = (x[ii] - sp_dot_product(S[i],x, incx)) / diag[i];
      }
      if (alpha != (T) 1.0)
      {
        for (int i=N-1, ii=(N-1)*incx ;   0<=i ;    i--, ii-=incx)
          x[ii] /= alpha; 
      }

    }
    else
        ERROR_RETURN(1);

    return 0;
  }

  int transpose_triangular_solve(T alpha, T* x, int incx) const
  {
    if ( ! is_triangular())
      return -1;

    int N = num_rows();

    if (is_lower_triangular())
    {

      for (int j=N-1, jj=(N-1)*incx; 0<=j; j--, jj -= incx)
      {
        x[jj] /= diag[j] ;
        sp_axpy( -x[jj], S[j], x, incx);
      }
      if (alpha != (T) 1.0)
      {
        for (int jj=(N-1)*incx; 0<=jj; jj -=incx)
          x[jj] /= alpha;
      }
    }
    else if (is_upper_triangular())
    {
      
      for (int j=0, jj=0; j<N; j++, jj += incx)
      {
        x[jj] /= diag[j];
        sp_axpy(- x[jj], S[j], x, incx);
      }
      if (alpha != (T) 1.0)
      {
        for (int jj=(N-1)*incx; 0<=jj; jj -=incx)
          x[jj] /= alpha;
      }
    }
    else
         ERROR_RETURN(1);

    return 0;
  }

  int transpose_triangular_conj_solve(T alpha, T* x, int incx) const
  {
    if ( ! is_triangular())
      return -1;

    int N = num_rows();

    if (is_lower_triangular())
    {

      for (int j=N-1, jj=(N-1)*incx; 0<=j; j--, jj -= incx)
      {
        x[jj] /= conj(diag[j]) ;
        sp_conj_axpy( -x[jj], S[j], x, incx);
      }
      if (alpha != (T) 1.0)
      {
        for (int jj=(N-1)*incx; 0<=jj; jj -=incx)
          x[jj] /= alpha;
      }
    }
    else if (is_upper_triangular())
    {
      
      for (int j=0, jj=0; j<N; j++, jj += incx)
      {
        x[jj] /= conj(diag[j]);
        sp_conj_axpy(- x[jj], S[j], x, incx);
      }
      if (alpha != (T) 1.0)
      {
        for (int jj=(N-1)*incx; 0<=jj; jj -=incx)
          x[jj] /= alpha;
      }
    }
    else
         ERROR_RETURN(1);

    return 0;
  }

 public:
  inline T& val(pair<T, int> &VP) { return VP.first; }
  inline int& col_index(pair<T,int> &VP) { return VP.second; } 

  inline const T& val(pair<T, int> const &VP) const { return VP.first; }
  inline int col_index(pair<T,int> const &VP) const { return VP.second; } 

  TSp_mat( int M, int N) : Sp_mat(M,N), S(M), diag() {}

  void destroy()
  {
    // set vector sizes to zero
    (vector<T>(0)).swap(diag);
    (vector< vector< pair<T, int> > > (0) ).swap(S);
  }

/**

    This function is the entry point for all of the insert routines in 
    this implementation.  It fills the sparse matrix, one entry at a time.
    If matrix is declared unit_diagonal, then inserting any diagonal
    values is ignored.  If it is symmetric (upper/lower) or triangular
    (upper/lower) inconsistent values are not caught.  (That is, entries
    into the upper region of a lower triangular matrix is not reported.)

    [NOTE: the base is determined at the creation phase, and can be determined
    by testing whether  BLAS_usgp(A, blas_one_base) returns 1.  If it returns 0,
    then offsets are zero based.]

    @param val  the numeric value of entry A(i,j)
    @param i  the row index of A(i,j)  
    @param j  the column index of A(i,j)

    @return 0 if succesful, 1 otherwise
*/  
  int insert_entry(T val, int i, int j)
  {
    if (is_one_base())        
    {
      i--;
      j--;
    }

    /* make sure the indices are in range */
    ASSERT_RETURN(i >= 0, 1);
    ASSERT_RETURN(i < num_rows(), 1);
    ASSERT_RETURN(j >= 0, 1);
    ASSERT_RETURN(j < num_cols(), 1);

    /* allocate space for the diagonal, if this is the first time
     * trying to insert values.
    */
    if (is_new())
    {
      set_open();
      
      if (is_triangular() || is_symmetric())
      {
        diag.resize(num_rows());

        if (is_unit_diag())
        {
          for (unsigned int ii=0; ii< diag.size(); ii++)
              diag[ii] = T(1.0); 
        }
        else
        {
          for (unsigned int ii=0; ii< diag.size(); ii++)
              diag[ii] = (T) 0.0; 
        }
      }

    }
    if (is_open())
    {

      if (i==j && (is_triangular() || is_symmetric() || is_hermitian()) )
      {
        if (!is_unit_diag())
        {
          diag[i] += val;
        }
        else /* if unit diagonal */
        {
          if (val != (T) 1) 
            ERROR_RETURN(0);    /* tries to insert non-unit diagonal */
        }

        if (is_upper_storage() && i > j)
            ERROR_RETURN(0);    /* tries to fill lower-triangular region */
        else 
        
          if (is_lower_storage() && i < j)
            ERROR_RETURN(0);  /* tries to fill upper-triangular region */

      }
      else
      {
        S[i].push_back( make_pair(val, j) );
      }

      num_nonzeros() ++;
    }

    return 0;
  }

  int insert_entries( int nz, const T* Val, const int *I, const int *J)
  {
    for (int i=0; i<nz; i++)
    {
      insert_entry(Val[i], I[i], J[i]) ;
    }
    return 0;

  }

  int insert_row(int k, int nz, const T* Val, const int *J)
  {
    for (int i=0; i<nz; i++)
      insert_entry(Val[i], k, J[i]);  
    return 0;
  }

  int insert_col(int k, int nz, const T* Val, const int *I)
  {
    for (int i=0; i<nz; i++)
      insert_entry(Val[i], I[i], k);  
    return 0;
  }

  int insert_block(const T* Val, int row_stride, 
        int col_stride, int bi, int bj)
  {
    /* translate from block index to global indices */
    int Iend = K(bi+1);
    int Jend = L(bj+1);
    for (int i=K(bi), r=0; i<Iend; i++, r += row_stride)
      for (int j=L(bi); j<Jend; j++, r += col_stride)
        insert_entry( Val[r], i, j );

    return 0;
  }

  int end_construction()
  {
    return Sp_mat::end_construction();
  }

  int usmv(enum blas_trans_type transa, const T& alpha, const  T* x , int incx, 
    T* y, int incy) const
  {
    
  ASSERT_RETURN(is_valid(), -1);

  if (transa == blas_no_trans)
    mult_vec(alpha, x, incx, y, incy);
  else
  if (transa == blas_conj_trans)
    mult_vec_conj_transpose(alpha, x, incx, y, incy);
  else
  if ( transa == blas_trans)
    mult_vec_transpose(alpha, x, incx, y, incy);
  else
    ERROR_RETURN(1);
  
    return 0;
  }

  int usmm(enum blas_order_type ordera, enum blas_trans_type transa, 
    int nrhs, const T& alpha, const  T* b, int ldb, T* C, int ldC) const
  {
    if (ordera == blas_rowmajor)
    {
      /* for each column of C, perform a mat_vec */
      for (int i=0; i<nrhs; i++)
      {
        usmv( transa, alpha, &b[i], ldb, &C[i], ldC );
      }
      return 0;
    }
    else
    if (ordera == blas_colmajor)
    {
      /* for each column of C, perform a mat_vec */
      for (int i=0; i<nrhs; i++)
      {
        usmv( transa, alpha, &b[i*ldb], 1, &C[i*ldC], 1 );
      }
      return 0;
    }
    else
      ERROR_RETURN(1);
  }

  int ussv( enum blas_trans_type transa, const T& alpha,  T* x, int incx) const
  {
      if (transa == blas_trans)
        return transpose_triangular_solve(alpha, x, incx);
      else 
      if (transa == blas_conj_trans)
        return transpose_triangular_conj_solve(alpha, x, incx);
      else
      if (transa == blas_no_trans)
        return triangular_solve(alpha, x, incx);
      else
        ERROR_RETURN(1);
  }

  int ussm( enum blas_order_type ordera, enum blas_trans_type transa, int nrhs,
      const T& alpha, T* C, int ldC) const
  {
    if (ordera == blas_rowmajor)
    {
      /* for each column of C, perform a usmv */
      for (int i=0; i<nrhs; i++)
      {
        ussv( 
            transa, alpha, &C[i], ldC );
      }
      return 0;
    }
    else
    if (ordera == blas_colmajor)
    {
      /* for each column of C, perform a mat_vec */
      for (int i=0; i<nrhs; i++)
      {
        ussv( transa, alpha, &C[i*ldC], 1 );
      }
      return 0;
    }
    else
      ERROR_RETURN(1);
  } 

  void print() const
  {
    Sp_mat::print();  /* print matrix header info */

    /* if there is actual data, print out contents */
    for (int i=0; i<num_rows(); i++)
      for (unsigned int j=0; j< S[i].size(); j++)
        cout << i << "    " << col_index(S[i][j]) <<
              "        "  << val(S[i][j]) << "\n";

    /* if matrix is triangular, print out diagonals */
    if (is_upper_triangular() || is_lower_triangular())
    {
      for (unsigned int i=0; i< diag.size(); i++)
        cout << i << "    " << i << "     " << diag[i] << "\n";
    }
  }
};

typedef TSp_mat<float> FSp_mat;
typedef TSp_mat<double> DSp_mat;
typedef TSp_mat<complex<float> > CSp_mat;
typedef TSp_mat<complex<double> > ZSp_mat;

void table_print();
void print(int A);

} /* namespace NIST_SPBLAS */

namespace NIST_SPBLAS
{
static vector<Sp_mat *> Table;
static unsigned int Table_active_matrices = 0;
int Table_insert(Sp_mat* S);
int Table_remove(unsigned int i);

/* dummy variable to use Table_active_matrices
 * use "&" to use Table_active_matrices and the dummy variable itself
 * to avoid unused static variable warning for Table_active_matrices
 * Jichao Sun
 * Last modified: Jan 08, 2012
 */
static unsigned int dummy_static_variable = 
	&dummy_static_variable - &Table_active_matrices;

} /* namespace NIST_SPBLAS */

#endif /* NIST_SPBLAS_ */
