/* DenMatSin.h */

/* DenMatSin (row-order dense single-float matrix) class
 * Jichao Sun (js87@njit.edu)
 *
 * July 30, 2013
 *   added set(), unset() and isSet() functions
 *
 * June 5, 2013 Initialized based on DenMatSin.h (last modified on Mar
 * 14, 2013) from kprop.6 
 */

#ifndef KPROP_DENMATSIN_H_
#define KPROP_DENMATSIN_H_

#include <fstream>
using std::ostream;
#include <string>
using std::string;
#include "nist_spblas/nist_spblas.h" /* for sparse matrix multiplication */

namespace kprop
{

class DenMatSin
{
    public:
        /* constructor that creates a 0-by-0 matrix */
        DenMatSin();
        /* constructor that creates a m-by-n zero matrix */
        DenMatSin(int m, int n);
        /* constructor that creates a matrix from a file */
        DenMatSin(const string& filename, bool ascii);
        /* destructor */
        ~DenMatSin();
        /* #(rows) */
        int getM() const;
        /* #(columns) */
        int getN() const;
        /* saves a matrix to a file
         * a non-negative prec denotes the number of digits after the
         * decimal points when saving to an ascii file (-1 for default %f)
         */
        void save(const string& filename, bool ascii, int prec) const;
        /* returns true when the matrix is empty */
        bool isEmpty() const;
        /* max value of the matrix */
        float max() const;
        /* min value of the matrix */
        float min() const;
        /* re-initializes the matrix to a zero matrix */
        void reinit();
        /* to copy between two same-sized non-empty matrices
         * do *NOT* use the default = operator
         */
        void copy(const DenMatSin& mat);
        /* To print at most x rows and columns of the matrix, (x<=0 for all)
         * and y digits after the decimal point for each value.
         */
        void print(int x, int y) const;
        /* accesses an element (writable) */
        float& operator() (int i, int j);
        /* accesses an element (read-only) */
        const float& operator() (int i, int j) const;
        /* reinitializes this matrix to a transpose of another matrix */
        void transpose(const DenMatSin& mat);
        /* redirects a matrix to an output stream */
        friend ostream& operator<< (ostream& os, const DenMatSin& mat);

        /* set a value to non-zero (1.0F) */
        void set(int i, int j);
        /* set a value to zero (0.0F) */
        void unset(int i, int j);
        /* get whether a value is non-zero */
        bool isSet(int i, int j) const;

    private:
        /* number of rows */
        int m_;
        /* number of columns*/
        int n_;
        /* float numbers saved in row-major order */
        float*	data_;
        /* error if index is out of bound */
        void checkBounds(int x, int y) const;
}; /* class DenMatSin */

ostream& operator<< (ostream& os, const DenMatSin& mat);

//  /* matrix multiplication C=A*B where A, B and C are of DenMatSin
//   * requires libcblas
//   */
//  void MatMultiply(const DenMatSin& A, const DenMatSin& B, DenMatSin& C);

/* matrix multiplication C=A*B where A is of NIST_SPBLAS::FSp_mat,
 * and B and C are of DenMatSin
 * requires NIST spblas library
 */
void MatMultiply2(const NIST_SPBLAS::FSp_mat& A,
                  const DenMatSin& B,
                  DenMatSin& C);

/* matrix addition C=A+B where A, B and C are of DenMatSin */
void MatAdd(const DenMatSin& A, const DenMatSin& B, DenMatSin& C);



} /* namespace kprop */

#endif /* KPROP_DENMATSIN_H_ */
