//! Implement linear solver and inverse matrix

use lapack::c::*;
use std::cmp::min;

use error::LapackError;

pub trait ImplSolve: Sized {
    /// execute LU decomposition
    fn lu(layout: Layout, m: usize, n: usize, a: Vec<Self>) -> Result<(Vec<i32>, Vec<Self>), LapackError>;
    /// calc inverse matrix with LU factorized matrix
    fn inv(layout: Layout, size: usize, a: Vec<Self>, ipiv: &Vec<i32>) -> Result<Vec<Self>, LapackError>;
    /// solve linear problem with LU factorized matrix
    fn solve(layout: Layout,
             size: usize,
             a: &Vec<Self>,
             ipiv: &Vec<i32>,
             b: Vec<Self>)
             -> Result<Vec<Self>, LapackError>;
    /// solve triangular linear problem
    fn solve_triangle<'a, 'b>(layout: Layout,
                              uplo: u8,
                              size: usize,
                              a: &'a [Self],
                              b: &'b mut [Self],
                              nrhs: i32)
                              -> Result<&'b mut [Self], LapackError>;
}

macro_rules! impl_solve {
    ($scalar:ty, $getrf:path, $getri:path, $getrs:path, $trtrs:path) => {
impl ImplSolve for $scalar {
    fn lu(layout: Layout, m: usize, n: usize, mut a: Vec<Self>) -> Result<(Vec<i32>, Vec<Self>), LapackError> {
        let m = m as i32;
        let n = n as i32;
        let k = min(m, n);
        let lda = match layout {
            Layout::ColumnMajor => m,
            Layout::RowMajor => n,
        };
        let mut ipiv = vec![0; k as usize];
        let info = $getrf(layout, m, n, &mut a, lda, &mut ipiv);
        if info == 0 {
            Ok((ipiv, a))
        } else {
            Err(From::from(info))
        }
    }
    fn inv(layout: Layout, size: usize, mut a: Vec<Self>, ipiv: &Vec<i32>) -> Result<Vec<Self>, LapackError> {
        let n = size as i32;
        let lda = n;
        let info = $getri(layout, n, &mut a, lda, &ipiv);
        if info == 0 {
            Ok(a)
        } else {
            Err(From::from(info))
        }
    }
    fn solve(layout: Layout, size: usize, a: &Vec<Self>, ipiv: &Vec<i32>, mut b: Vec<Self>) -> Result<Vec<Self>, LapackError> {
        let n = size as i32;
        let lda = n;
        let info = $getrs(layout, 'N' as u8, n, 1, a, lda, ipiv, &mut b, n);
        if info == 0 {
            Ok(b)
        } else {
            Err(From::from(info))
        }
    }
    fn solve_triangle<'a, 'b>(layout: Layout, uplo: u8, size: usize, a: &'a [Self], mut b: &'b mut [Self], nrhs: i32) -> Result<&'b mut [Self], LapackError> {
        let n = size as i32;
        let lda = n;
        let ldb = match layout {
            Layout::ColumnMajor => n,
            Layout::RowMajor => 1,
        };
        let info = $trtrs(layout, uplo, 'N' as u8, 'N' as u8, n, nrhs, a, lda, &mut b, ldb);
        if info == 0 {
            Ok(b)
        } else {
            Err(From::from(info))
        }
    }
}
}} // end macro_rules

impl_solve!(f64, dgetrf, dgetri, dgetrs, dtrtrs);
impl_solve!(f32, sgetrf, sgetri, sgetrs, strtrs);
