//! Implement linear solver and inverse matrix

use lapack::fortran::*;
use std::cmp::min;
use num_traits::Zero;

use error::LapackError;

pub trait ImplSolve: Sized {
    fn inv(size: usize, mut a: Vec<Self>) -> Result<Vec<Self>, LapackError>;
    fn lu(m: usize, n: usize, mut a: Vec<Self>) -> Result<(Vec<i32>, Vec<Self>), LapackError>;
}

macro_rules! impl_solve {
    ($scalar:ty, $getrf:path, $getri:path) => {
impl ImplSolve for $scalar {
    fn inv(size: usize, mut a: Vec<Self>) -> Result<Vec<Self>, LapackError> {
        let n = size as i32;
        let lda = n;
        let mut ipiv = vec![0; size];
        let mut info = 0;
        $getrf(n, n, &mut a, lda, &mut ipiv, &mut info);
        if info != 0 {
            return Err(From::from(info));
        }
        let lwork = n;
        let mut work = vec![Self::zero(); size];
        $getri(n, &mut a, lda, &mut ipiv, &mut work, lwork, &mut info);
        if info == 0 {
            Ok(a)
        } else {
            Err(From::from(info))
        }
    }
    fn lu(m: usize, n: usize, mut a: Vec<Self>) -> Result<(Vec<i32>, Vec<Self>), LapackError> {
        let m = m as i32;
        let n = n as i32;
        let k = min(m, n);
        let lda = m;
        let mut ipiv = vec![0; k as usize];
        let mut info = 0;
        $getrf(m, n, &mut a, lda, &mut ipiv, &mut info);
        if info == 0 {
            Ok((ipiv, a))
        } else {
            Err(From::from(info))
        }
    }
}
}} // end macro_rules

impl_solve!(f64, dgetrf, dgetri);
impl_solve!(f32, sgetrf, sgetri);
