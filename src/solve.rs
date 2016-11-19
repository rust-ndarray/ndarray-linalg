//! Implement linear solver and inverse matrix

use lapack::c::*;
use std::cmp::min;

use error::LapackError;

pub trait ImplSolve: Sized {
    fn inv(layout: Layout, size: usize, a: Vec<Self>) -> Result<Vec<Self>, LapackError>;
    fn lu(layout: Layout,
          m: usize,
          n: usize,
          a: Vec<Self>)
          -> Result<(Vec<i32>, Vec<Self>), LapackError>;
}

macro_rules! impl_solve {
    ($scalar:ty, $getrf:path, $getri:path, $laswp:path) => {
impl ImplSolve for $scalar {
    fn inv(layout: Layout, size: usize, mut a: Vec<Self>) -> Result<Vec<Self>, LapackError> {
        let n = size as i32;
        let lda = n;
        let mut ipiv = vec![0; size];
        let info = $getrf(layout, n, n, &mut a, lda, &mut ipiv);
        if info != 0 {
            return Err(From::from(info));
        }
        let info = $getri(layout, n, &mut a, lda, &mut ipiv);
        if info == 0 {
            Ok(a)
        } else {
            Err(From::from(info))
        }
    }
    fn lu(layout: Layout, m: usize, n: usize, mut a: Vec<Self>) -> Result<(Vec<i32>, Vec<Self>), LapackError> {
        let m = m as i32;
        let n = n as i32;
        let k = min(m, n);
        let lda = m;
        let mut ipiv = vec![0; k as usize];
        let info = $getrf(layout, m, n, &mut a, lda, &mut ipiv);
        if info == 0 {
            Ok((ipiv, a))
        } else {
            Err(From::from(info))
        }
    }
}
}} // end macro_rules

impl_solve!(f64, dgetrf, dgetri, dlaswp);
impl_solve!(f32, sgetrf, sgetri, slaswp);
