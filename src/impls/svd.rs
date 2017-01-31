//! Implement SVD

use std::cmp::min;
use lapack::c::*;
use num_traits::Zero;

use error::LapackError;

pub trait ImplSVD: Sized {
    fn svd(layout: Layout, n: usize, m: usize, a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>, Vec<Self>), LapackError>;
}

macro_rules! impl_svd {
    ($scalar:ty, $gesvd:path) => {
impl ImplSVD for $scalar {
    fn svd(layout: Layout, n: usize, m: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>, Vec<Self>), LapackError> {
        let k = min(n, m);
        let n = n as i32;
        let m = m as i32;
        let lda = match layout {
            Layout::RowMajor => n,
            Layout::ColumnMajor => m,
        };
        let ldu = m;
        let ldvt = n;
        let mut u = vec![Self::zero(); (ldu * m) as usize];
        let mut vt = vec![Self::zero(); (ldvt * n) as usize];
        let mut s = vec![Self::zero(); n as usize];
        let mut superb = vec![Self::zero(); k-2];
        let info = $gesvd(layout, 'A' as u8, 'A' as u8, m, n, &mut a, lda, &mut s, &mut u, ldu, &mut vt, ldvt, &mut superb);
        if info == 0 {
            Ok((u, s, vt))
        } else {
            Err(From::from(info))
        }
    }
}
}} // end macro_rules

impl_svd!(f64, dgesvd);
impl_svd!(f32, sgesvd);
