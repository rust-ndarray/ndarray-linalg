//! Implement SVD

use lapack::fortran::*;
use num_traits::Zero;

use error::LapackError;

pub trait ImplSVD: Sized {
    fn svd(n: usize, m: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>, Vec<Self>), LapackError>;
}

macro_rules! impl_svd {
    ($scalar:ty, $gesvd:path) => {
impl ImplSVD for $scalar {
    fn svd(n: usize,
           m: usize,
           mut a: Vec<Self>)
           -> Result<(Vec<Self>, Vec<Self>, Vec<Self>), LapackError> {
        let mut info = 0;
        let n = n as i32;
        let m = m as i32;
        let lda = m;
        let ldu = m;
        let ldvt = n;
        let lwork = -1;
        let lw_default = 1000;
        let mut u = vec![Self::zero(); (ldu * m) as usize];
        let mut vt = vec![Self::zero(); (ldvt * n) as usize];
        let mut s = vec![Self::zero(); n as usize];
        let mut work = vec![Self::zero(); lw_default];
        $gesvd('A' as u8,
               'A' as u8,
               m,
               n,
               &mut a,
               lda,
               &mut s,
               &mut u,
               ldu,
               &mut vt,
               ldvt,
               &mut work,
               lwork,
               &mut info); // calc optimal work
        let lwork = work[0] as i32;
        if lwork > lw_default as i32 {
            work = vec![Self::zero(); lwork as usize];
        }
        $gesvd('A' as u8,
               'A' as u8,
               m,
               n,
               &mut a,
               lda,
               &mut s,
               &mut u,
               ldu,
               &mut vt,
               ldvt,
               &mut work,
               lwork,
               &mut info);
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
