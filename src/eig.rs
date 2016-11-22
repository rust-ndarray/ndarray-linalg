//! Implement eigenvalue decomposition of general matrix

use lapack::fortran::*;
use num_traits::Zero;

use error::LapackError;

pub trait ImplEig: Sized {
    fn eig(n: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>, Vec<Self>), LapackError>;
}

macro_rules! impl_eig {
    ($scalar:ty, $geev:path) => {
impl ImplEig for $scalar {
    fn eig(n: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>, Vec<Self>), LapackError> {
        let lda = n as i32;
        let ldvr = n as i32;
        let ldvl = n as i32;
        let lw_default = 1000;
        let mut wr = vec![Self::zero(); n];
        let mut wi = vec![Self::zero(); n];
        let mut vr = vec![Self::zero(); n * n];
        let mut vl = Vec::new();
        let mut work = vec![Self::zero(); lw_default];
        let mut info = 0;
        $geev(b'N', b'V', n as i32, &mut a, lda, &mut wr, &mut wi,
              &mut vl, ldvl, &mut vr, ldvr, &mut work, -1, &mut info);
        let lwork = work[0] as i32;
        if lwork > lw_default as i32 {
            work = vec![Self::zero(); lwork as usize];
        }
        $geev(b'N', b'V', n as i32, &mut a, lda, &mut wr, &mut wi,
              &mut vl, ldvl, &mut vr, ldvr, &mut work, lwork, &mut info);
        if info == 0 {
            Ok((wr, wi, vr))
        } else {
            Err(From::from(info))
        }
    }
}
}} // end macro_rules

impl_eig!(f64, dgeev);
impl_eig!(f32, sgeev);
