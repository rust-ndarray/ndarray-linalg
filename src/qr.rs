//! Implement QR decomposition

extern crate lapack;

use std::cmp::min;
use self::lapack::fortran::*;
use num_traits::Zero;

use error::LapackError;

pub trait ImplQR: Sized {
    fn qr(n: usize, m: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>), LapackError>;
}

impl ImplQR for f64 {
    fn qr(n: usize, m: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>), LapackError> {
        let mut info = 0;
        let n = n as i32;
        let m = m as i32;
        let k = min(m, n);
        let lda = m;
        let lw_default = 1000;
        let mut tau = vec![Self::zero(); k as usize];
        let mut work = vec![Self::zero(); lw_default];
        // estimate lwork
        dgeqrf(m, n, &mut a, lda, &mut tau, &mut work, -1, &mut info);
        let lwork_r = work[0] as i32;
        if lwork_r > lw_default as i32 {
            work = vec![Self::zero(); lwork_r as usize];
        }
        // calc R
        dgeqrf(m, n, &mut a, lda, &mut tau, &mut work, lwork_r, &mut info);
        if info != 0 {
            return Err(From::from(info));
        }
        let r = a.clone();
        // re-estimate lwork
        dorgqr(m, n, k, &mut a, lda, &mut tau, &mut work, -1, &mut info);
        let lwork_q = work[0] as i32;
        if lwork_q > lwork_r {
            work = vec![Self::zero(); lwork_q as usize];
        }
        // calc Q
        dorgqr(m,
               n,
               k,
               &mut a,
               lda,
               &mut tau,
               &mut work,
               lwork_q,
               &mut info);
        if info == 0 {
            Ok((a, r))
        } else {
            Err(From::from(info))
        }
    }
}
