
use std::cmp::min;
use ndarray::LinalgScalar;
use num_traits::float::Float;

use error::LapackError;
use binding;

pub trait LapackScalar: LinalgScalar + binding::LapackBinding {
    fn eigh(n: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>), LapackError> {
        let mut w = vec![Self::zero(); n];
        let mut work = vec![Self::zero(); 4 * n];
        let mut info = 0;
        Self::_syev(b'V',
                    b'U',
                    n as i32,
                    &mut a,
                    n as i32,
                    &mut w,
                    &mut work,
                    4 * n as i32,
                    &mut info);
        if info == 0 {
            Ok((w, a))
        } else {
            Err(From::from(info))
        }
    }
    fn inv(size: usize, mut a: Vec<Self>) -> Result<Vec<Self>, LapackError> {
        let n = size as i32;
        let lda = n;
        let mut ipiv = vec![0; size];
        let mut info = 0;
        Self::_getrf(n, n, &mut a, lda, &mut ipiv, &mut info);
        if info != 0 {
            return Err(From::from(info));
        }
        let lwork = n;
        let mut work = vec![Self::zero(); size];
        Self::_getri(n, &mut a, lda, &mut ipiv, &mut work, lwork, &mut info);
        if info == 0 {
            Ok(a)
        } else {
            Err(From::from(info))
        }
    }
    fn norm_1(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = Vec::<Self>::new();
        Self::_lange(b'o', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
    fn norm_i(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = vec![Self::zero(); m];
        Self::_lange(b'i', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
    fn norm_f(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = Vec::<Self>::new();
        Self::_lange(b'f', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
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
        let lwmax = 1000; // XXX
        let lwork = -1;
        let mut u = vec![Self::zero(); (ldu * m) as usize];
        let mut vt = vec![Self::zero(); (ldvt * n) as usize];
        let mut s = vec![Self::zero(); n as usize];
        let mut work = vec![Self::zero(); lwmax];
        Self::_gesvd('A' as u8,
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
        let lwork = min(lwmax as i32, work[0] as i32);
        Self::_gesvd('A' as u8,
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
        if info == 0 {
            Ok((u, s, vt))
        } else {
            Err(From::from(info))
        }

    }
}

impl LapackScalar for f64 {}
impl LapackScalar for f32 {}
