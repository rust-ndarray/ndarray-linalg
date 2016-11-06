
extern crate lapack;

use self::lapack::fortran::*;
use ndarray::LinalgScalar;
use num_traits::Zero;

use error::LapackError;

pub trait LapackScalar: LinalgScalar {
    fn eigh(n: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>), LapackError>;
    fn inv(size: usize, mut a: Vec<Self>) -> Result<Vec<Self>, LapackError>;
    fn norm_1(m: usize, n: usize, mut a: Vec<Self>) -> Self;
    fn norm_i(m: usize, n: usize, mut a: Vec<Self>) -> Self;
    fn norm_f(m: usize, n: usize, mut a: Vec<Self>) -> Self;
    fn svd(n: usize,
           m: usize,
           mut a: Vec<Self>)
           -> Result<(Vec<Self>, Vec<Self>, Vec<Self>), LapackError>;
}

impl LapackScalar for f64 {
    fn eigh(n: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>), LapackError> {
        let mut w = vec![Self::zero(); n];
        let mut work = vec![Self::zero(); 4 * n];
        let mut info = 0;
        dsyev(b'V',
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
        dgetrf(n, n, &mut a, lda, &mut ipiv, &mut info);
        if info != 0 {
            return Err(From::from(info));
        }
        let lwork = n;
        let mut work = vec![Self::zero(); size];
        dgetri(n, &mut a, lda, &mut ipiv, &mut work, lwork, &mut info);
        if info == 0 {
            Ok(a)
        } else {
            Err(From::from(info))
        }
    }
    fn norm_1(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = Vec::<Self>::new();
        dlange(b'o', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
    fn norm_i(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = vec![Self::zero(); m];
        dlange(b'i', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
    fn norm_f(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = Vec::<Self>::new();
        dlange(b'f', m as i32, n as i32, &mut a, m as i32, &mut work)
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
        let lwork = -1;
        let lw_default = 1000;
        let mut u = vec![Self::zero(); (ldu * m) as usize];
        let mut vt = vec![Self::zero(); (ldvt * n) as usize];
        let mut s = vec![Self::zero(); n as usize];
        let mut work = vec![Self::zero(); lw_default];
        dgesvd('A' as u8,
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
        dgesvd('A' as u8,
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
