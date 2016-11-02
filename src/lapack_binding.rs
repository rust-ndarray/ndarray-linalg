
extern crate lapack;

use std::cmp;
use self::lapack::fortran::*;
use error::LapackError;
use ndarray::LinalgScalar;

pub trait LapackScalar: LinalgScalar {
    fn _syev(jobz: u8,
             uplo: u8,
             n: i32,
             a: &mut Vec<Self>,
             lda: i32,
             w: &mut Vec<Self>,
             work: &mut Vec<Self>,
             lwork: i32,
             info: &mut i32);
    fn _getrf(m: i32, n: i32, a: &mut Vec<Self>, lda: i32, ipiv: &mut [i32], info: &mut i32);
    fn _getri(n: i32,
              a: &mut Vec<Self>,
              lda: i32,
              ipiv: &[i32],
              work: &mut Vec<Self>,
              lwork: i32,
              info: &mut i32);
    fn _lange(norm: u8, m: i32, n: i32, a: &Vec<Self>, lda: i32, work: &mut Vec<Self>) -> Self;
    fn _geqp3(m: i32,
              n: i32,
              a: &mut Vec<Self>,
              lda: i32,
              jpvt: &mut [i32],
              tau: &mut Vec<Self>,
              work: &mut Vec<Self>,
              lwork: i32,
              info: &mut i32);
    fn _orgqr(m: i32,
              n: i32,
              k: i32,
              a: &mut Vec<Self>,
              lda: i32,
              tau: &Vec<Self>,
              work: &mut Vec<Self>,
              lwork: i32,
              info: &mut i32);

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
    fn qr(m: usize, n: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>), LapackError> {
        let m = m as i32;
        let n = n as i32;
        let k = cmp::min(m, n);
        let lda = m;
        let lwork = 4 * n;
        let mut tau = vec![0.0; k as usize];
        let mut work = vec![0.0; lwork as usize];
        let mut info = 0;
        let mut jpvt = vec![0; n as usize];
        Self::_geqp3(m,
                     n,
                     &mut a,
                     lda,
                     &mut jpvt,
                     &mut tau,
                     &mut work,
                     lwork,
                     &mut info);
        if info != 0 {
            return Err(From::from(info));
        }
        let r = a.clone();
        Self::_orgqr(m, k, k, &mut a, lda, &mut tau, &mut work, lwork, &mut info);
        if info == 0 {
            Ok((a, r))
        } else {
            Err(From::from(info))
        }
    }
}

impl LapackScalar for f64 {
    fn _syev(jobz: u8,
             uplo: u8,
             n: i32,
             a: &mut Vec<Self>,
             lda: i32,
             w: &mut Vec<Self>,
             work: &mut Vec<Self>,
             lwork: i32,
             info: &mut i32) {
        dsyev(jobz, uplo, n, a, lda, w, work, lwork, info);
    }
    fn _getrf(m: i32, n: i32, a: &mut Vec<Self>, lda: i32, ipiv: &mut [i32], info: &mut i32) {
        dgetrf(m, n, a, lda, ipiv, info);
    }
    fn _getri(n: i32,
              a: &mut Vec<Self>,
              lda: i32,
              ipiv: &[i32],
              work: &mut Vec<Self>,
              lwork: i32,
              info: &mut i32) {
        dgetri(n, a, lda, ipiv, work, lwork, info);
    }
    fn _lange(norm: u8, m: i32, n: i32, a: &Vec<Self>, lda: i32, work: &mut Vec<Self>) -> Self {
        dlange(norm, m, n, a, lda, work)
    }
    fn _geqp3(m: i32,
              n: i32,
              a: &mut Vec<Self>,
              lda: i32,
              jpvt: &mut [i32],
              tau: &mut Vec<Self>,
              work: &mut Vec<Self>,
              lwork: i32,
              info: &mut i32) {
        dgeqp3(m,
               n,
               &mut a,
               lda,
               &mut jpvt,
               &mut tau,
               &mut work,
               lwork,
               &mut info);
    }
    fn _orgqr(m: i32,
              n: i32,
              k: i32,
              a: &mut Vec<Self>,
              lda: i32,
              tau: &Vec<Self>,
              work: &mut Vec<Self>,
              lwork: i32,
              info: &mut i32) {
        dorgqr(m, n, k, &mut a, lda, &mut tau, &mut work, lwork, &mut info);
    }
}

impl LapackScalar for f32 {
    fn _syev(jobz: u8,
             uplo: u8,
             n: i32,
             a: &mut Vec<Self>,
             lda: i32,
             w: &mut Vec<Self>,
             work: &mut Vec<Self>,
             lwork: i32,
             info: &mut i32) {
        ssyev(jobz, uplo, n, a, lda, w, work, lwork, info);
    }
    fn _getrf(m: i32, n: i32, a: &mut Vec<Self>, lda: i32, ipiv: &mut [i32], info: &mut i32) {
        sgetrf(m, n, a, lda, ipiv, info);
    }
    fn _getri(n: i32,
              a: &mut Vec<Self>,
              lda: i32,
              ipiv: &[i32],
              work: &mut Vec<Self>,
              lwork: i32,
              info: &mut i32) {
        sgetri(n, a, lda, ipiv, work, lwork, info);
    }
    fn _lange(norm: u8, m: i32, n: i32, a: &Vec<Self>, lda: i32, work: &mut Vec<Self>) -> Self {
        slange(norm, m, n, a, lda, work)
    }
    fn _geqp3(m: i32,
              n: i32,
              a: &mut Vec<Self>,
              lda: i32,
              jpvt: &mut [i32],
              tau: &mut Vec<Self>,
              work: &mut Vec<Self>,
              lwork: i32,
              info: &mut i32) {
        sgeqp3(m,
               n,
               &mut a,
               lda,
               &mut jpvt,
               &mut tau,
               &mut work,
               lwork,
               &mut info);
    }
    fn _orgqr(m: i32,
              n: i32,
              k: i32,
              a: &mut Vec<Self>,
              lda: i32,
              tau: &Vec<Self>,
              work: &mut Vec<Self>,
              lwork: i32,
              info: &mut i32) {
        sorgqr(m, n, k, &mut a, lda, &mut tau, &mut work, lwork, &mut info);
    }
}
