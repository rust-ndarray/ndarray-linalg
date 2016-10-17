
extern crate lapack;

use self::lapack::fortran::*;
use error::LapackError;
use ndarray::LinalgScalar;

pub trait LapackScalar: LinalgScalar {
    fn syev(jobz: u8,
            uplo: u8,
            n: i32,
            a: &mut Vec<Self>,
            lda: i32,
            w: &mut Vec<Self>,
            work: &mut Vec<Self>,
            lwork: i32,
            info: &mut i32);
    fn getrf(m: i32, n: i32, a: &mut Vec<Self>, lda: i32, ipiv: &mut [i32], info: &mut i32);
    fn getri(n: i32,
             a: &mut Vec<Self>,
             lda: i32,
             ipiv: &[i32],
             work: &mut Vec<Self>,
             lwork: i32,
             info: &mut i32);
    fn lange(norm: u8, m: i32, n: i32, a: &Vec<Self>, lda: i32, work: &mut Vec<Self>) -> Self;

    fn eigh(n: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>), LapackError> {
        let mut w = vec![Self::zero(); n];
        let mut work = vec![Self::zero(); 4 * n];
        let mut info = 0;
        Self::syev(b'V',
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
        Self::getrf(n, n, &mut a, lda, &mut ipiv, &mut info);
        if info != 0 {
            return Err(From::from(info));
        }
        let lwork = n;
        let mut work = vec![Self::zero(); size];
        Self::getri(n, &mut a, lda, &mut ipiv, &mut work, lwork, &mut info);
        if info == 0 {
            Ok(a)
        } else {
            Err(From::from(info))
        }
    }
    fn norm_1(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = Vec::<Self>::new();
        Self::lange(b'o', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
    fn norm_i(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = vec![Self::zero(); m];
        Self::lange(b'i', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
    fn norm_f(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = Vec::<Self>::new();
        Self::lange(b'f', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
}

impl LapackScalar for f64 {
    fn syev(jobz: u8,
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
    fn getrf(m: i32, n: i32, a: &mut Vec<Self>, lda: i32, ipiv: &mut [i32], info: &mut i32) {
        dgetrf(m, n, a, lda, ipiv, info);
    }
    fn getri(n: i32,
             a: &mut Vec<Self>,
             lda: i32,
             ipiv: &[i32],
             work: &mut Vec<Self>,
             lwork: i32,
             info: &mut i32) {
        dgetri(n, a, lda, ipiv, work, lwork, info);
    }
    fn lange(norm: u8, m: i32, n: i32, a: &Vec<Self>, lda: i32, work: &mut Vec<Self>) -> Self {
        dlange(norm, m, n, a, lda, work)
    }
}
