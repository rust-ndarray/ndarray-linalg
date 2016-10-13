
extern crate lapack;

use self::lapack::fortran::*;
use error::LapackError;
use ndarray::LinalgScalar;

/// Eigenvalue decomposition for Hermite matrix
pub trait LapackScalar: LinalgScalar {
    /// execute *syev subroutine
    fn eigh(row_size: usize, matrix: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>), LapackError>;
    fn norm_1(rows: usize, cols: usize, matrix: Vec<Self>) -> Self;
    fn norm_i(rows: usize, cols: usize, matrix: Vec<Self>) -> Self;
    fn norm_f(rows: usize, cols: usize, matrix: Vec<Self>) -> Self;
}

impl LapackScalar for f64 {
    fn eigh(n: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>), LapackError> {
        let mut w = vec![0.0; n ];
        let mut work = vec![0.0; 4 * n ];
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

    fn norm_1(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = Vec::<Self>::new();
        dlange(b'o', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
    fn norm_i(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = vec![0.0; m];
        dlange(b'i', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
    fn norm_f(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = Vec::<Self>::new();
        dlange(b'f', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
}

impl LapackScalar for f32 {
    fn eigh(n: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>), LapackError> {
        let mut w = vec![0.0; n];
        let mut work = vec![0.0; 4 * n];
        let mut info = 0;
        ssyev(b'V',
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
    fn norm_1(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = Vec::<Self>::new();
        slange(b'o', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
    fn norm_i(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = vec![0.0; m];
        slange(b'i', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
    fn norm_f(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = Vec::<Self>::new();
        slange(b'f', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
}
