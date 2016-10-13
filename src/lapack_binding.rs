
extern crate lapack;

use self::lapack::fortran::*;
use error::LapackError;

/// Eigenvalue decomposition for Hermite matrix
pub trait Eigh: Sized {
    /// execute *syev subroutine
    fn eigh(row_size: usize, matrix: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>), LapackError>;
}

impl Eigh for f64 {
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
}

impl Eigh for f32 {
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
}
