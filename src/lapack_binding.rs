
extern crate lapack;

use self::lapack::fortran::*;
use error::LapackError;

/// wrapper for *syev functions in LAPACK
pub trait Eigh: Sized {
    /// execute *syev subroutine
    fn syev(row_size: i32, matrix: &mut [Self]) -> Result<Vec<Self>, LapackError>;
}

impl Eigh for f64 {
    fn syev(n: i32, a: &mut [Self]) -> Result<Vec<Self>, LapackError> {
        let mut w = vec![0.0; n as usize];
        let mut work = vec![0.0; 4 * n as usize];
        let mut info = 0;
        dsyev(b'V', b'U', n, a, n, &mut w, &mut work, 4 * n, &mut info);
        if info == 0 {
            Ok(w)
        } else {
            Err(LapackError { return_code: info })
        }
    }
}

impl Eigh for f32 {
    fn syev(n: i32, a: &mut [Self]) -> Result<Vec<Self>, LapackError> {
        let mut w = vec![0.0; n as usize];
        let mut work = vec![0.0; 4 * n as usize];
        let mut info = 0;
        ssyev(b'V', b'U', n, a, n, &mut w, &mut work, 4 * n, &mut info);
        if info == 0 {
            Ok(w)
        } else {
            Err(LapackError { return_code: info })
        }
    }
}
