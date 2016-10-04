
extern crate lapack;
use self::lapack::fortran::*;

use std::error;
use std::fmt;

#[derive(Debug)]
struct LapackError {
    return_code: i32,
}

impl fmt::Display for LapackError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LAPACK: return_code = {}", self.return_code)
    }
}

impl error::Error for LapackError {
    fn description(&self) -> &str {
        "LAPACK subroutine returns non-zero code"
    }
}

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
