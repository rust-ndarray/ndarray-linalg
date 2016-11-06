
extern crate lapack;

use self::lapack::fortran::*;
use num_traits::Zero;

use error::LapackError;

pub trait ImplSolve: Sized {
    fn inv(size: usize, mut a: Vec<Self>) -> Result<Vec<Self>, LapackError>;
}

impl ImplSolve for f64 {
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
}
