use super::*;
use crate::{error::*, layout::*};

pub fn cholesky_semi(l: MatrixLayout, uplo: UPLO, a: &mut [f64], rank: &mut i32) -> Result<Vec<i32>> {
    let (n, _) = l.size();
    let mut ipiv = vec![0; n as usize];
    let tol = 1.0e-12_f64;
    let info = unsafe {
        lapacke::dpstrf(
            l.lapacke_layout(),
            uplo as u8,
            n,
            a,
            l.lda(),
            &mut ipiv,
            rank,
            tol,
        )
    };
    if info < 0 {
        Err(LinalgError::Lapack { return_code: info })
    } else {
        Ok(ipiv)
    }
}
