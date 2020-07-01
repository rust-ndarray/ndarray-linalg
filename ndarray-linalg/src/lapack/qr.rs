//! QR decomposition

use super::*;
use crate::{error::*, layout::MatrixLayout, types::*};
use num_traits::Zero;
use std::cmp::min;

/// Wraps `*geqrf` and `*orgqr` (`*ungqr` for complex numbers)
pub trait QR_: Sized {
    unsafe fn householder(l: MatrixLayout, a: &mut [Self]) -> Result<Vec<Self>>;
    unsafe fn q(l: MatrixLayout, a: &mut [Self], tau: &[Self]) -> Result<()>;
    unsafe fn qr(l: MatrixLayout, a: &mut [Self]) -> Result<Vec<Self>>;
}

macro_rules! impl_qr {
    ($scalar:ty, $qrf:path, $gqr:path) => {
        impl QR_ for $scalar {
            unsafe fn householder(l: MatrixLayout, mut a: &mut [Self]) -> Result<Vec<Self>> {
                let (row, col) = l.size();
                let k = min(row, col);
                let mut tau = vec![Self::zero(); k as usize];
                $qrf(l.lapacke_layout(), row, col, &mut a, l.lda(), &mut tau).as_lapack_result()?;
                Ok(tau)
            }

            unsafe fn q(l: MatrixLayout, mut a: &mut [Self], tau: &[Self]) -> Result<()> {
                let (row, col) = l.size();
                let k = min(row, col);
                $gqr(l.lapacke_layout(), row, k, k, &mut a, l.lda(), &tau).as_lapack_result()?;
                Ok(())
            }

            unsafe fn qr(l: MatrixLayout, a: &mut [Self]) -> Result<Vec<Self>> {
                let tau = Self::householder(l, a)?;
                let r = Vec::from(&*a);
                Self::q(l, a, &tau)?;
                Ok(r)
            }
        }
    };
} // endmacro

impl_qr!(f64, lapacke::dgeqrf, lapacke::dorgqr);
impl_qr!(f32, lapacke::sgeqrf, lapacke::sorgqr);
impl_qr!(c64, lapacke::zgeqrf, lapacke::zungqr);
impl_qr!(c32, lapacke::cgeqrf, lapacke::cungqr);
