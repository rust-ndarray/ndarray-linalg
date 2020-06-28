//! QR decomposition

use num_traits::Zero;
use std::cmp::min;

use crate::error::*;
use crate::layout::MatrixLayout;
use crate::types::*;

use super::into_result;

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
                let info = $qrf(l.lapacke_layout(), row, col, &mut a, l.lda(), &mut tau);
                into_result(info, tau)
            }

            unsafe fn q(l: MatrixLayout, mut a: &mut [Self], tau: &[Self]) -> Result<()> {
                let (row, col) = l.size();
                let k = min(row, col);
                let info = $gqr(l.lapacke_layout(), row, k, k, &mut a, l.lda(), &tau);
                into_result(info, ())
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
