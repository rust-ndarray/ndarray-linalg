//! QR decomposition

use std::cmp::min;
use num_traits::Zero;
use lapack::c;

use types::*;
use error::*;
use layout::Layout;

use super::into_result;

/// Wraps `*geqrf` and `*orgqr` (`*ungqr` for complex numbers)
pub trait QR_: Sized {
    fn householder(Layout, a: &mut [Self]) -> Result<Vec<Self>>;
    fn q(Layout, a: &mut [Self], tau: &[Self]) -> Result<()>;
    fn qr(Layout, a: &mut [Self]) -> Result<Vec<Self>>;
}

macro_rules! impl_qr {
    ($scalar:ty, $qrf:path, $gqr:path) => {
impl QR_ for $scalar {
    fn householder(l: Layout, mut a: &mut [Self]) -> Result<Vec<Self>> {
        let (row, col) = l.size();
        let k = min(row, col);
        let mut tau = vec![Self::zero(); k as usize];
        let info = $qrf(l.lapacke_layout(), row, col, &mut a, l.lda(), &mut tau);
        into_result(info, tau)
    }

    fn q(l: Layout, mut a: &mut [Self], tau: &[Self]) -> Result<()> {
        let (row, col) = l.size();
        let k = min(row, col);
        let info = $gqr(l.lapacke_layout(), row, k, k, &mut a, l.lda(), &tau);
        into_result(info, ())
    }

    fn qr(l: Layout, mut a: &mut [Self]) -> Result<Vec<Self>> {
        let tau = Self::householder(l, a)?;
        let r = Vec::from(&*a);
        Self::q(l, a, &tau)?;
        Ok(r)
    }
}
}} // endmacro

impl_qr!(f64, c::dgeqrf, c::dorgqr);
impl_qr!(f32, c::sgeqrf, c::sorgqr);
impl_qr!(c64, c::zgeqrf, c::zungqr);
impl_qr!(c32, c::cgeqrf, c::cungqr);
