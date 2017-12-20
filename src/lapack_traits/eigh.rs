//! Eigenvalue decomposition for Hermite matrices

use lapacke;
use num_traits::Zero;

use error::*;
use layout::MatrixLayout;
use types::*;

use super::{UPLO, into_result};

/// Wraps `*syev` for real and `*heev` for complex
pub trait Eigh_: AssociatedReal {
    unsafe fn eigh(calc_eigenvec: bool, MatrixLayout, UPLO, a: &mut [Self]) -> Result<Vec<Self::Real>>;
}

macro_rules! impl_eigh {
    ($scalar:ty, $ev:path) => {
impl Eigh_ for $scalar {
    unsafe fn eigh(calc_v: bool, l: MatrixLayout, uplo: UPLO, mut a: &mut [Self]) -> Result<Vec<Self::Real>> {
        let (n, _) = l.size();
        let jobz = if calc_v { b'V' } else { b'N' };
        let mut w = vec![Self::Real::zero(); n as usize];
        let info = $ev(l.lapacke_layout(), jobz, uplo as u8, n, &mut a, n, &mut w);
        into_result(info, w)
    }
}
}} // impl_eigh!

impl_eigh!(f64, lapacke::dsyev);
impl_eigh!(f32, lapacke::ssyev);
impl_eigh!(c64, lapacke::zheev);
impl_eigh!(c32, lapacke::cheev);
