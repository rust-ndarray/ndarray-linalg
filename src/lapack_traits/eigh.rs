//! Eigenvalue decomposition for Hermite matrices

use lapack::c;
use num_traits::Zero;

use types::*;
use error::*;
use layout::Layout;

use super::{into_result, UPLO};

/// Wraps `*syev` for real and `*heev` for complex
pub trait Eigh_: AssociatedReal {
    fn eigh(calc_eigenvec: bool, Layout, UPLO, a: &mut [Self]) -> Result<Vec<Self::Real>>;
}

macro_rules! impl_eigh {
    ($scalar:ty, $ev:path) => {
impl Eigh_ for $scalar {
    fn eigh(calc_v: bool, l: Layout, uplo: UPLO, mut a: &mut [Self]) -> Result<Vec<Self::Real>> {
        let (n, _) = l.size();
        let jobz = if calc_v { b'V' } else { b'N' };
        let mut w = vec![Self::Real::zero(); n as usize];
        let info = $ev(l.lapacke_layout(), jobz, uplo as u8, n, &mut a, n, &mut w);
        into_result(info, w)
    }
}
}} // impl_eigh!

impl_eigh!(f64, c::dsyev);
impl_eigh!(f32, c::ssyev);
impl_eigh!(c64, c::zheev);
impl_eigh!(c32, c::cheev);
