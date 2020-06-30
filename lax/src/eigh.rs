//! Eigenvalue decomposition for Hermite matrices

use super::*;
use crate::{error::*, layout::MatrixLayout, types::*};
use num_traits::Zero;

/// Wraps `*syev` for real and `*heev` for complex
pub trait Eigh_: Scalar {
    unsafe fn eigh(
        calc_eigenvec: bool,
        l: MatrixLayout,
        uplo: UPLO,
        a: &mut [Self],
    ) -> Result<Vec<Self::Real>>;
    unsafe fn eigh_generalized(
        calc_eigenvec: bool,
        l: MatrixLayout,
        uplo: UPLO,
        a: &mut [Self],
        b: &mut [Self],
    ) -> Result<Vec<Self::Real>>;
}

macro_rules! impl_eigh {
    ($scalar:ty, $ev:path, $evg:path) => {
        impl Eigh_ for $scalar {
            unsafe fn eigh(
                calc_v: bool,
                l: MatrixLayout,
                uplo: UPLO,
                mut a: &mut [Self],
            ) -> Result<Vec<Self::Real>> {
                let (n, _) = l.size();
                let jobz = if calc_v { b'V' } else { b'N' };
                let mut w = vec![Self::Real::zero(); n as usize];
                $ev(l.lapacke_layout(), jobz, uplo as u8, n, &mut a, n, &mut w)
                    .as_lapack_result()?;
                Ok(w)
            }

            unsafe fn eigh_generalized(
                calc_v: bool,
                l: MatrixLayout,
                uplo: UPLO,
                mut a: &mut [Self],
                mut b: &mut [Self],
            ) -> Result<Vec<Self::Real>> {
                let (n, _) = l.size();
                let jobz = if calc_v { b'V' } else { b'N' };
                let mut w = vec![Self::Real::zero(); n as usize];
                $evg(
                    l.lapacke_layout(),
                    1,
                    jobz,
                    uplo as u8,
                    n,
                    &mut a,
                    n,
                    &mut b,
                    n,
                    &mut w,
                )
                .as_lapack_result()?;
                Ok(w)
            }
        }
    };
} // impl_eigh!

impl_eigh!(f64, lapacke::dsyev, lapacke::dsygv);
impl_eigh!(f32, lapacke::ssyev, lapacke::ssygv);
impl_eigh!(c64, lapacke::zheev, lapacke::zhegv);
impl_eigh!(c32, lapacke::cheev, lapacke::chegv);
