//! Eigenvalue decomposition for Symmetric/Hermite matrices

use super::*;
use crate::{error::*, layout::MatrixLayout};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

pub trait Eigh_: Scalar {
    /// Wraps `*syev` for real and `*heev` for complex
    fn eigh(
        calc_eigenvec: bool,
        layout: MatrixLayout,
        uplo: UPLO,
        a: &mut [Self],
    ) -> Result<Vec<Self::Real>>;

    /// Wraps `*syegv` for real and `*heegv` for complex
    fn eigh_generalized(
        calc_eigenvec: bool,
        layout: MatrixLayout,
        uplo: UPLO,
        a: &mut [Self],
        b: &mut [Self],
    ) -> Result<Vec<Self::Real>>;
}

macro_rules! impl_eigh {
    ($scalar:ty, $ev:path, $evg:path) => {
        impl Eigh_ for $scalar {
            fn eigh(
                calc_v: bool,
                layout: MatrixLayout,
                uplo: UPLO,
                mut a: &mut [Self],
            ) -> Result<Vec<Self::Real>> {
                assert_eq!(layout.len(), layout.lda());
                let n = layout.len();
                let jobz = if calc_v { b'V' } else { b'N' };
                let mut eigs = vec![Self::Real::zero(); n as usize];
                let n = n as i32;

                // calc work size
                let mut info = 0;
                let mut work_size = [0.0];
                unsafe {
                    $ev(
                        jobz,
                        uplo as u8,
                        n,
                        &mut a,
                        n,
                        &mut eigs,
                        &mut work_size,
                        -1,
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                // actual ev
                let lwork = work_size[0].to_usize().unwrap();
                let mut work = vec![Self::zero(); lwork];
                unsafe {
                    $ev(
                        jobz,
                        uplo as u8,
                        n,
                        &mut a,
                        n,
                        &mut eigs,
                        &mut work,
                        lwork as i32,
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                Ok(eigs)
            }

            fn eigh_generalized(
                calc_v: bool,
                layout: MatrixLayout,
                uplo: UPLO,
                mut a: &mut [Self],
                mut b: &mut [Self],
            ) -> Result<Vec<Self::Real>> {
                assert_eq!(layout.len(), layout.lda());
                let n = layout.len();
                let jobz = if calc_v { b'V' } else { b'N' };
                let mut eigs = vec![Self::Real::zero(); n as usize];
                let n = n as i32;

                // calc work size
                let mut info = 0;
                let mut work_size = [0.0];
                unsafe {
                    $evg(
                        &[1],
                        jobz,
                        uplo as u8,
                        n,
                        &mut a,
                        n,
                        &mut b,
                        n,
                        &mut eigs,
                        &mut work_size,
                        -1,
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                // actual evg
                let lwork = work_size[0].to_usize().unwrap();
                let mut work = vec![Self::zero(); lwork];
                unsafe {
                    $evg(
                        &[1],
                        jobz,
                        uplo as u8,
                        n,
                        &mut a,
                        n,
                        &mut b,
                        n,
                        &mut eigs,
                        &mut work,
                        lwork as i32,
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                Ok(eigs)
            }
        }
    };
} // impl_eigh!

impl_eigh!(f64, lapack::dsyev, lapack::dsygv);
impl_eigh!(f32, lapack::ssyev, lapack::ssygv);

// splitted for RWORK
macro_rules! impl_eighc {
    ($scalar:ty, $ev:path, $evg:path) => {
        impl Eigh_ for $scalar {
            fn eigh(
                calc_v: bool,
                layout: MatrixLayout,
                uplo: UPLO,
                mut a: &mut [Self],
            ) -> Result<Vec<Self::Real>> {
                assert_eq!(layout.len(), layout.lda());
                let n = layout.len();
                let jobz = if calc_v { b'V' } else { b'N' };
                let mut eigs = vec![Self::Real::zero(); n as usize];
                let mut work = vec![Self::zero(); 2 * n as usize - 1];
                let mut rwork = vec![Self::Real::zero(); 3 * n as usize - 2];
                let mut info = 0;
                let n = n as i32;

                unsafe {
                    $ev(
                        jobz,
                        uplo as u8,
                        n,
                        &mut a,
                        n,
                        &mut eigs,
                        &mut work,
                        2 * n - 1,
                        &mut rwork,
                        &mut info,
                    )
                };
                info.as_lapack_result()?;
                Ok(eigs)
            }

            fn eigh_generalized(
                calc_v: bool,
                layout: MatrixLayout,
                uplo: UPLO,
                mut a: &mut [Self],
                mut b: &mut [Self],
            ) -> Result<Vec<Self::Real>> {
                assert_eq!(layout.len(), layout.lda());
                let n = layout.len();
                let jobz = if calc_v { b'V' } else { b'N' };
                let mut eigs = vec![Self::Real::zero(); n as usize];
                let mut work = vec![Self::zero(); 2 * n as usize - 1];
                let mut rwork = vec![Self::Real::zero(); 3 * n as usize - 2];
                let n = n as i32;
                let mut info = 0;

                unsafe {
                    $evg(
                        &[1],
                        jobz,
                        uplo as u8,
                        n,
                        &mut a,
                        n,
                        &mut b,
                        n,
                        &mut eigs,
                        &mut work,
                        2 * n - 1,
                        &mut rwork,
                        &mut info,
                    )
                };
                info.as_lapack_result()?;
                Ok(eigs)
            }
        }
    };
} // impl_eigh!

impl_eighc!(c64, lapack::zheev, lapack::zhegv);
impl_eighc!(c32, lapack::cheev, lapack::chegv);
