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

    /// Wraps `*sygv` for real and `*hegv` for complex
    fn eigh_generalized(
        calc_eigenvec: bool,
        layout: MatrixLayout,
        uplo: UPLO,
        a: &mut [Self],
        b: &mut [Self],
    ) -> Result<Vec<Self::Real>>;
}

macro_rules! impl_eigh {
    (@real, $scalar:ty, $ev:path, $evg:path) => {
        impl_eigh!(@body, $scalar, $ev, $evg, );
    };
    (@complex, $scalar:ty, $ev:path, $evg:path) => {
        impl_eigh!(@body, $scalar, $ev, $evg, rwork);
    };
    (@body, $scalar:ty, $ev:path, $evg:path, $($rwork_ident:ident),*) => {
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
                let mut eigs = unsafe { vec_uninit(n as usize) };

                $(
                let mut $rwork_ident = unsafe { vec_uninit(3 * n as usize - 2 as usize) };
                )*

                // calc work size
                let mut info = 0;
                let mut work_size = [Self::zero()];
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
                        $(&mut $rwork_ident,)*
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                // actual ev
                let lwork = work_size[0].to_usize().unwrap();
                let mut work = unsafe { vec_uninit(lwork) };
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
                        $(&mut $rwork_ident,)*
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
                let mut eigs = unsafe { vec_uninit(n as usize) };

                $(
                let mut $rwork_ident = unsafe { vec_uninit(3 * n as usize - 2) };
                )*

                // calc work size
                let mut info = 0;
                let mut work_size = [Self::zero()];
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
                        $(&mut $rwork_ident,)*
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                // actual evg
                let lwork = work_size[0].to_usize().unwrap();
                let mut work = unsafe { vec_uninit(lwork) };
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
                        $(&mut $rwork_ident,)*
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                Ok(eigs)
            }
        }
    };
} // impl_eigh!

impl_eigh!(@real, f64, lapack::dsyev, lapack::dsygv);
impl_eigh!(@real, f32, lapack::ssyev, lapack::ssygv);
impl_eigh!(@complex, c64, lapack::zheev, lapack::zhegv);
impl_eigh!(@complex, c32, lapack::cheev, lapack::chegv);
