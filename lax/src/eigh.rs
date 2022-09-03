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
                a: &mut [Self],
            ) -> Result<Vec<Self::Real>> {
                assert_eq!(layout.len(), layout.lda());
                let n = layout.len();
                let jobz = if calc_v { JobEv::All } else { JobEv::None };
                let mut eigs: Vec<MaybeUninit<Self::Real>> = unsafe { vec_uninit(n as usize) };

                $(
                let mut $rwork_ident: Vec<MaybeUninit<Self::Real>> = unsafe { vec_uninit(3 * n as usize - 2 as usize) };
                )*

                // calc work size
                let mut info = 0;
                let mut work_size = [Self::zero()];
                unsafe {
                    $ev(
                        jobz.as_ptr() ,
                        uplo.as_ptr(),
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &n,
                        AsPtr::as_mut_ptr(&mut eigs),
                        AsPtr::as_mut_ptr(&mut work_size),
                        &(-1),
                        $(AsPtr::as_mut_ptr(&mut $rwork_ident),)*
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                // actual ev
                let lwork = work_size[0].to_usize().unwrap();
                let mut work: Vec<MaybeUninit<Self>> = unsafe { vec_uninit(lwork) };
                let lwork = lwork as i32;
                unsafe {
                    $ev(
                        jobz.as_ptr(),
                        uplo.as_ptr(),
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &n,
                        AsPtr::as_mut_ptr(&mut eigs),
                        AsPtr::as_mut_ptr(&mut work),
                        &lwork,
                        $(AsPtr::as_mut_ptr(&mut $rwork_ident),)*
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                let eigs = unsafe { eigs.assume_init() };
                Ok(eigs)
            }

            fn eigh_generalized(
                calc_v: bool,
                layout: MatrixLayout,
                uplo: UPLO,
                a: &mut [Self],
                b: &mut [Self],
            ) -> Result<Vec<Self::Real>> {
                assert_eq!(layout.len(), layout.lda());
                let n = layout.len();
                let jobz = if calc_v { JobEv::All } else { JobEv::None };
                let mut eigs: Vec<MaybeUninit<Self::Real>> = unsafe { vec_uninit(n as usize) };

                $(
                let mut $rwork_ident: Vec<MaybeUninit<Self::Real>> = unsafe { vec_uninit(3 * n as usize - 2) };
                )*

                // calc work size
                let mut info = 0;
                let mut work_size = [Self::zero()];
                unsafe {
                    $evg(
                        &1, // ITYPE A*x = (lambda)*B*x
                        jobz.as_ptr(),
                        uplo.as_ptr(),
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &n,
                        AsPtr::as_mut_ptr(b),
                        &n,
                        AsPtr::as_mut_ptr(&mut eigs),
                        AsPtr::as_mut_ptr(&mut work_size),
                        &(-1),
                        $(AsPtr::as_mut_ptr(&mut $rwork_ident),)*
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                // actual evg
                let lwork = work_size[0].to_usize().unwrap();
                let mut work: Vec<MaybeUninit<Self>> = unsafe { vec_uninit(lwork) };
                let lwork = lwork as i32;
                unsafe {
                    $evg(
                        &1, // ITYPE A*x = (lambda)*B*x
                        jobz.as_ptr(),
                        uplo.as_ptr(),
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &n,
                        AsPtr::as_mut_ptr(b),
                        &n,
                        AsPtr::as_mut_ptr(&mut eigs),
                        AsPtr::as_mut_ptr(&mut work),
                        &lwork,
                        $(AsPtr::as_mut_ptr(&mut $rwork_ident),)*
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                let eigs = unsafe { eigs.assume_init() };
                Ok(eigs)
            }
        }
    };
} // impl_eigh!

impl_eigh!(@real, f64, lapack_sys::dsyev_, lapack_sys::dsygv_);
impl_eigh!(@real, f32, lapack_sys::ssyev_, lapack_sys::ssygv_);
impl_eigh!(@complex, c64, lapack_sys::zheev_, lapack_sys::zhegv_);
impl_eigh!(@complex, c32, lapack_sys::cheev_, lapack_sys::chegv_);
