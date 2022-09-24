use super::*;
use crate::{error::*, layout::MatrixLayout};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

#[cfg_attr(doc, katexit::katexit)]
/// Eigenvalue problem for symmetric/hermite matrix
pub trait EighGeneralized_: Scalar {
    /// Compute generalized right eigenvalue and eigenvectors $Ax = \lambda B x$
    ///
    /// LAPACK correspondance
    /// ----------------------
    ///
    /// | f32   | f64   | c32   | c64   |
    /// |:------|:------|:------|:------|
    /// | ssygv | dsygv | chegv | zhegv |
    ///
    fn eigh_generalized(
        calc_eigenvec: bool,
        layout: MatrixLayout,
        uplo: UPLO,
        a: &mut [Self],
        b: &mut [Self],
    ) -> Result<Vec<Self::Real>>;
}

macro_rules! impl_eigh {
    (@real, $scalar:ty, $evg:path) => {
        impl_eigh!(@body, $scalar, $evg, );
    };
    (@complex, $scalar:ty, $evg:path) => {
        impl_eigh!(@body, $scalar, $evg, rwork);
    };
    (@body, $scalar:ty, $evg:path, $($rwork_ident:ident),*) => {
        impl EighGeneralized_ for $scalar {
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
                let mut eigs: Vec<MaybeUninit<Self::Real>> = vec_uninit(n as usize);

                $(
                let mut $rwork_ident: Vec<MaybeUninit<Self::Real>> = vec_uninit(3 * n as usize - 2);
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
                let mut work: Vec<MaybeUninit<Self>> = vec_uninit(lwork);
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

impl_eigh!(@real, f64, lapack_sys::dsygv_);
impl_eigh!(@real, f32, lapack_sys::ssygv_);
impl_eigh!(@complex, c64, lapack_sys::zhegv_);
impl_eigh!(@complex, c32, lapack_sys::chegv_);
