use super::*;
use crate::{error::*, layout::MatrixLayout};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

/// Generalized eigenvalue problem for Symmetric/Hermite matrices
pub(crate) trait EighGeneralized: Scalar {
    /// Allocate working memory
    fn eigh_generalized_work(
        calc_eigenvec: bool,
        layout: MatrixLayout,
        uplo: UPLO,
    ) -> Result<EighGeneralizedWork<Self>>;

    /// Solve generalized eigenvalue problem
    fn eigh_generalized_calc<'work>(
        work: &'work mut EighGeneralizedWork<Self>,
        a: &mut [Self],
        b: &mut [Self],
    ) -> Result<&'work [Self::Real]>;
}

/// Working memory for symmetric/Hermitian generalized eigenvalue problem.
/// See [LapackStrict trait](trait.LapackStrict.html)
pub struct EighGeneralizedWork<T: Scalar> {
    jobz: u8,
    uplo: UPLO,
    n: i32,
    eigs: Vec<T::Real>,
    // This array is NOT initialized. Do not touch from Rust.
    work: Vec<T>,
    // Needs only for complex case
    rwork: Option<Vec<T::Real>>,
}

macro_rules! impl_eigh_work_real {
    ($scalar:ty, $ev:path) => {
        impl EighGeneralized for $scalar {
            fn eigh_generalized_work(
                calc_v: bool,
                layout: MatrixLayout,
                uplo: UPLO,
            ) -> Result<EighGeneralizedWork<Self>> {
                assert_eq!(layout.len(), layout.lda());
                let n = layout.len();
                let jobz = if calc_v { b'V' } else { b'N' };
                let mut eigs = unsafe { vec_uninit(n as usize) };

                let mut info = 0;
                let mut work_size = [Self::zero()];
                unsafe {
                    $ev(
                        &[ITYPE::AxlBx as i32],
                        jobz,
                        uplo as u8,
                        n,
                        &mut [], // matrix A is not referenced in query mode
                        n,
                        &mut [], // matrix B is not referenced in query mode
                        n,
                        &mut eigs,
                        &mut work_size,
                        -1,
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                let lwork = work_size[0].to_usize().unwrap();
                let work = unsafe { vec_uninit(lwork) };
                Ok(EighGeneralizedWork {
                    jobz,
                    uplo,
                    n,
                    eigs,
                    work,
                    rwork: None,
                })
            }

            fn eigh_generalized_calc<'work>(
                work: &'work mut EighGeneralizedWork<Self>,
                a: &mut [Self],
                b: &mut [Self],
            ) -> Result<&'work [Self::Real]> {
                assert_eq!(a.len(), (work.n * work.n) as usize);
                let mut info = 0;
                let lwork = work.work.len() as i32;
                unsafe {
                    $ev(
                        &[ITYPE::AxlBx as i32],
                        work.jobz,
                        work.uplo as u8,
                        work.n,
                        a,
                        work.n,
                        b,
                        work.n,
                        &mut work.eigs,
                        &mut work.work,
                        lwork,
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                Ok(&work.eigs)
            }
        }
    };
}

impl_eigh_work_real!(f32, lapack::ssygv);
impl_eigh_work_real!(f64, lapack::dsygv);

macro_rules! impl_eigh_work_complex {
    ($scalar:ty, $ev:path) => {
        impl EighGeneralized for $scalar {
            fn eigh_generalized_work(
                calc_v: bool,
                layout: MatrixLayout,
                uplo: UPLO,
            ) -> Result<EighGeneralizedWork<Self>> {
                assert_eq!(layout.len(), layout.lda());
                let n = layout.len();
                let jobz = if calc_v { b'V' } else { b'N' };

                // Different from work array, eigs must be touched from Rust
                let mut eigs = unsafe { vec_uninit(n as usize) };

                let mut info = 0;
                let mut work_size = [Self::zero()];
                let mut rwork = unsafe { vec_uninit(3 * n as usize - 2) };
                unsafe {
                    $ev(
                        &[ITYPE::AxlBx as i32],
                        jobz,
                        uplo as u8,
                        n,
                        &mut [],
                        n,
                        &mut [],
                        n,
                        &mut eigs,
                        &mut work_size,
                        -1,
                        &mut rwork,
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                let lwork = work_size[0].to_usize().unwrap();
                let work = unsafe { vec_uninit(lwork) };
                Ok(EighGeneralizedWork {
                    jobz,
                    uplo,
                    n,
                    eigs,
                    work,
                    rwork: Some(rwork),
                })
            }

            fn eigh_generalized_calc<'work>(
                work: &'work mut EighGeneralizedWork<Self>,
                a: &mut [Self],
                b: &mut [Self],
            ) -> Result<&'work [Self::Real]> {
                assert_eq!(a.len(), (work.n * work.n) as usize);
                let mut info = 0;
                let lwork = work.work.len() as i32;
                unsafe {
                    $ev(
                        &[ITYPE::AxlBx as i32],
                        work.jobz,
                        work.uplo as u8,
                        work.n,
                        a,
                        work.n,
                        b,
                        work.n,
                        &mut work.eigs,
                        &mut work.work,
                        lwork,
                        work.rwork.as_mut().unwrap(),
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                Ok(&work.eigs)
            }
        }
    };
}

impl_eigh_work_complex!(c32, lapack::chegv);
impl_eigh_work_complex!(c64, lapack::zhegv);
