//! Eigenvalue decomposition for Symmetric/Hermite matrices

use super::*;
use crate::{error::*, layout::MatrixLayout};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

pub trait Eigh: Sized {
    type Elem: Scalar;

    /// Allocate working memory for eigenvalue problem
    fn eigh_work(calc_eigenvec: bool, layout: MatrixLayout, uplo: UPLO) -> Result<Self>;

    /// Solve eigenvalue problem
    fn eigh_calc(&mut self, a: &mut [Self::Elem]) -> Result<&[<Self::Elem as Scalar>::Real]>;
}

/// Working memory for symmetric/Hermitian eigenvalue problem. See [Eigh trait](trait.Eigh.html)
pub struct EighWork<T: Scalar> {
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
        impl Eigh for EighWork<$scalar> {
            type Elem = $scalar;

            fn eigh_work(calc_v: bool, layout: MatrixLayout, uplo: UPLO) -> Result<Self> {
                assert_eq!(layout.len(), layout.lda());
                let n = layout.len();
                let jobz = if calc_v { b'V' } else { b'N' };
                let mut eigs = unsafe { vec_uninit(n as usize) };

                let mut info = 0;
                let mut work_size = [Self::Elem::zero()];
                unsafe {
                    $ev(
                        jobz,
                        uplo as u8,
                        n,
                        &mut [], // matrix A is not referenced in query mode
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
                Ok(EighWork {
                    jobz,
                    uplo,
                    n,
                    eigs,
                    work,
                    rwork: None,
                })
            }

            fn eigh_calc(
                &mut self,
                a: &mut [Self::Elem],
            ) -> Result<&[<Self::Elem as Scalar>::Real]> {
                assert_eq!(a.len(), (self.n * self.n) as usize);
                let mut info = 0;
                let lwork = self.work.len() as i32;
                unsafe {
                    $ev(
                        self.jobz,
                        self.uplo as u8,
                        self.n,
                        a,
                        self.n,
                        &mut self.eigs,
                        &mut self.work,
                        lwork,
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                Ok(&self.eigs)
            }
        }
    };
}

impl_eigh_work_real!(f32, lapack::ssyev);
impl_eigh_work_real!(f64, lapack::dsyev);

macro_rules! impl_eigh_work_complex {
    ($scalar:ty, $ev:path) => {
        impl Eigh for EighWork<$scalar> {
            type Elem = $scalar;

            fn eigh_work(calc_v: bool, layout: MatrixLayout, uplo: UPLO) -> Result<Self> {
                assert_eq!(layout.len(), layout.lda());
                let n = layout.len();
                let jobz = if calc_v { b'V' } else { b'N' };
                let mut eigs = unsafe { vec_uninit(n as usize) };

                let mut info = 0;
                let mut work_size = [Self::Elem::zero()];
                let mut rwork = unsafe { vec_uninit(3 * n as usize - 2) };
                unsafe {
                    $ev(
                        jobz,
                        uplo as u8,
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
                Ok(EighWork {
                    jobz,
                    uplo,
                    n,
                    eigs,
                    work,
                    rwork: Some(rwork),
                })
            }

            fn eigh_calc(
                &mut self,
                a: &mut [Self::Elem],
            ) -> Result<&[<Self::Elem as Scalar>::Real]> {
                assert_eq!(a.len(), (self.n * self.n) as usize);
                let mut info = 0;
                let lwork = self.work.len() as i32;
                unsafe {
                    $ev(
                        self.jobz,
                        self.uplo as u8,
                        self.n,
                        a,
                        self.n,
                        &mut self.eigs,
                        &mut self.work,
                        lwork,
                        self.rwork.as_mut().unwrap(),
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                Ok(&self.eigs)
            }
        }
    };
}

impl_eigh_work_complex!(c32, lapack::cheev);
impl_eigh_work_complex!(c64, lapack::zheev);
