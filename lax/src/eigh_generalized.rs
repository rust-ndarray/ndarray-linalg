use super::*;
use crate::{error::*, layout::MatrixLayout};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

/// Types of generalized eigenvalue problem
#[allow(dead_code)] // FIXME create interface to use ABxlx and BAxlx
#[repr(i32)]
pub enum ITYPE {
    /// Solve $ A x = \lambda B x $
    AxlBx = 1,
    /// Solve $ A B x = \lambda x $
    ABxlx = 2,
    /// Solve $ B A x = \lambda x $
    BAxlx = 3,
}

/// Generalized eigenvalue problem for Symmetric/Hermite matrices
pub trait EighGeneralized: Sized {
    type Elem: Scalar;

    /// Allocate working memory
    fn eigh_generalized_work(calc_eigenvec: bool, layout: MatrixLayout, uplo: UPLO)
        -> Result<Self>;

    /// Solve generalized eigenvalue problem
    fn eigh_generalized_calc(
        &mut self,
        a: &mut [Self::Elem],
        b: &mut [Self::Elem],
    ) -> Result<&[<Self::Elem as Scalar>::Real]>;
}

/// Working memory for symmetric/Hermitian generalized eigenvalue problem.
/// See [EighGeneralized trait](trait.EighGeneralized.html)
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
        impl EighGeneralized for EighGeneralizedWork<$scalar> {
            type Elem = $scalar;

            fn eigh_generalized_work(
                calc_v: bool,
                layout: MatrixLayout,
                uplo: UPLO,
            ) -> Result<Self> {
                assert_eq!(layout.len(), layout.lda());
                let n = layout.len();
                let jobz = if calc_v { b'V' } else { b'N' };
                let mut eigs = unsafe { vec_uninit(n as usize) };

                let mut info = 0;
                let mut work_size = [Self::Elem::zero()];
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

            fn eigh_generalized_calc(
                &mut self,
                a: &mut [Self::Elem],
                b: &mut [Self::Elem],
            ) -> Result<&[<Self::Elem as Scalar>::Real]> {
                assert_eq!(a.len(), (self.n * self.n) as usize);
                let mut info = 0;
                let lwork = self.work.len() as i32;
                unsafe {
                    $ev(
                        &[ITYPE::AxlBx as i32],
                        self.jobz,
                        self.uplo as u8,
                        self.n,
                        a,
                        self.n,
                        b,
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

impl_eigh_work_real!(f32, lapack::ssygv);
impl_eigh_work_real!(f64, lapack::dsygv);

macro_rules! impl_eigh_work_complex {
    ($scalar:ty, $ev:path) => {
        impl EighGeneralized for EighGeneralizedWork<$scalar> {
            type Elem = $scalar;

            fn eigh_generalized_work(
                calc_v: bool,
                layout: MatrixLayout,
                uplo: UPLO,
            ) -> Result<Self> {
                assert_eq!(layout.len(), layout.lda());
                let n = layout.len();
                let jobz = if calc_v { b'V' } else { b'N' };

                // Different from work array, eigs must be touched from Rust
                let mut eigs = unsafe { vec_uninit(n as usize) };

                let mut info = 0;
                let mut work_size = [Self::Elem::zero()];
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

            fn eigh_generalized_calc(
                &mut self,
                a: &mut [Self::Elem],
                b: &mut [Self::Elem],
            ) -> Result<&[<Self::Elem as Scalar>::Real]> {
                assert_eq!(a.len(), (self.n * self.n) as usize);
                let mut info = 0;
                let lwork = self.work.len() as i32;
                unsafe {
                    $ev(
                        &[ITYPE::AxlBx as i32],
                        self.jobz,
                        self.uplo as u8,
                        self.n,
                        a,
                        self.n,
                        b,
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

impl_eigh_work_complex!(c32, lapack::chegv);
impl_eigh_work_complex!(c64, lapack::zhegv);
