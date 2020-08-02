//! QR decomposition

use crate::{error::*, layout::MatrixLayout, *};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

pub trait QR_: Sized {
    /// Execute Householder reflection as the first step of QR-decomposition
    ///
    /// For C-continuous array,
    /// this will call LQ-decomposition of the transposed matrix $ A^T = LQ^T $
    fn householder(l: MatrixLayout, a: &mut [Self]) -> Result<Vec<Self>>;

    /// Reconstruct Q-matrix from Householder-reflectors
    fn q(l: MatrixLayout, a: &mut [Self], tau: &[Self]) -> Result<()>;

    /// Execute QR-decomposition at once
    fn qr(l: MatrixLayout, a: &mut [Self]) -> Result<Vec<Self>>;
}

macro_rules! impl_qr {
    ($scalar:ty, $qrf:path, $lqf:path, $gqr:path, $glq:path) => {
        impl QR_ for $scalar {
            fn householder(l: MatrixLayout, mut a: &mut [Self]) -> Result<Vec<Self>> {
                let m = l.lda();
                let n = l.len();
                let k = m.min(n);
                let mut tau = unsafe { vec_uninit(k as usize) };

                // eval work size
                let mut info = 0;
                let mut work_size = [Self::zero()];
                unsafe {
                    match l {
                        MatrixLayout::F { .. } => {
                            $qrf(m, n, &mut a, m, &mut tau, &mut work_size, -1, &mut info);
                        }
                        MatrixLayout::C { .. } => {
                            $lqf(m, n, &mut a, m, &mut tau, &mut work_size, -1, &mut info);
                        }
                    }
                }
                info.as_lapack_result()?;

                // calc
                let lwork = work_size[0].to_usize().unwrap();
                let mut work = unsafe { vec_uninit(lwork) };
                unsafe {
                    match l {
                        MatrixLayout::F { .. } => {
                            $qrf(
                                m,
                                n,
                                &mut a,
                                m,
                                &mut tau,
                                &mut work,
                                lwork as i32,
                                &mut info,
                            );
                        }
                        MatrixLayout::C { .. } => {
                            $lqf(
                                m,
                                n,
                                &mut a,
                                m,
                                &mut tau,
                                &mut work,
                                lwork as i32,
                                &mut info,
                            );
                        }
                    }
                }
                info.as_lapack_result()?;

                Ok(tau)
            }

            fn q(l: MatrixLayout, mut a: &mut [Self], tau: &[Self]) -> Result<()> {
                let m = l.lda();
                let n = l.len();
                let k = m.min(n);
                assert_eq!(tau.len(), k as usize);

                // eval work size
                let mut info = 0;
                let mut work_size = [Self::zero()];
                unsafe {
                    match l {
                        MatrixLayout::F { .. } => {
                            $gqr(m, k, k, &mut a, m, &tau, &mut work_size, -1, &mut info)
                        }
                        MatrixLayout::C { .. } => {
                            $glq(k, n, k, &mut a, m, &tau, &mut work_size, -1, &mut info)
                        }
                    }
                };

                // calc
                let lwork = work_size[0].to_usize().unwrap();
                let mut work = unsafe { vec_uninit(lwork) };
                unsafe {
                    match l {
                        MatrixLayout::F { .. } => {
                            $gqr(m, k, k, &mut a, m, &tau, &mut work, lwork as i32, &mut info)
                        }
                        MatrixLayout::C { .. } => {
                            $glq(k, n, k, &mut a, m, &tau, &mut work, lwork as i32, &mut info)
                        }
                    }
                }
                info.as_lapack_result()?;
                Ok(())
            }

            fn qr(l: MatrixLayout, a: &mut [Self]) -> Result<Vec<Self>> {
                let tau = Self::householder(l, a)?;
                let r = Vec::from(&*a);
                Self::q(l, a, &tau)?;
                Ok(r)
            }
        }
    };
} // endmacro

impl_qr!(
    f64,
    lapack::dgeqrf,
    lapack::dgelqf,
    lapack::dorgqr,
    lapack::dorglq
);
impl_qr!(
    f32,
    lapack::sgeqrf,
    lapack::sgelqf,
    lapack::sorgqr,
    lapack::sorglq
);
impl_qr!(
    c64,
    lapack::zgeqrf,
    lapack::zgelqf,
    lapack::zungqr,
    lapack::zunglq
);
impl_qr!(
    c32,
    lapack::cgeqrf,
    lapack::cgelqf,
    lapack::cungqr,
    lapack::cunglq
);
