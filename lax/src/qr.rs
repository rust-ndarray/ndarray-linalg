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
            fn householder(l: MatrixLayout, a: &mut [Self]) -> Result<Vec<Self>> {
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
                            $qrf(
                                &m,
                                &n,
                                AsPtr::as_mut_ptr(a),
                                &m,
                                AsPtr::as_mut_ptr(&mut tau),
                                AsPtr::as_mut_ptr(&mut work_size),
                                &(-1),
                                &mut info,
                            );
                        }
                        MatrixLayout::C { .. } => {
                            $lqf(
                                &m,
                                &n,
                                AsPtr::as_mut_ptr(a),
                                &m,
                                AsPtr::as_mut_ptr(&mut tau),
                                AsPtr::as_mut_ptr(&mut work_size),
                                &(-1),
                                &mut info,
                            );
                        }
                    }
                }
                info.as_lapack_result()?;

                // calc
                let lwork = work_size[0].to_usize().unwrap();
                let mut work: Vec<MaybeUninit<Self>> = unsafe { vec_uninit(lwork) };
                unsafe {
                    match l {
                        MatrixLayout::F { .. } => {
                            $qrf(
                                &m,
                                &n,
                                AsPtr::as_mut_ptr(a),
                                &m,
                                AsPtr::as_mut_ptr(&mut tau),
                                AsPtr::as_mut_ptr(&mut work),
                                &(lwork as i32),
                                &mut info,
                            );
                        }
                        MatrixLayout::C { .. } => {
                            $lqf(
                                &m,
                                &n,
                                AsPtr::as_mut_ptr(a),
                                &m,
                                AsPtr::as_mut_ptr(&mut tau),
                                AsPtr::as_mut_ptr(&mut work),
                                &(lwork as i32),
                                &mut info,
                            );
                        }
                    }
                }
                info.as_lapack_result()?;

                let tau = unsafe { tau.assume_init() };

                Ok(tau)
            }

            fn q(l: MatrixLayout, a: &mut [Self], tau: &[Self]) -> Result<()> {
                let m = l.lda();
                let n = l.len();
                let k = m.min(n);
                assert_eq!(tau.len(), k as usize);

                // eval work size
                let mut info = 0;
                let mut work_size = [Self::zero()];
                unsafe {
                    match l {
                        MatrixLayout::F { .. } => $gqr(
                            &m,
                            &k,
                            &k,
                            AsPtr::as_mut_ptr(a),
                            &m,
                            AsPtr::as_ptr(&tau),
                            AsPtr::as_mut_ptr(&mut work_size),
                            &(-1),
                            &mut info,
                        ),
                        MatrixLayout::C { .. } => $glq(
                            &k,
                            &n,
                            &k,
                            AsPtr::as_mut_ptr(a),
                            &m,
                            AsPtr::as_ptr(&tau),
                            AsPtr::as_mut_ptr(&mut work_size),
                            &(-1),
                            &mut info,
                        ),
                    }
                };

                // calc
                let lwork = work_size[0].to_usize().unwrap();
                let mut work: Vec<MaybeUninit<Self>> = unsafe { vec_uninit(lwork) };
                unsafe {
                    match l {
                        MatrixLayout::F { .. } => $gqr(
                            &m,
                            &k,
                            &k,
                            AsPtr::as_mut_ptr(a),
                            &m,
                            AsPtr::as_ptr(&tau),
                            AsPtr::as_mut_ptr(&mut work),
                            &(lwork as i32),
                            &mut info,
                        ),
                        MatrixLayout::C { .. } => $glq(
                            &k,
                            &n,
                            &k,
                            AsPtr::as_mut_ptr(a),
                            &m,
                            AsPtr::as_ptr(&tau),
                            AsPtr::as_mut_ptr(&mut work),
                            &(lwork as i32),
                            &mut info,
                        ),
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
    lapack_sys::dgeqrf_,
    lapack_sys::dgelqf_,
    lapack_sys::dorgqr_,
    lapack_sys::dorglq_
);
impl_qr!(
    f32,
    lapack_sys::sgeqrf_,
    lapack_sys::sgelqf_,
    lapack_sys::sorgqr_,
    lapack_sys::sorglq_
);
impl_qr!(
    c64,
    lapack_sys::zgeqrf_,
    lapack_sys::zgelqf_,
    lapack_sys::zungqr_,
    lapack_sys::zunglq_
);
impl_qr!(
    c32,
    lapack_sys::cgeqrf_,
    lapack_sys::cgelqf_,
    lapack_sys::cungqr_,
    lapack_sys::cunglq_
);
