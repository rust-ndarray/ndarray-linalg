//! Singular-value decomposition

use crate::{error::*, layout::MatrixLayout, *};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

#[repr(u8)]
#[derive(Debug, Copy, Clone)]
enum FlagSVD {
    All = b'A',
    // OverWrite = b'O',
    // Separately = b'S',
    No = b'N',
}

impl FlagSVD {
    fn from_bool(calc_uv: bool) -> Self {
        if calc_uv {
            FlagSVD::All
        } else {
            FlagSVD::No
        }
    }

    fn as_ptr(&self) -> *const i8 {
        self as *const FlagSVD as *const i8
    }
}

/// Result of SVD
pub struct SVDOutput<A: Scalar> {
    /// diagonal values
    pub s: Vec<A::Real>,
    /// Unitary matrix for destination space
    pub u: Option<Vec<A>>,
    /// Unitary matrix for departure space
    pub vt: Option<Vec<A>>,
}

/// Wraps `*gesvd`
pub trait SVD_: Scalar {
    /// Calculate singular value decomposition $ A = U \Sigma V^T $
    fn svd(l: MatrixLayout, calc_u: bool, calc_vt: bool, a: &mut [Self])
        -> Result<SVDOutput<Self>>;
}

macro_rules! impl_svd {
    (@real, $scalar:ty, $gesvd:path) => {
        impl_svd!(@body, $scalar, $gesvd, );
    };
    (@complex, $scalar:ty, $gesvd:path) => {
        impl_svd!(@body, $scalar, $gesvd, rwork);
    };
    (@body, $scalar:ty, $gesvd:path, $($rwork_ident:ident),*) => {
        impl SVD_ for $scalar {
            fn svd(l: MatrixLayout, calc_u: bool, calc_vt: bool, a: &mut [Self],) -> Result<SVDOutput<Self>> {
                let ju = match l {
                    MatrixLayout::F { .. } => FlagSVD::from_bool(calc_u),
                    MatrixLayout::C { .. } => FlagSVD::from_bool(calc_vt),
                };
                let jvt = match l {
                    MatrixLayout::F { .. } => FlagSVD::from_bool(calc_vt),
                    MatrixLayout::C { .. } => FlagSVD::from_bool(calc_u),
                };

                let m = l.lda();
                let mut u = match ju {
                    FlagSVD::All => Some(unsafe { vec_uninit( (m * m) as usize) }),
                    FlagSVD::No => None,
                };

                let n = l.len();
                let mut vt = match jvt {
                    FlagSVD::All => Some(unsafe { vec_uninit( (n * n) as usize) }),
                    FlagSVD::No => None,
                };

                let k = std::cmp::min(m, n);
                let mut s = unsafe { vec_uninit( k as usize) };

                $(
                let mut $rwork_ident: Vec<MaybeUninit<Self::Real>> = unsafe { vec_uninit( 5 * k as usize) };
                )*

                // eval work size
                let mut info = 0;
                let mut work_size = [Self::zero()];
                unsafe {
                    $gesvd(
                        ju.as_ptr(),
                        jvt.as_ptr(),
                        &m,
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &m,
                        AsPtr::as_mut_ptr(&mut s),
                        AsPtr::as_mut_ptr(u.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &m,
                        AsPtr::as_mut_ptr(vt.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &n,
                        AsPtr::as_mut_ptr(&mut work_size),
                        &(-1),
                        $(AsPtr::as_mut_ptr(&mut $rwork_ident),)*
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                // calc
                let lwork = work_size[0].to_usize().unwrap();
                let mut work: Vec<MaybeUninit<Self>> = unsafe { vec_uninit( lwork) };
                unsafe {
                    $gesvd(
                        ju.as_ptr(),
                        jvt.as_ptr() ,
                        &m,
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &m,
                        AsPtr::as_mut_ptr(&mut s),
                        AsPtr::as_mut_ptr(u.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &m,
                        AsPtr::as_mut_ptr(vt.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &n,
                        AsPtr::as_mut_ptr(&mut work),
                        &(lwork as i32),
                        $(AsPtr::as_mut_ptr(&mut $rwork_ident),)*
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                let s = unsafe { s.assume_init() };
                let u = u.map(|v| unsafe { v.assume_init() });
                let vt = vt.map(|v| unsafe { v.assume_init() });

                match l {
                    MatrixLayout::F { .. } => Ok(SVDOutput { s, u, vt }),
                    MatrixLayout::C { .. } => Ok(SVDOutput { s, u: vt, vt: u }),
                }
            }
        }
    };
} // impl_svd!

impl_svd!(@real, f64, lapack_sys::dgesvd_);
impl_svd!(@real, f32, lapack_sys::sgesvd_);
impl_svd!(@complex, c64, lapack_sys::zgesvd_);
impl_svd!(@complex, c32, lapack_sys::cgesvd_);
