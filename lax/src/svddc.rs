use crate::{error::*, layout::MatrixLayout, *};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

#[cfg_attr(doc, katexit::katexit)]
/// Singular value decomposition with divide-and-conquer method
pub trait SVDDC_: Scalar {
    /// Compute singular value decomposition $A = U \Sigma V^T$
    ///
    /// LAPACK correspondance
    /// ----------------------
    ///
    /// | f32    | f64    | c32    | c64    |
    /// |:-------|:-------|:-------|:-------|
    /// | sgesdd | dgesdd | cgesdd | zgesdd |
    ///
    fn svddc(l: MatrixLayout, jobz: JobSvd, a: &mut [Self]) -> Result<SvdOwned<Self>>;
}

macro_rules! impl_svddc {
    (@real, $scalar:ty, $gesdd:path) => {
        impl_svddc!(@body, $scalar, $gesdd, );
    };
    (@complex, $scalar:ty, $gesdd:path) => {
        impl_svddc!(@body, $scalar, $gesdd, rwork);
    };
    (@body, $scalar:ty, $gesdd:path, $($rwork_ident:ident),*) => {
        impl SVDDC_ for $scalar {
            fn svddc(l: MatrixLayout, jobz: JobSvd, a: &mut [Self],) -> Result<SvdOwned<Self>> {
                let m = l.lda();
                let n = l.len();
                let k = m.min(n);
                let mut s = vec_uninit(k as usize);

                let (u_col, vt_row) = match jobz {
                    JobSvd::All | JobSvd::None => (m, n),
                    JobSvd::Some => (k, k),
                };
                let (mut u, mut vt) = match jobz {
                    JobSvd::All => (
                        Some(vec_uninit((m * m) as usize)),
                        Some(vec_uninit((n * n) as usize)),
                    ),
                    JobSvd::Some => (
                        Some(vec_uninit((m * u_col) as usize)),
                        Some(vec_uninit((n * vt_row) as usize)),
                    ),
                    JobSvd::None => (None, None),
                };

                $( // for complex only
                let mx = n.max(m) as usize;
                let mn = n.min(m) as usize;
                let lrwork = match jobz {
                    JobSvd::None => 7 * mn,
                    _ => std::cmp::max(5*mn*mn + 5*mn, 2*mx*mn + 2*mn*mn + mn),
                };
                let mut $rwork_ident: Vec<MaybeUninit<Self::Real>> = vec_uninit(lrwork);
                )*

                // eval work size
                let mut info = 0;
                let mut iwork: Vec<MaybeUninit<i32>> = vec_uninit(8 * k as usize);
                let mut work_size = [Self::zero()];
                unsafe {
                    $gesdd(
                        jobz.as_ptr(),
                        &m,
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &m,
                        AsPtr::as_mut_ptr(&mut s),
                        AsPtr::as_mut_ptr(u.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &m,
                        AsPtr::as_mut_ptr(vt.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &vt_row,
                        AsPtr::as_mut_ptr(&mut work_size),
                        &(-1),
                        $(AsPtr::as_mut_ptr(&mut $rwork_ident),)*
                        AsPtr::as_mut_ptr(&mut iwork),
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                // do svd
                let lwork = work_size[0].to_usize().unwrap();
                let mut work: Vec<MaybeUninit<Self>> = vec_uninit(lwork);
                unsafe {
                    $gesdd(
                        jobz.as_ptr(),
                        &m,
                        &n,
                        AsPtr::as_mut_ptr(a),
                        &m,
                        AsPtr::as_mut_ptr(&mut s),
                        AsPtr::as_mut_ptr(u.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &m,
                        AsPtr::as_mut_ptr(vt.as_mut().map(|x| x.as_mut_slice()).unwrap_or(&mut [])),
                        &vt_row,
                        AsPtr::as_mut_ptr(&mut work),
                        &(lwork as i32),
                        $(AsPtr::as_mut_ptr(&mut $rwork_ident),)*
                        AsPtr::as_mut_ptr(&mut iwork),
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                let s = unsafe { s.assume_init() };
                let u = u.map(|v| unsafe { v.assume_init() });
                let vt = vt.map(|v| unsafe { v.assume_init() });

                match l {
                    MatrixLayout::F { .. } => Ok(SvdOwned { s, u, vt }),
                    MatrixLayout::C { .. } => Ok(SvdOwned { s, u: vt, vt: u }),
                }
            }
        }
    };
}

impl_svddc!(@real, f32, lapack_sys::sgesdd_);
impl_svddc!(@real, f64, lapack_sys::dgesdd_);
impl_svddc!(@complex, c32, lapack_sys::cgesdd_);
impl_svddc!(@complex, c64, lapack_sys::zgesdd_);
