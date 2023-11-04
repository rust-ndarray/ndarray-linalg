//! Solve linear equations using LU-decomposition

use crate::{error::*, layout::MatrixLayout, *};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

/// Helper trait to abstract `*getrf` LAPACK routines for implementing [Lapack::lu]
///
/// LAPACK correspondance
/// ----------------------
///
/// | f32    | f64    | c32    | c64    |
/// |:-------|:-------|:-------|:-------|
/// | sgetrf | dgetrf | cgetrf | zgetrf |
///
pub trait LuImpl: Scalar {
    fn lu(l: MatrixLayout, a: &mut [Self]) -> Result<Pivot>;
}

macro_rules! impl_lu {
    ($scalar:ty, $getrf:path) => {
        impl LuImpl for $scalar {
            fn lu(l: MatrixLayout, a: &mut [Self]) -> Result<Pivot> {
                let (row, col) = l.size();
                assert_eq!(a.len() as i32, row * col);
                if row == 0 || col == 0 {
                    // Do nothing for empty matrix
                    return Ok(Vec::new());
                }
                let k = ::std::cmp::min(row, col);
                let mut ipiv = vec_uninit(k as usize);
                let mut info = 0;
                unsafe {
                    $getrf(
                        &l.lda(),
                        &l.len(),
                        AsPtr::as_mut_ptr(a),
                        &l.lda(),
                        AsPtr::as_mut_ptr(&mut ipiv),
                        &mut info,
                    )
                };
                info.as_lapack_result()?;
                let ipiv = unsafe { ipiv.assume_init() };
                Ok(ipiv)
            }
        }
    };
}

impl_lu!(c64, lapack_sys::zgetrf_);
impl_lu!(c32, lapack_sys::cgetrf_);
impl_lu!(f64, lapack_sys::dgetrf_);
impl_lu!(f32, lapack_sys::sgetrf_);

#[cfg_attr(doc, katexit::katexit)]
/// Helper trait to abstract `*getrs` LAPACK routines for implementing [Lapack::solve]
///
/// If the array has C layout, then it needs to be handled
/// specially, since LAPACK expects a Fortran-layout array.
/// Reinterpreting a C layout array as Fortran layout is
/// equivalent to transposing it. So, we can handle the "no
/// transpose" and "transpose" cases by swapping to "transpose"
/// or "no transpose", respectively. For the "Hermite" case, we
/// can take advantage of the following:
///
/// $$
/// \begin{align*}
///   A^H x &= b \\\\
///   \Leftrightarrow \overline{A^T} x &= b \\\\
///   \Leftrightarrow \overline{\overline{A^T} x} &= \overline{b} \\\\
///   \Leftrightarrow \overline{\overline{A^T}} \overline{x} &= \overline{b} \\\\
///   \Leftrightarrow A^T \overline{x} &= \overline{b}
/// \end{align*}
/// $$
///
/// So, we can handle this case by switching to "no transpose"
/// (which is equivalent to transposing the array since it will
/// be reinterpreted as Fortran layout) and applying the
/// elementwise conjugate to `x` and `b`.
///
pub trait SolveImpl: Scalar {
    /// LAPACK correspondance
    /// ----------------------
    ///
    /// | f32    | f64    | c32    | c64    |
    /// |:-------|:-------|:-------|:-------|
    /// | sgetrs | dgetrs | cgetrs | zgetrs |
    ///
    fn solve(l: MatrixLayout, t: Transpose, a: &[Self], p: &Pivot, b: &mut [Self]) -> Result<()>;
}

macro_rules! impl_solve {
    ($scalar:ty, $getrs:path) => {
        impl SolveImpl for $scalar {
            fn solve(
                l: MatrixLayout,
                t: Transpose,
                a: &[Self],
                ipiv: &Pivot,
                b: &mut [Self],
            ) -> Result<()> {
                let (t, conj) = match l {
                    MatrixLayout::C { .. } => match t {
                        Transpose::No => (Transpose::Transpose, false),
                        Transpose::Transpose => (Transpose::No, false),
                        Transpose::Hermite => (Transpose::No, true),
                    },
                    MatrixLayout::F { .. } => (t, false),
                };
                let (n, _) = l.size();
                let nrhs = 1;
                let ldb = l.lda();
                let mut info = 0;
                if conj {
                    for b_elem in &mut *b {
                        *b_elem = b_elem.conj();
                    }
                }
                unsafe {
                    $getrs(
                        t.as_ptr().cast(),
                        &n,
                        &nrhs,
                        AsPtr::as_ptr(a),
                        &l.lda(),
                        ipiv.as_ptr().cast(),
                        AsPtr::as_mut_ptr(b),
                        &ldb,
                        &mut info,
                    )
                };
                if conj {
                    for b_elem in &mut *b {
                        *b_elem = b_elem.conj();
                    }
                }
                info.as_lapack_result()?;
                Ok(())
            }
        }
    };
} // impl_solve!

impl_solve!(f64, lapack_sys::dgetrs_);
impl_solve!(f32, lapack_sys::sgetrs_);
impl_solve!(c64, lapack_sys::zgetrs_);
impl_solve!(c32, lapack_sys::cgetrs_);

/// Working memory for computing inverse matrix
pub struct InvWork<T: Scalar> {
    pub layout: MatrixLayout,
    pub work: Vec<MaybeUninit<T>>,
}

/// Helper trait to abstract `*getri` LAPACK rotuines for implementing [Lapack::inv]
///
/// LAPACK correspondance
/// ----------------------
///
/// | f32    | f64    | c32    | c64    |
/// |:-------|:-------|:-------|:-------|
/// | sgetri | dgetri | cgetri | zgetri |
///
pub trait InvWorkImpl: Sized {
    type Elem: Scalar;
    fn new(layout: MatrixLayout) -> Result<Self>;
    fn calc(&mut self, a: &mut [Self::Elem], p: &Pivot) -> Result<()>;
}

macro_rules! impl_inv_work {
    ($s:ty, $tri:path) => {
        impl InvWorkImpl for InvWork<$s> {
            type Elem = $s;

            fn new(layout: MatrixLayout) -> Result<Self> {
                let (n, _) = layout.size();
                let mut info = 0;
                let mut work_size = [Self::Elem::zero()];
                unsafe {
                    $tri(
                        &n,
                        std::ptr::null_mut(),
                        &layout.lda(),
                        std::ptr::null(),
                        AsPtr::as_mut_ptr(&mut work_size),
                        &(-1),
                        &mut info,
                    )
                };
                info.as_lapack_result()?;
                let lwork = work_size[0].to_usize().unwrap();
                let work = vec_uninit(lwork);
                Ok(InvWork { layout, work })
            }

            fn calc(&mut self, a: &mut [Self::Elem], ipiv: &Pivot) -> Result<()> {
                if self.layout.len() == 0 {
                    return Ok(());
                }
                let lwork = self.work.len().to_i32().unwrap();
                let mut info = 0;
                unsafe {
                    $tri(
                        &self.layout.len(),
                        AsPtr::as_mut_ptr(a),
                        &self.layout.lda(),
                        ipiv.as_ptr().cast(),
                        AsPtr::as_mut_ptr(&mut self.work),
                        &lwork,
                        &mut info,
                    )
                };
                info.as_lapack_result()?;
                Ok(())
            }
        }
    };
}

impl_inv_work!(c64, lapack_sys::zgetri_);
impl_inv_work!(c32, lapack_sys::cgetri_);
impl_inv_work!(f64, lapack_sys::dgetri_);
impl_inv_work!(f32, lapack_sys::sgetri_);
