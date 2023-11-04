//! Linear problem for triangular matrices

use crate::{error::*, layout::*, *};
use cauchy::*;

/// Solve linear problem for triangular matrices
///
/// LAPACK correspondance
/// ----------------------
///
/// | f32    | f64    | c32    | c64    |
/// |:-------|:-------|:-------|:-------|
/// | strtrs | dtrtrs | ctrtrs | ztrtrs |
///
pub trait SolveTriangularImpl: Scalar {
    fn solve_triangular(
        al: MatrixLayout,
        bl: MatrixLayout,
        uplo: UPLO,
        d: Diag,
        a: &[Self],
        b: &mut [Self],
    ) -> Result<()>;
}

macro_rules! impl_triangular {
    ($scalar:ty, $trtrs:path) => {
        impl SolveTriangularImpl for $scalar {
            fn solve_triangular(
                a_layout: MatrixLayout,
                b_layout: MatrixLayout,
                uplo: UPLO,
                diag: Diag,
                a: &[Self],
                b: &mut [Self],
            ) -> Result<()> {
                // Transpose if a is C-continuous
                let mut a_t = None;
                let a_layout = match a_layout {
                    MatrixLayout::C { .. } => {
                        let (layout, t) = transpose(a_layout, a);
                        a_t = Some(t);
                        layout
                    }
                    MatrixLayout::F { .. } => a_layout,
                };

                // Transpose if b is C-continuous
                let mut b_t = None;
                let b_layout = match b_layout {
                    MatrixLayout::C { .. } => {
                        let (layout, t) = transpose(b_layout, b);
                        b_t = Some(t);
                        layout
                    }
                    MatrixLayout::F { .. } => b_layout,
                };

                let (m, n) = a_layout.size();
                let (n_, nrhs) = b_layout.size();
                assert_eq!(n, n_);

                let mut info = 0;
                unsafe {
                    $trtrs(
                        uplo.as_ptr().cast(),
                        Transpose::No.as_ptr().cast(),
                        diag.as_ptr().cast(),
                        &m,
                        &nrhs,
                        AsPtr::as_ptr(a_t.as_ref().map(|v| v.as_slice()).unwrap_or(a)),
                        &a_layout.lda(),
                        AsPtr::as_mut_ptr(b_t.as_mut().map(|v| v.as_mut_slice()).unwrap_or(b)),
                        &b_layout.lda(),
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                // Re-transpose b
                if let Some(b_t) = b_t {
                    transpose_over(b_layout, &b_t, b);
                }
                Ok(())
            }
        }
    };
} // impl_triangular!

impl_triangular!(f64, lapack_sys::dtrtrs_);
impl_triangular!(f32, lapack_sys::strtrs_);
impl_triangular!(c64, lapack_sys::ztrtrs_);
impl_triangular!(c32, lapack_sys::ctrtrs_);
