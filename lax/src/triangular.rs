//! Implement linear solver and inverse matrix

use crate::{error::*, layout::*, *};
use cauchy::*;

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Diag {
    Unit = b'U',
    NonUnit = b'N',
}

/// Wraps `*trtri` and `*trtrs`
pub trait Triangular_: Scalar {
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
    ($scalar:ty, $trtri:path, $trtrs:path) => {
        impl Triangular_ for $scalar {
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
                        a_t = Some(unsafe { vec_uninit(a.len()) });
                        transpose(a_layout, a, a_t.as_mut().unwrap())
                    }
                    MatrixLayout::F { .. } => a_layout,
                };

                // Transpose if b is C-continuous
                let mut b_t = None;
                let b_layout = match b_layout {
                    MatrixLayout::C { .. } => {
                        b_t = Some(unsafe { vec_uninit(b.len()) });
                        transpose(b_layout, b, b_t.as_mut().unwrap())
                    }
                    MatrixLayout::F { .. } => b_layout,
                };

                let (m, n) = a_layout.size();
                let (n_, nrhs) = b_layout.size();
                assert_eq!(n, n_);

                let mut info = 0;
                unsafe {
                    $trtrs(
                        uplo as u8,
                        Transpose::No as u8,
                        diag as u8,
                        m,
                        nrhs,
                        a_t.as_ref().map(|v| v.as_slice()).unwrap_or(a),
                        a_layout.lda(),
                        b_t.as_mut().map(|v| v.as_mut_slice()).unwrap_or(b),
                        b_layout.lda(),
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                // Re-transpose b
                if let Some(b_t) = b_t {
                    transpose(b_layout, &b_t, b);
                }
                Ok(())
            }
        }
    };
} // impl_triangular!

impl_triangular!(f64, lapack::dtrtri, lapack::dtrtrs);
impl_triangular!(f32, lapack::strtri, lapack::strtrs);
impl_triangular!(c64, lapack::ztrtri, lapack::ztrtrs);
impl_triangular!(c32, lapack::ctrtri, lapack::ctrtrs);
