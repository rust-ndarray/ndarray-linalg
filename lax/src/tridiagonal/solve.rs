use crate::{error::*, layout::*, *};
use cauchy::*;

pub trait SolveTridiagonalImpl: Scalar {
    fn solve_tridiagonal(
        lu: &LUFactorizedTridiagonal<Self>,
        bl: MatrixLayout,
        t: Transpose,
        b: &mut [Self],
    ) -> Result<()>;
}

macro_rules! impl_solve_tridiagonal {
    ($s:ty, $trs:path) => {
        impl SolveTridiagonalImpl for $s {
            fn solve_tridiagonal(
                lu: &LUFactorizedTridiagonal<Self>,
                b_layout: MatrixLayout,
                t: Transpose,
                b: &mut [Self],
            ) -> Result<()> {
                let (n, _) = lu.a.l.size();
                let ipiv = &lu.ipiv;
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
                let (ldb, nrhs) = b_layout.size();
                let mut info = 0;
                unsafe {
                    $trs(
                        t.as_ptr().cast(),
                        &n,
                        &nrhs,
                        AsPtr::as_ptr(&lu.a.dl),
                        AsPtr::as_ptr(&lu.a.d),
                        AsPtr::as_ptr(&lu.a.du),
                        AsPtr::as_ptr(&lu.du2),
                        ipiv.as_ptr().cast(),
                        AsPtr::as_mut_ptr(b_t.as_mut().map(|v| v.as_mut_slice()).unwrap_or(b)),
                        &ldb,
                        &mut info,
                    );
                }
                info.as_lapack_result()?;
                if let Some(b_t) = b_t {
                    transpose_over(b_layout, &b_t, b);
                }
                Ok(())
            }
        }
    };
}

impl_solve_tridiagonal!(c64, lapack_sys::zgttrs_);
impl_solve_tridiagonal!(c32, lapack_sys::cgttrs_);
impl_solve_tridiagonal!(f64, lapack_sys::dgttrs_);
impl_solve_tridiagonal!(f32, lapack_sys::sgttrs_);
