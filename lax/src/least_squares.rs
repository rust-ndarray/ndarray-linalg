//! Least squares

use crate::{error::*, layout::*, *};
use cauchy::*;
use num_traits::{ToPrimitive, Zero};

/// Result of LeastSquares
pub struct LeastSquaresOutput<A: Scalar> {
    /// singular values
    pub singular_values: Vec<A::Real>,
    /// The rank of the input matrix A
    pub rank: i32,
}

/// Wraps `*gelsd`
pub trait LeastSquaresSvdDivideConquer_: Scalar {
    fn least_squares(
        a_layout: MatrixLayout,
        a: &mut [Self],
        b: &mut [Self],
    ) -> Result<LeastSquaresOutput<Self>>;

    fn least_squares_nrhs(
        a_layout: MatrixLayout,
        a: &mut [Self],
        b_layout: MatrixLayout,
        b: &mut [Self],
    ) -> Result<LeastSquaresOutput<Self>>;
}

macro_rules! impl_least_squares {
    (@real, $scalar:ty, $gelsd:path) => {
        impl_least_squares!(@body, $scalar, $gelsd, );
    };
    (@complex, $scalar:ty, $gelsd:path) => {
        impl_least_squares!(@body, $scalar, $gelsd, rwork);
    };

    (@body, $scalar:ty, $gelsd:path, $($rwork:ident),*) => {
        impl LeastSquaresSvdDivideConquer_ for $scalar {
            fn least_squares(
                l: MatrixLayout,
                a: &mut [Self],
                b: &mut [Self],
            ) -> Result<LeastSquaresOutput<Self>> {
                let b_layout = l.resized(b.len() as i32, 1);
                Self::least_squares_nrhs(l, a, b_layout, b)
            }

            fn least_squares_nrhs(
                a_layout: MatrixLayout,
                a: &mut [Self],
                b_layout: MatrixLayout,
                b: &mut [Self],
            ) -> Result<LeastSquaresOutput<Self>> {
                // Minimize |b - Ax|_2
                //
                // where
                //   A : (m, n)
                //   b : (max(m, n), nrhs)  // `b` has to store `x` on exit
                //   x : (n, nrhs)
                let (m, n) = a_layout.size();
                let (m_, nrhs) = b_layout.size();
                let k = m.min(n);
                assert!(m_ >= m);

                // Transpose if a is C-continuous
                let mut a_t = None;
                let a_layout = match a_layout {
                    MatrixLayout::C { .. } => {
                        a_t = Some(unsafe { vec_uninit( a.len()) });
                        transpose(a_layout, a, a_t.as_mut().unwrap())
                    }
                    MatrixLayout::F { .. } => a_layout,
                };

                // Transpose if b is C-continuous
                let mut b_t = None;
                let b_layout = match b_layout {
                    MatrixLayout::C { .. } => {
                        b_t = Some(unsafe { vec_uninit( b.len()) });
                        transpose(b_layout, b, b_t.as_mut().unwrap())
                    }
                    MatrixLayout::F { .. } => b_layout,
                };

                let rcond: Self::Real = -1.;
                let mut singular_values: Vec<Self::Real> = unsafe { vec_uninit( k as usize) };
                let mut rank: i32 = 0;

                // eval work size
                let mut info = 0;
                let mut work_size = [Self::zero()];
                let mut iwork_size = [0];
                $(
                let mut $rwork = [Self::Real::zero()];
                )*
                unsafe {
                    $gelsd(
                        m,
                        n,
                        nrhs,
                        a_t.as_mut().map(|v| v.as_mut_slice()).unwrap_or(a),
                        a_layout.lda(),
                        b_t.as_mut().map(|v| v.as_mut_slice()).unwrap_or(b),
                        b_layout.lda(),
                        &mut singular_values,
                        rcond,
                        &mut rank,
                        &mut work_size,
                        -1,
                        $(&mut $rwork,)*
                        &mut iwork_size,
                        &mut info,
                    )
                };
                info.as_lapack_result()?;

                // calc
                let lwork = work_size[0].to_usize().unwrap();
                let mut work = unsafe { vec_uninit( lwork) };
                let liwork = iwork_size[0].to_usize().unwrap();
                let mut iwork = unsafe { vec_uninit( liwork) };
                $(
                let lrwork = $rwork[0].to_usize().unwrap();
                let mut $rwork = unsafe { vec_uninit( lrwork) };
                )*
                unsafe {
                    $gelsd(
                        m,
                        n,
                        nrhs,
                        a_t.as_mut().map(|v| v.as_mut_slice()).unwrap_or(a),
                        a_layout.lda(),
                        b_t.as_mut().map(|v| v.as_mut_slice()).unwrap_or(b),
                        b_layout.lda(),
                        &mut singular_values,
                        rcond,
                        &mut rank,
                        &mut work,
                        lwork as i32,
                        $(&mut $rwork,)*
                        &mut iwork,
                        &mut info,
                    );
                }
                info.as_lapack_result()?;

                // Skip a_t -> a transpose because A has been destroyed
                // Re-transpose b
                if let Some(b_t) = b_t {
                    transpose(b_layout, &b_t, b);
                }

                Ok(LeastSquaresOutput {
                    singular_values,
                    rank,
                })
            }
        }
    };
}

impl_least_squares!(@real, f64, lapack::dgelsd);
impl_least_squares!(@real, f32, lapack::sgelsd);
impl_least_squares!(@complex, c64, lapack::zgelsd);
impl_least_squares!(@complex, c32, lapack::cgelsd);
