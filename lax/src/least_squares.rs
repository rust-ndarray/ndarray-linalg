//! Least squares

use crate::{error::*, layout::MatrixLayout};
use cauchy::*;
use num_traits::Zero;

/// Result of LeastSquares
pub struct LeastSquaresOutput<A: Scalar> {
    /// singular values
    pub singular_values: Vec<A::Real>,
    /// The rank of the input matrix A
    pub rank: i32,
}

/// Wraps `*gelsd`
pub trait LeastSquaresSvdDivideConquer_: Scalar {
    unsafe fn least_squares(
        a_layout: MatrixLayout,
        a: &mut [Self],
        b: &mut [Self],
    ) -> Result<LeastSquaresOutput<Self>>;

    unsafe fn least_squares_nrhs(
        a_layout: MatrixLayout,
        a: &mut [Self],
        b_layout: MatrixLayout,
        b: &mut [Self],
    ) -> Result<LeastSquaresOutput<Self>>;
}

macro_rules! impl_least_squares {
    ($scalar:ty, $gelsd:path) => {
        impl LeastSquaresSvdDivideConquer_ for $scalar {
            unsafe fn least_squares(
                a_layout: MatrixLayout,
                a: &mut [Self],
                b: &mut [Self],
            ) -> Result<LeastSquaresOutput<Self>> {
                let (m, n) = a_layout.size();
                if (m as usize) > b.len() || (n as usize) > b.len() {
                    return Err(Error::InvalidShape);
                }
                let k = ::std::cmp::min(m, n);
                let nrhs = 1;
                let ldb = match a_layout {
                    MatrixLayout::F { .. } => m.max(n),
                    MatrixLayout::C { .. } => 1,
                };
                let rcond: Self::Real = -1.;
                let mut singular_values: Vec<Self::Real> = vec![Self::Real::zero(); k as usize];
                let mut rank: i32 = 0;

                $gelsd(
                    a_layout.lapacke_layout(),
                    m,
                    n,
                    nrhs,
                    a,
                    a_layout.lda(),
                    b,
                    ldb,
                    &mut singular_values,
                    rcond,
                    &mut rank,
                )
                .as_lapack_result()?;

                Ok(LeastSquaresOutput {
                    singular_values,
                    rank,
                })
            }

            unsafe fn least_squares_nrhs(
                a_layout: MatrixLayout,
                a: &mut [Self],
                b_layout: MatrixLayout,
                b: &mut [Self],
            ) -> Result<LeastSquaresOutput<Self>> {
                let (m, n) = a_layout.size();
                if (m as usize) > b.len()
                    || (n as usize) > b.len()
                    || a_layout.lapacke_layout() != b_layout.lapacke_layout()
                {
                    return Err(Error::InvalidShape);
                }
                let k = ::std::cmp::min(m, n);
                let nrhs = b_layout.size().1;
                let rcond: Self::Real = -1.;
                let mut singular_values: Vec<Self::Real> = vec![Self::Real::zero(); k as usize];
                let mut rank: i32 = 0;

                $gelsd(
                    a_layout.lapacke_layout(),
                    m,
                    n,
                    nrhs,
                    a,
                    a_layout.lda(),
                    b,
                    b_layout.lda(),
                    &mut singular_values,
                    rcond,
                    &mut rank,
                )
                .as_lapack_result()?;
                Ok(LeastSquaresOutput {
                    singular_values,
                    rank,
                })
            }
        }
    };
}

impl_least_squares!(f64, lapacke::dgelsd);
impl_least_squares!(f32, lapacke::sgelsd);
impl_least_squares!(c64, lapacke::zgelsd);
impl_least_squares!(c32, lapacke::cgelsd);
