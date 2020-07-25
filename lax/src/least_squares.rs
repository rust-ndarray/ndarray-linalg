//! Least squares

use crate::{error::*, layout::*};
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

macro_rules! impl_least_squares_real {
    ($scalar:ty, $gelsd:path) => {
        impl LeastSquaresSvdDivideConquer_ for $scalar {
            unsafe fn least_squares(
                l: MatrixLayout,
                a: &mut [Self],
                b: &mut [Self],
            ) -> Result<LeastSquaresOutput<Self>> {
                let b_layout = l.resized(b.len() as i32, 1);
                Self::least_squares_nrhs(l, a, b_layout, b)
            }

            unsafe fn least_squares_nrhs(
                a_layout: MatrixLayout,
                a: &mut [Self],
                b_layout: MatrixLayout,
                b: &mut [Self],
            ) -> Result<LeastSquaresOutput<Self>> {
                // Minimize |b - Ax|_2
                //
                // where
                //   A : (m, n)
                //   b : (m, p)
                //   x : (n, p)
                let (m, n) = a_layout.size();
                let (m_, p) = b_layout.size();
                let k = m.min(n);
                assert_eq!(m, m_);

                // Transpose if a is C-continuous
                let mut a_t = None;
                let a_layout = match a_layout {
                    MatrixLayout::C { .. } => {
                        a_t = Some(vec![Self::zero(); a.len()]);
                        transpose(a_layout, a, a_t.as_mut().unwrap())
                    }
                    MatrixLayout::F { .. } => a_layout,
                };

                // Transpose if b is C-continuous
                let mut b_t = None;
                let b_layout = match b_layout {
                    MatrixLayout::C { .. } => {
                        b_t = Some(vec![Self::zero(); b.len()]);
                        transpose(b_layout, b, b_t.as_mut().unwrap())
                    }
                    MatrixLayout::F { .. } => b_layout,
                };

                let rcond: Self::Real = -1.;
                let mut singular_values: Vec<Self::Real> = vec![Self::Real::zero(); k as usize];
                let mut rank: i32 = 0;

                // eval work size
                let mut info = 0;
                let mut work_size = [Self::zero()];
                let mut iwork_size = [0];
                $gelsd(
                    m,
                    n,
                    p,
                    a_t.as_mut().map(|v| v.as_mut_slice()).unwrap_or(a),
                    a_layout.lda(),
                    b_t.as_mut().map(|v| v.as_mut_slice()).unwrap_or(b),
                    b_layout.lda(),
                    &mut singular_values,
                    rcond,
                    &mut rank,
                    &mut work_size,
                    -1,
                    &mut iwork_size,
                    &mut info,
                );
                info.as_lapack_result()?;

                // calc
                let lwork = work_size[0].to_usize().unwrap();
                let mut work = vec![Self::zero(); lwork];
                let liwork = iwork_size[0].to_usize().unwrap();
                let mut iwork = vec![0; liwork];
                $gelsd(
                    m,
                    n,
                    p,
                    a_t.as_mut().map(|v| v.as_mut_slice()).unwrap_or(a),
                    a_layout.lda(),
                    b_t.as_mut().map(|v| v.as_mut_slice()).unwrap_or(b),
                    b_layout.lda(),
                    &mut singular_values,
                    rcond,
                    &mut rank,
                    &mut work,
                    lwork as i32,
                    &mut iwork,
                    &mut info,
                );
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

impl_least_squares_real!(f64, lapack::dgelsd);
impl_least_squares_real!(f32, lapack::sgelsd);

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

impl_least_squares!(c64, lapacke::zgelsd);
impl_least_squares!(c32, lapacke::cgelsd);
