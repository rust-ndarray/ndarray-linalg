use crate::layout::*;
use cauchy::*;
use num_traits::Zero;
use std::ops::{Index, IndexMut};

/// Represents a tridiagonal matrix as 3 one-dimensional vectors.
///
/// ```text
/// [d0, u1,  0,   ...,       0,
///  l1, d1, u2,            ...,
///   0, l2, d2,
///  ...           ...,  u{n-1},
///   0,  ...,  l{n-1},  d{n-1},]
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct Tridiagonal<A: Scalar> {
    /// layout of raw matrix
    pub l: MatrixLayout,
    /// (n-1) sub-diagonal elements of matrix.
    pub dl: Vec<A>,
    /// (n) diagonal elements of matrix.
    pub d: Vec<A>,
    /// (n-1) super-diagonal elements of matrix.
    pub du: Vec<A>,
}

impl<A: Scalar> Tridiagonal<A> {
    pub fn opnorm_one(&self) -> A::Real {
        let mut col_sum: Vec<A::Real> = self.d.iter().map(|val| val.abs()).collect();
        for i in 0..col_sum.len() {
            if i < self.dl.len() {
                col_sum[i] += self.dl[i].abs();
            }
            if i > 0 {
                col_sum[i] += self.du[i - 1].abs();
            }
        }
        let mut max = A::Real::zero();
        for &val in &col_sum {
            if max < val {
                max = val;
            }
        }
        max
    }
}

impl<A: Scalar> Index<(i32, i32)> for Tridiagonal<A> {
    type Output = A;
    #[inline]
    fn index(&self, (row, col): (i32, i32)) -> &A {
        let (n, _) = self.l.size();
        assert!(
            std::cmp::max(row, col) < n,
            "ndarray: index {:?} is out of bounds for array of shape {}",
            [row, col],
            n
        );
        match row - col {
            0 => &self.d[row as usize],
            1 => &self.dl[col as usize],
            -1 => &self.du[row as usize],
            _ => panic!(
                "ndarray-linalg::tridiagonal: index {:?} is not tridiagonal element",
                [row, col]
            ),
        }
    }
}

impl<A: Scalar> Index<[i32; 2]> for Tridiagonal<A> {
    type Output = A;
    #[inline]
    fn index(&self, [row, col]: [i32; 2]) -> &A {
        &self[(row, col)]
    }
}

impl<A: Scalar> IndexMut<(i32, i32)> for Tridiagonal<A> {
    #[inline]
    fn index_mut(&mut self, (row, col): (i32, i32)) -> &mut A {
        let (n, _) = self.l.size();
        assert!(
            std::cmp::max(row, col) < n,
            "ndarray: index {:?} is out of bounds for array of shape {}",
            [row, col],
            n
        );
        match row - col {
            0 => &mut self.d[row as usize],
            1 => &mut self.dl[col as usize],
            -1 => &mut self.du[row as usize],
            _ => panic!(
                "ndarray-linalg::tridiagonal: index {:?} is not tridiagonal element",
                [row, col]
            ),
        }
    }
}

impl<A: Scalar> IndexMut<[i32; 2]> for Tridiagonal<A> {
    #[inline]
    fn index_mut(&mut self, [row, col]: [i32; 2]) -> &mut A {
        &mut self[(row, col)]
    }
}
