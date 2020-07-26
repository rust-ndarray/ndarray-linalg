//! Memory layout of matrices
//!
//! Different from ndarray format which consists of shape and strides,
//! matrix format in LAPACK consists of row or column size and leading dimension.
//!
//! ndarray format and stride
//! --------------------------
//!
//! Let us consider 3-dimensional array for explaining ndarray structure.
//! The address of `(x,y,z)`-element in ndarray satisfies following relation:
//!
//! ```text
//! shape = [Nx, Ny, Nz]
//!     where Nx > 0, Ny > 0, Nz > 0
//! stride = [Sx, Sy, Sz]
//!
//! &data[(x, y, z)] = &data[(0, 0, 0)] + Sx*x + Sy*y + Sz*z
//!     for x < Nx, y < Ny, z < Nz
//! ```
//!
//! The array is called
//!
//! - C-continuous if `[Sx, Sy, Sz] = [Nz*Ny, Nz, 1]`
//! - F(Fortran)-continuous if `[Sx, Sy, Sz] = [1, Nx, Nx*Ny]`
//!
//! Strides of ndarray `[Sx, Sy, Sz]` take arbitrary value,
//! e.g. it can be non-ordered `Sy > Sx > Sz`, or can be negative `Sx < 0`.
//! If the minimum of `[Sx, Sy, Sz]` equals to `1`,
//! the value of elements fills `data` memory region and called "continuous".
//! Non-continuous ndarray is useful to get sub-array without copying data.
//!
//! Matrix layout for LAPACK
//! -------------------------
//!
//! LAPACK interface focuses on the linear algebra operations for F-continuous 2-dimensional array.
//! Under this restriction, stride becomes far simpler; we only have to consider the case `[1, S]`
//! This `S` for a matrix `A` is called "leading dimension of the array A" in LAPACK document, and denoted by `lda`.
//!

use cauchy::Scalar;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    C { row: i32, lda: i32 },
    F { col: i32, lda: i32 },
}

impl MatrixLayout {
    pub fn size(&self) -> (i32, i32) {
        match *self {
            MatrixLayout::C { row, lda } => (row, lda),
            MatrixLayout::F { col, lda } => (lda, col),
        }
    }

    pub fn resized(&self, row: i32, col: i32) -> MatrixLayout {
        match *self {
            MatrixLayout::C { .. } => MatrixLayout::C { row, lda: col },
            MatrixLayout::F { .. } => MatrixLayout::F { col, lda: row },
        }
    }

    pub fn lda(&self) -> i32 {
        std::cmp::max(
            1,
            match *self {
                MatrixLayout::C { lda, .. } | MatrixLayout::F { lda, .. } => lda,
            },
        )
    }

    pub fn len(&self) -> i32 {
        match *self {
            MatrixLayout::C { row, .. } => row,
            MatrixLayout::F { col, .. } => col,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn same_order(&self, other: &MatrixLayout) -> bool {
        match (self, other) {
            (MatrixLayout::C { .. }, MatrixLayout::C { .. }) => true,
            (MatrixLayout::F { .. }, MatrixLayout::F { .. }) => true,
            _ => false,
        }
    }

    pub fn toggle_order(&self) -> Self {
        match *self {
            MatrixLayout::C { row, lda } => MatrixLayout::F { lda: row, col: lda },
            MatrixLayout::F { col, lda } => MatrixLayout::C { row: lda, lda: col },
        }
    }

    /// Transpose without changing memory representation
    ///
    /// C-contigious row=2, lda=3
    ///
    /// ```text
    /// [[1, 2, 3]
    ///  [4, 5, 6]]
    /// ```
    ///
    /// and F-contigious col=2, lda=3
    ///
    /// ```text
    /// [[1, 4]
    ///  [2, 5]
    ///  [3, 6]]
    /// ```
    ///
    /// have same memory representation `[1, 2, 3, 4, 5, 6]`, and this toggles them.
    ///
    /// ```
    /// # use lax::layout::*;
    /// let layout = MatrixLayout::C { row: 2, lda: 3 };
    /// assert_eq!(layout.t(), MatrixLayout::F { col: 2, lda: 3 });
    /// ```
    pub fn t(&self) -> Self {
        match *self {
            MatrixLayout::C { row, lda } => MatrixLayout::F { col: row, lda },
            MatrixLayout::F { col, lda } => MatrixLayout::C { row: col, lda },
        }
    }
}

/// In-place transpose of a square matrix by keeping F/C layout
///
/// Transpose for C-continuous array
///
/// ```rust
/// # use lax::layout::*;
/// let layout = MatrixLayout::C { row: 2, lda: 2 };
/// let mut a = vec![1., 2., 3., 4.];
/// square_transpose(layout, &mut a);
/// assert_eq!(a, &[1., 3., 2., 4.]);
/// ```
///
/// Transpose for F-continuous array
///
/// ```rust
/// # use lax::layout::*;
/// let layout = MatrixLayout::F { col: 2, lda: 2 };
/// let mut a = vec![1., 3., 2., 4.];
/// square_transpose(layout, &mut a);
/// assert_eq!(a, &[1., 2., 3., 4.]);
/// ```
///
/// Panics
/// ------
/// - If size of `a` and `layout` size mismatch
///
pub fn square_transpose<T: Scalar>(layout: MatrixLayout, a: &mut [T]) {
    let (m, n) = layout.size();
    let n = n as usize;
    let m = m as usize;
    assert_eq!(a.len(), n * m);
    for i in 0..m {
        for j in (i + 1)..n {
            let a_ij = a[i * n + j];
            let a_ji = a[j * m + i];
            a[i * n + j] = a_ji.conj();
            a[j * m + i] = a_ij.conj();
        }
    }
}

/// Out-place transpose for general matrix
///
/// Inplace transpose of non-square matrices is hard.
/// See also: https://en.wikipedia.org/wiki/In-place_matrix_transposition
///
/// ```rust
/// # use lax::layout::*;
/// let layout = MatrixLayout::C { row: 2, lda: 3 };
/// let a = vec![1., 2., 3., 4., 5., 6.];
/// let mut b = vec![0.0; a.len()];
/// let l = transpose(layout, &a, &mut b);
/// assert_eq!(l, MatrixLayout::F { col: 3, lda: 2 });
/// assert_eq!(b, &[1., 4., 2., 5., 3., 6.]);
/// ```
///
/// ```rust
/// # use lax::layout::*;
/// let layout = MatrixLayout::F { col: 2, lda: 3 };
/// let a = vec![1., 2., 3., 4., 5., 6.];
/// let mut b = vec![0.0; a.len()];
/// let l = transpose(layout, &a, &mut b);
/// assert_eq!(l, MatrixLayout::C { row: 3, lda: 2 });
/// assert_eq!(b, &[1., 4., 2., 5., 3., 6.]);
/// ```
///
/// Panics
/// ------
/// - If size of `a` and `layout` size mismatch
///
pub fn transpose<T: Scalar>(layout: MatrixLayout, from: &[T], to: &mut [T]) -> MatrixLayout {
    let (m, n) = layout.size();
    let transposed = layout.resized(n, m).t();
    let m = m as usize;
    let n = n as usize;
    assert_eq!(from.len(), m * n);
    assert_eq!(to.len(), m * n);

    match layout {
        MatrixLayout::C { .. } => {
            for i in 0..m {
                for j in 0..n {
                    to[j * m + i] = from[i * n + j];
                }
            }
        }
        MatrixLayout::F { .. } => {
            for i in 0..m {
                for j in 0..n {
                    to[i * n + j] = from[j * m + i];
                }
            }
        }
    }
    transposed
}
