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

    pub fn lapacke_layout(&self) -> lapacke::Layout {
        match *self {
            MatrixLayout::C { .. } => lapacke::Layout::RowMajor,
            MatrixLayout::F { .. } => lapacke::Layout::ColumnMajor,
        }
    }

    pub fn same_order(&self, other: &MatrixLayout) -> bool {
        self.lapacke_layout() == other.lapacke_layout()
    }

    pub fn toggle_order(&self) -> Self {
        match *self {
            MatrixLayout::C { row, lda } => MatrixLayout::F { lda: row, col: lda },
            MatrixLayout::F { col, lda } => MatrixLayout::C { row: lda, lda: col },
        }
    }
}
