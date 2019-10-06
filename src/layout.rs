//! Memory layout of matrices

use lapacke;
use ndarray::*;

use super::error::*;

pub type LDA = i32;
pub type LEN = i32;
pub type Col = i32;
pub type Row = i32;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatrixLayout {
    C((Row, LDA)),
    F((Col, LDA)),
}

impl MatrixLayout {
    pub fn size(&self) -> (Row, Col) {
        match *self {
            MatrixLayout::C((row, lda)) => (row, lda),
            MatrixLayout::F((col, lda)) => (lda, col),
        }
    }

    pub fn resized(&self, row: Row, col: Col) -> MatrixLayout {
        match *self {
            MatrixLayout::C(_) => MatrixLayout::C((row, col)),
            MatrixLayout::F(_) => MatrixLayout::F((col, row)),
        }
    }

    pub fn lda(&self) -> LDA {
        std::cmp::max(
            1,
            match *self {
                MatrixLayout::C((_, lda)) | MatrixLayout::F((_, lda)) => lda,
            },
        )
    }

    pub fn len(&self) -> LEN {
        match *self {
            MatrixLayout::C((row, _)) => row,
            MatrixLayout::F((col, _)) => col,
        }
    }

    pub fn lapacke_layout(&self) -> lapacke::Layout {
        match *self {
            MatrixLayout::C(_) => lapacke::Layout::RowMajor,
            MatrixLayout::F(_) => lapacke::Layout::ColumnMajor,
        }
    }

    pub fn same_order(&self, other: &MatrixLayout) -> bool {
        self.lapacke_layout() == other.lapacke_layout()
    }

    pub fn as_shape(&self) -> Shape<Ix2> {
        match *self {
            MatrixLayout::C((row, col)) => (row as usize, col as usize).into_shape(),
            MatrixLayout::F((col, row)) => (row as usize, col as usize).f().into_shape(),
        }
    }

    pub fn toggle_order(&self) -> Self {
        match *self {
            MatrixLayout::C((row, col)) => MatrixLayout::F((col, row)),
            MatrixLayout::F((col, row)) => MatrixLayout::C((row, col)),
        }
    }
}

pub trait AllocatedArray {
    type Elem;
    fn layout(&self) -> Result<MatrixLayout>;
    fn square_layout(&self) -> Result<MatrixLayout>;
    /// Returns Ok iff the matrix is square (without computing the layout).
    fn ensure_square(&self) -> Result<()>;
    fn as_allocated(&self) -> Result<&[Self::Elem]>;
}

pub trait AllocatedArrayMut: AllocatedArray {
    fn as_allocated_mut(&mut self) -> Result<&mut [Self::Elem]>;
}

impl<A, S> AllocatedArray for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
{
    type Elem = A;

    fn layout(&self) -> Result<MatrixLayout> {
        let shape = self.shape();
        let strides = self.strides();
        if shape[0] == strides[1] as usize {
            return Ok(MatrixLayout::F((self.ncols() as i32, self.nrows() as i32)));
        }
        if shape[1] == strides[0] as usize {
            return Ok(MatrixLayout::C((self.nrows() as i32, self.ncols() as i32)));
        }
        Err(LinalgError::InvalidStride {
            s0: strides[0],
            s1: strides[1],
        })
    }

    fn square_layout(&self) -> Result<MatrixLayout> {
        let l = self.layout()?;
        let (n, m) = l.size();
        if n == m {
            Ok(l)
        } else {
            Err(LinalgError::NotSquare { rows: n, cols: m })
        }
    }

    fn ensure_square(&self) -> Result<()> {
        if self.is_square() {
            Ok(())
        } else {
            Err(LinalgError::NotSquare {
                rows: self.nrows() as i32,
                cols: self.ncols() as i32,
            })
        }
    }

    fn as_allocated(&self) -> Result<&[A]> {
        Ok(self.as_slice_memory_order().ok_or_else(|| LinalgError::MemoryNotCont)?)
    }
}

impl<A, S> AllocatedArrayMut for ArrayBase<S, Ix2>
where
    S: DataMut<Elem = A>,
{
    fn as_allocated_mut(&mut self) -> Result<&mut [A]> {
        Ok(self
            .as_slice_memory_order_mut()
            .ok_or_else(|| LinalgError::MemoryNotCont)?)
    }
}
