//! Memory layout of matrices

use super::error::*;
use ndarray::*;

pub use lapack::layout::MatrixLayout;

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
            return Ok(MatrixLayout::F {
                col: self.ncols() as i32,
                lda: self.nrows() as i32,
            });
        }
        if shape[1] == strides[0] as usize {
            return Ok(MatrixLayout::C {
                row: self.nrows() as i32,
                lda: self.ncols() as i32,
            });
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
        Ok(self
            .as_slice_memory_order()
            .ok_or_else(|| LinalgError::MemoryNotCont)?)
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
