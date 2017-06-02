
use ndarray::*;
use lapack::c;

use super::error::*;

pub type LDA = i32;
pub type Col = i32;
pub type Row = i32;

#[derive(Debug, Clone, Copy)]
pub enum Layout {
    C((Row, LDA)),
    F((Col, LDA)),
}

impl Layout {
    pub fn size(&self) -> (Row, Col) {
        match *self {
            Layout::C((row, lda)) => (row, lda),
            Layout::F((col, lda)) => (lda, col),
        }
    }

    pub fn lda(&self) -> LDA {
        match *self {
            Layout::C((_, lda)) => lda,
            Layout::F((_, lda)) => lda,
        }
    }

    pub fn lapacke_layout(&self) -> c::Layout {
        match *self {
            Layout::C(_) => c::Layout::RowMajor,
            Layout::F(_) => c::Layout::ColumnMajor,
        }
    }
}

pub trait AllocatedArray {
    type Scalar;
    fn layout(&self) -> Result<Layout>;
    fn square_layout(&self) -> Result<Layout>;
    fn as_allocated(&self) -> Result<&[Self::Scalar]>;
}

pub trait AllocatedArrayMut: AllocatedArray {
    fn as_allocated_mut(&mut self) -> Result<&mut [Self::Scalar]>;
}

impl<A, S> AllocatedArray for ArrayBase<S, Ix2>
    where S: Data<Elem = A>
{
    type Scalar = A;

    fn layout(&self) -> Result<Layout> {
        let strides = self.strides();
        if ::std::cmp::min(strides[0], strides[1]) != 1 {
            return Err(StrideError::new(strides[0], strides[1]).into());
        }
        if strides[0] > strides[1] {
            Ok(Layout::C((self.rows() as i32, self.cols() as i32)))
        } else {
            Ok(Layout::F((self.cols() as i32, self.rows() as i32)))
        }
    }

    fn square_layout(&self) -> Result<Layout> {
        let l = self.layout()?;
        let (n, m) = l.size();
        if n == m {
            Ok(l)
        } else {
            Err(NotSquareError::new(n, m).into())
        }
    }

    fn as_allocated(&self) -> Result<&[A]> {
        let slice = self.as_slice_memory_order().ok_or(MemoryContError::new())?;
        Ok(slice)
    }
}

impl<A, S> AllocatedArrayMut for ArrayBase<S, Ix2>
    where S: DataMut<Elem = A>
{
    fn as_allocated_mut(&mut self) -> Result<&mut [A]> {
        let slice = self.as_slice_memory_order_mut().ok_or(MemoryContError::new())?;
        Ok(slice)
    }
}

pub fn reconstruct<A, S>(l: Layout, a: Vec<A>) -> Result<ArrayBase<S, Ix2>>
    where S: DataOwned<Elem = A>
{
    Ok(match l {
        Layout::C((row, col)) => ArrayBase::from_shape_vec((row as usize, col as usize), a)?,
        Layout::F((col, row)) => ArrayBase::from_shape_vec((row as usize, col as usize).f(), a)?,
    })
}
