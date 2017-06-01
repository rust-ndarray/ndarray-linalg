
use ndarray::*;

use super::error::*;
use lapack::c::Layout as Layout_;

pub type Row = usize;
pub type Col = usize;

pub type Row_ = i32;
pub type Col_ = i32;

pub enum Layout {
    C((Row, Col)),
    F((Row, Col)),
}

impl Layout {
    pub fn size(&self) -> (Row, Col) {
        match self {
            &Layout::C(s) => s,
            &Layout::F(s) => s,
        }
    }

    pub fn ffi_size(&self) -> (Row_, Col_) {
        let (n, m) = self.size();
        (n as Row_, m as Col_)
    }

    pub fn ffi_layout(&self) -> Layout_ {
        match self {
            &Layout::C(_) => Layout_::RowMajor,
            &Layout::F(_) => Layout_::ColumnMajor,
        }
    }
}

pub trait AllocatedArray2D {
    type Scalar;
    fn layout(&self) -> Result<Layout>;
    fn square_layout(&self) -> Result<Layout>;
    fn as_allocated(&self) -> Result<&[Self::Scalar]>;
}

impl<A, S> AllocatedArray2D for ArrayBase<S, Ix2>
    where S: Data<Elem = A>
{
    type Scalar = A;

    fn layout(&self) -> Result<Layout> {
        let strides = self.strides();
        if ::std::cmp::min(strides[0], strides[1]) != 1 {
            return Err(StrideError::new(strides[0], strides[1]).into());
        }
        if strides[0] < strides[1] {
            Ok(Layout::C((self.rows(), self.cols())))
        } else {
            Ok(Layout::F((self.rows(), self.cols())))
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
