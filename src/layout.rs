
use ndarray::*;
use lapack::c;

use super::error::*;

pub type LDA = i32;
pub type LEN = i32;
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

    pub fn resized(&self, row: Row, col: Col) -> Layout {
        match *self {
            Layout::C(_) => Layout::C((row, col)),
            Layout::F(_) => Layout::F((col, row)),
        }
    }

    pub fn lda(&self) -> LDA {
        match *self {
            Layout::C((_, lda)) => lda,
            Layout::F((_, lda)) => lda,
        }
    }

    pub fn len(&self) -> LEN {
        match *self {
            Layout::C((row, _)) => row,
            Layout::F((col, _)) => col,
        }
    }

    pub fn lapacke_layout(&self) -> c::Layout {
        match *self {
            Layout::C(_) => c::Layout::RowMajor,
            Layout::F(_) => c::Layout::ColumnMajor,
        }
    }

    pub fn same_order(&self, other: &Layout) -> bool {
        match *self {
            Layout::C(_) => {
                match *other {
                    Layout::C(_) => true,
                    Layout::F(_) => false,
                }
            }
            Layout::F(_) => {
                match *other {
                    Layout::C(_) => false,
                    Layout::F(_) => true,
                }
            }
        }
    }

    pub fn as_shape(&self) -> Shape<Ix2> {
        match *self {
            Layout::C((row, col)) => (row as usize, col as usize).into_shape(),
            Layout::F((col, row)) => (row as usize, col as usize).f().into_shape(),
        }
    }

    pub fn t(&self) -> Self {
        match *self {
            Layout::C((row, col)) => Layout::F((col, row)),
            Layout::F((col, row)) => Layout::C((row, col)),
        }
    }
}

pub trait AllocatedArray {
    type Elem;
    fn layout(&self) -> Result<Layout>;
    fn square_layout(&self) -> Result<Layout>;
    fn as_allocated(&self) -> Result<&[Self::Elem]>;
}

pub trait AllocatedArrayMut: AllocatedArray {
    fn as_allocated_mut(&mut self) -> Result<&mut [Self::Elem]>;
}

impl<A, S> AllocatedArray for ArrayBase<S, Ix2>
    where S: Data<Elem = A>
{
    type Elem = A;

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
        Ok(self.as_slice_memory_order().ok_or(MemoryContError::new())?)
    }
}

impl<A, S> AllocatedArrayMut for ArrayBase<S, Ix2>
    where S: DataMut<Elem = A>
{
    fn as_allocated_mut(&mut self) -> Result<&mut [A]> {
        Ok(self.as_slice_memory_order_mut().ok_or(MemoryContError::new())?)
    }
}

impl<A, S> AllocatedArray for ArrayBase<S, Ix1>
    where S: Data<Elem = A>
{
    type Elem = A;

    fn layout(&self) -> Result<Layout> {
        Ok(Layout::F((1, self.len() as i32)))
    }

    fn square_layout(&self) -> Result<Layout> {
        Err(NotSquareError::new(1, self.len() as i32).into())
    }

    fn as_allocated(&self) -> Result<&[A]> {
        Ok(self.as_slice_memory_order().ok_or(MemoryContError::new())?)
    }
}

impl<A, S> AllocatedArrayMut for ArrayBase<S, Ix1>
    where S: DataMut<Elem = A>
{
    fn as_allocated_mut(&mut self) -> Result<&mut [A]> {
        Ok(self.as_slice_memory_order_mut().ok_or(MemoryContError::new())?)
    }
}


pub fn reconstruct<A, S>(l: Layout, a: Vec<A>) -> Result<ArrayBase<S, Ix2>>
    where S: DataOwned<Elem = A>
{
    Ok(ArrayBase::from_shape_vec(l.as_shape(), a)?)
}

pub fn uninitialized<A, S>(l: Layout) -> ArrayBase<S, Ix2>
    where A: Copy,
          S: DataOwned<Elem = A>
{
    unsafe { ArrayBase::uninitialized(l.as_shape()) }
}

pub fn replicate<A, Sv, So, D>(a: &ArrayBase<Sv, D>) -> ArrayBase<So, D>
    where A: Copy,
          Sv: Data<Elem = A>,
          So: DataOwned<Elem = A> + DataMut,
          D: Dimension
{
    let mut b = unsafe { ArrayBase::uninitialized(a.dim()) };
    b.assign(a);
    b
}

pub fn clone_with_layout<A, Si, So>(l: Layout, a: &ArrayBase<Si, Ix2>) -> ArrayBase<So, Ix2>
    where A: Copy,
          Si: Data<Elem = A>,
          So: DataOwned<Elem = A> + DataMut
{
    let mut b = uninitialized(l);
    b.assign(a);
    b
}
