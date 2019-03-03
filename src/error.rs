//! Define Errors

use ndarray::{Ixs, ShapeError};
use std::error;
use std::fmt;

pub type Result<T> = ::std::result::Result<T, LinalgError>;

/// Master Error type of this crate
#[derive(Debug)]
pub enum LinalgError {
    /// Matrix is not square
    NotSquare { rows: i32, cols: i32 },
    /// LAPACK subroutine returns non-zero code
    Lapack { return_code: i32 },
    /// Strides of the array is not supported
    InvalidStride { s0: Ixs, s1: Ixs },
    /// Memory is not aligned continously
    MemoryNotCont,
    /// Array is null (0-sized)
    NullArray,
    /// Strides of the array is not supported
    Shape(ShapeError),
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LinalgError::NotSquare { rows, cols } => write!(f, "Not square: rows({}) != cols({})", rows, cols),
            LinalgError::Lapack { return_code } => write!(f, "LAPACK: return_code = {}", return_code),
            LinalgError::InvalidStride { s0, s1 } => write!(f, "invalid stride: s0={}, s1={}", s0, s1),
            LinalgError::MemoryNotCont => write!(f, "Memory is not contiguous"),
            LinalgError::NullArray => write!(f, "Array is null (0-sized)"),
            LinalgError::Shape(err) => write!(f, "Shape Error: {}", err),
        }
    }
}

impl error::Error for LinalgError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            LinalgError::Shape(err) => Some(err),
            _ => None,
        }
    }
}

impl From<ShapeError> for LinalgError {
    fn from(error: ShapeError) -> LinalgError {
        LinalgError::Shape(error)
    }
}
