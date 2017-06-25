//! Define Errors

use ndarray::{Ixs, ShapeError};
use std::error;
use std::fmt;

pub type Result<T> = ::std::result::Result<T, LinalgError>;

/// Master Error type of this crate
#[derive(Debug, EnumError)]
pub enum LinalgError {
    NotSquare(NotSquareError),
    Lapack(LapackError),
    Stride(StrideError),
    MemoryCont(MemoryContError),
    Shape(ShapeError),
}

/// Error from LAPACK
#[derive(Debug, new)]
pub struct LapackError {
    pub return_code: i32,
}

impl fmt::Display for LapackError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LAPACK: return_code = {}", self.return_code)
    }
}

impl error::Error for LapackError {
    fn description(&self) -> &str {
        "LAPACK subroutine returns non-zero code"
    }
}

impl From<i32> for LapackError {
    fn from(code: i32) -> LapackError {
        LapackError { return_code: code }
    }
}

/// Error that matrix is not square
#[derive(Debug, new)]
pub struct NotSquareError {
    pub rows: i32,
    pub cols: i32,
}

impl fmt::Display for NotSquareError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Not square: rows({}) != cols({})", self.rows, self.cols)
    }
}

impl error::Error for NotSquareError {
    fn description(&self) -> &str {
        "Matrix is not square"
    }
}

/// Error that strides of the array is not supported
#[derive(Debug, new)]
pub struct StrideError {
    pub s0: Ixs,
    pub s1: Ixs,
}

impl fmt::Display for StrideError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "invalid stride: s0={}, s1={}", self.s0, self.s1)
    }
}

impl error::Error for StrideError {
    fn description(&self) -> &str {
        "invalid stride"
    }
}

/// Error that the memory is not aligned continously
#[derive(Debug, new)]
pub struct MemoryContError {}

impl fmt::Display for MemoryContError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Memory is not contiguous")
    }
}

impl error::Error for MemoryContError {
    fn description(&self) -> &str {
        "Memory is not contiguous"
    }
}
