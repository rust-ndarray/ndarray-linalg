//! Define Errors

use std::error;
use std::fmt;
use ndarray::{Ixs, ShapeError};

#[derive(Debug, EnumError)]
pub enum LinalgError {
    NotSquare(NotSquareError),
    Lapack(LapackError),
    Stride(StrideError),
    Shape(ShapeError),
}

#[derive(Debug)]
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

#[derive(Debug)]
pub struct NotSquareError {
    pub rows: usize,
    pub cols: usize,
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

#[derive(Debug)]
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
