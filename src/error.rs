//! Define Errors

use std::error;
use std::fmt;

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
pub struct StrideError {}

impl fmt::Display for StrideError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "invalid stride")
    }
}

impl error::Error for StrideError {
    fn description(&self) -> &str {
        "invalid stride"
    }
}

#[derive(Debug)]
pub enum LinalgError {
    NotSquare(NotSquareError),
    Lapack(LapackError),
    Stride(StrideError),
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            LinalgError::NotSquare(ref err) => err.fmt(f),
            LinalgError::Lapack(ref err) => err.fmt(f),
            LinalgError::Stride(ref err) => err.fmt(f),
        }
    }
}

impl error::Error for LinalgError {
    fn description(&self) -> &str {
        match *self {
            LinalgError::NotSquare(ref err) => err.description(),
            LinalgError::Lapack(ref err) => err.description(),
            LinalgError::Stride(ref err) => err.description(),
        }
    }
}

impl From<NotSquareError> for LinalgError {
    fn from(err: NotSquareError) -> LinalgError {
        LinalgError::NotSquare(err)
    }
}

impl From<LapackError> for LinalgError {
    fn from(err: LapackError) -> LinalgError {
        LinalgError::Lapack(err)
    }
}

impl From<StrideError> for LinalgError {
    fn from(err: StrideError) -> LinalgError {
        LinalgError::Stride(err)
    }
}
