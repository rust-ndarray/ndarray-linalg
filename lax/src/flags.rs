//! Charactor flags, e.g. `'T'`, used in LAPACK API

/// Upper/Lower specification for seveal usages
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum UPLO {
    Upper = b'U',
    Lower = b'L',
}

impl UPLO {
    pub fn t(self) -> Self {
        match self {
            UPLO::Upper => UPLO::Lower,
            UPLO::Lower => UPLO::Upper,
        }
    }

    /// To use Fortran LAPACK API in lapack-sys crate
    pub fn as_ptr(&self) -> *const libc::c_char {
        self as *const UPLO as *const libc::c_char
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Transpose {
    No = b'N',
    Transpose = b'T',
    Hermite = b'C',
}

impl Transpose {
    /// To use Fortran LAPACK API in lapack-sys crate
    pub fn as_ptr(&self) -> *const libc::c_char {
        self as *const Transpose as *const libc::c_char
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum NormType {
    One = b'O',
    Infinity = b'I',
    Frobenius = b'F',
}

impl NormType {
    pub fn transpose(self) -> Self {
        match self {
            NormType::One => NormType::Infinity,
            NormType::Infinity => NormType::One,
            NormType::Frobenius => NormType::Frobenius,
        }
    }

    /// To use Fortran LAPACK API in lapack-sys crate
    pub fn as_ptr(&self) -> *const libc::c_char {
        self as *const NormType as *const libc::c_char
    }
}

/// Flag for calculating eigenvectors or not
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum JobEv {
    /// Calculate eigenvectors in addition to eigenvalues
    All = b'V',
    /// Do not calculate eigenvectors. Only calculate eigenvalues.
    None = b'N',
}

impl JobEv {
    pub fn is_calc(&self) -> bool {
        match self {
            JobEv::All => true,
            JobEv::None => false,
        }
    }

    pub fn then<T, F: FnOnce() -> T>(&self, f: F) -> Option<T> {
        if self.is_calc() {
            Some(f())
        } else {
            None
        }
    }

    /// To use Fortran LAPACK API in lapack-sys crate
    pub fn as_ptr(&self) -> *const libc::c_char {
        self as *const JobEv as *const libc::c_char
    }
}

/// Specifies how many singular vectors are computed
///
/// For an input matrix $A$ of shape $m \times n$,
/// the following are computed on the singular value decomposition $A = U\Sigma V^T$:
#[cfg_attr(doc, katexit::katexit)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum JobSvd {
    /// All $m$ columns of $U$, and/or all $n$ rows of $V^T$.
    All = b'A',
    /// The first $\min(m, n)$ columns of $U$ and/or the first $\min(m, n)$ rows of $V^T$.
    Some = b'S',
    /// No columns of $U$ and/or rows of $V^T$.
    None = b'N',
}

impl JobSvd {
    pub fn from_bool(calc_uv: bool) -> Self {
        if calc_uv {
            JobSvd::All
        } else {
            JobSvd::None
        }
    }

    pub fn as_ptr(&self) -> *const libc::c_char {
        self as *const JobSvd as *const libc::c_char
    }
}

/// Specify whether input triangular matrix is unit or not
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Diag {
    /// Unit triangular matrix, i.e. all diagonal elements of the matrix are `1`
    Unit = b'U',
    /// Non-unit triangular matrix. Its diagonal elements may be different from `1`
    NonUnit = b'N',
}

impl Diag {
    pub fn as_ptr(&self) -> *const libc::c_char {
        self as *const Diag as *const libc::c_char
    }
}
