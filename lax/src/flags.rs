//! Charactor flags, e.g. `'T'`, used in LAPACK API

/// Upper/Lower specification for seveal usages
#[derive(Debug, Clone, Copy)]
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
    pub fn as_ptr(&self) -> *const i8 {
        self as *const UPLO as *const i8
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Transpose {
    No = b'N',
    Transpose = b'T',
    Hermite = b'C',
}

impl Transpose {
    /// To use Fortran LAPACK API in lapack-sys crate
    pub fn as_ptr(&self) -> *const i8 {
        self as *const Transpose as *const i8
    }
}

#[derive(Debug, Clone, Copy)]
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
    pub fn as_ptr(&self) -> *const i8 {
        self as *const NormType as *const i8
    }
}

/// Flag for calculating eigenvectors or not
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub fn as_ptr(&self) -> *const i8 {
        self as *const JobEv as *const i8
    }
}

/// Specifies how many of the columns of *U* and rows of *V*ᵀ are computed and returned.
///
/// For an input array of shape *m*×*n*, the following are computed:
#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(u8)]
pub enum JobSvd {
    /// All *m* columns of *U* and all *n* rows of *V*ᵀ.
    All = b'A',
    /// The first min(*m*,*n*) columns of *U* and the first min(*m*,*n*) rows of *V*ᵀ.
    Some = b'S',
    /// No columns of *U* or rows of *V*ᵀ.
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

    pub fn as_ptr(&self) -> *const i8 {
        self as *const JobSvd as *const i8
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Diag {
    Unit = b'U',
    NonUnit = b'N',
}

impl Diag {
    pub fn as_ptr(&self) -> *const i8 {
        self as *const Diag as *const i8
    }
}
