//! Linear Algebra eXtension (LAX)
//! ===============================
//!
//! ndarray-free safe Rust wrapper for LAPACK FFI
//!
//! Linear equation, Inverse matrix, Condition number
//! --------------------------------------------------
//!
//! As the property of $A$, several types of triangular factorization are used:
//!
//! - LU-decomposition for general matrix
//!   - $PA = LU$, where $L$ is lower matrix, $U$ is upper matrix, and $P$ is permutation matrix
//! - Bunch-Kaufman diagonal pivoting method for nonpositive-definite Hermitian matrix
//!   - $A = U D U^\dagger$, where $U$ is upper matrix,
//!     $D$ is Hermitian and block diagonal with 1-by-1 and 2-by-2 diagonal blocks.
//!
//! | matrix type                     | Triangler factorization (TRF) | Solve (TRS) | Inverse matrix (TRI) | Reciprocal condition number (CON) |
//! |:--------------------------------|:------------------------------|:------------|:---------------------|:----------------------------------|
//! | General (GE)                    | [lu]                          | [solve]     | [inv]                | [rcond]                           |
//! | Symmetric (SY) / Hermitian (HE) | [bk]                          | [solveh]    | [invh]               | -                                 |
//!
//! [lu]:    solve/trait.Solve_.html#tymethod.lu
//! [solve]: solve/trait.Solve_.html#tymethod.solve
//! [inv]:   solve/trait.Solve_.html#tymethod.inv
//! [rcond]: solve/trait.Solve_.html#tymethod.rcond
//!
//! [bk]:     solveh/trait.Solveh_.html#tymethod.bk
//! [solveh]: solveh/trait.Solveh_.html#tymethod.solveh
//! [invh]:   solveh/trait.Solveh_.html#tymethod.invh
//!
//! Eigenvalue Problem
//! -------------------
//!
//! Solve eigenvalue problem for a matrix $A$
//!
//! $$ Av_i = \lambda_i v_i $$
//!
//! or generalized eigenvalue problem
//!
//! $$ Av_i = \lambda_i B v_i $$
//!
//! | matrix type                     | Eigenvalue (EV) | Generalized Eigenvalue Problem (EG) |
//! |:--------------------------------|:----------------|:------------------------------------|
//! | General (GE)                    |[eig]            | -                                   |
//! | Symmetric (SY) / Hermitian (HE) |[eigh]           |[eigh_generalized]                   |
//!
//! [eig]:              eig/trait.Eig_.html#tymethod.eig
//! [eigh]:             eigh/trait.Eigh_.html#tymethod.eigh
//! [eigh_generalized]: eigh/trait.Eigh_.html#tymethod.eigh_generalized
//!
//! Singular Value Decomposition (SVD), Least square problem
//! ----------------------------------------------------------
//!
//! | matrix type  | Singular Value Decomposition (SVD) | SVD with divided-and-conquer (SDD) | Least square problem (LSD) |
//! |:-------------|:-----------------------------------|:-----------------------------------|:---------------------------|
//! | General (GE) | [svd]                              | [svddc]                            | [least_squares]            |
//!
//! [svd]:   svd/trait.SVD_.html#tymethod.svd
//! [svddc]: svddck/trait.SVDDC_.html#tymethod.svddc
//! [least_squares]: least_squares/trait.LeastSquaresSvdDivideConquer_.html#tymethod.least_squares

#[cfg(any(feature = "intel-mkl-system", feature = "intel-mkl-static"))]
extern crate intel_mkl_src as _src;

#[cfg(any(feature = "openblas-system", feature = "openblas-static"))]
extern crate openblas_src as _src;

#[cfg(any(feature = "netlib-system", feature = "netlib-static"))]
extern crate netlib_src as _src;

pub mod error;
pub mod layout;

mod cholesky;
mod eig;
mod eigh;
mod least_squares;
mod opnorm;
mod qr;
mod rcond;
mod solve;
mod solveh;
mod svd;
mod svddc;
mod triangular;
mod tridiagonal;

pub use self::cholesky::*;
pub use self::eig::*;
pub use self::eigh::*;
pub use self::least_squares::*;
pub use self::opnorm::*;
pub use self::qr::*;
pub use self::rcond::*;
pub use self::solve::*;
pub use self::solveh::*;
pub use self::svd::*;
pub use self::svddc::*;
pub use self::triangular::*;
pub use self::tridiagonal::*;

use cauchy::*;
use std::mem::MaybeUninit;

pub type Pivot = Vec<i32>;

/// Trait for primitive types which implements LAPACK subroutines
pub trait Lapack:
    OperatorNorm_
    + QR_
    + SVD_
    + SVDDC_
    + Solve_
    + Solveh_
    + Cholesky_
    + Eig_
    + Eigh_
    + Triangular_
    + Tridiagonal_
    + Rcond_
    + LeastSquaresSvdDivideConquer_
{
}

impl Lapack for f32 {}
impl Lapack for f64 {}
impl Lapack for c32 {}
impl Lapack for c64 {}

/// Helper for getting pointer of slice
pub(crate) trait AsPtr: Sized {
    type Elem;
    fn as_ptr(vec: &[Self]) -> *const Self::Elem;
    fn as_mut_ptr(vec: &mut [Self]) -> *mut Self::Elem;
}

macro_rules! impl_as_ptr {
    ($target:ty, $elem:ty) => {
        impl AsPtr for $target {
            type Elem = $elem;
            fn as_ptr(vec: &[Self]) -> *const Self::Elem {
                vec.as_ptr() as *const _
            }
            fn as_mut_ptr(vec: &mut [Self]) -> *mut Self::Elem {
                vec.as_mut_ptr() as *mut _
            }
        }
    };
}
impl_as_ptr!(f32, f32);
impl_as_ptr!(f64, f64);
impl_as_ptr!(c32, lapack_sys::__BindgenComplex<f32>);
impl_as_ptr!(c64, lapack_sys::__BindgenComplex<f64>);
impl_as_ptr!(MaybeUninit<f32>, f32);
impl_as_ptr!(MaybeUninit<f64>, f64);
impl_as_ptr!(MaybeUninit<c32>, lapack_sys::__BindgenComplex<f32>);
impl_as_ptr!(MaybeUninit<c64>, lapack_sys::__BindgenComplex<f64>);

pub(crate) trait VecAssumeInit {
    type Target;
    unsafe fn assume_init(self) -> Self::Target;
}

macro_rules! impl_vec_assume_init {
    ($e:ty) => {
        impl VecAssumeInit for Vec<MaybeUninit<$e>> {
            type Target = Vec<$e>;
            unsafe fn assume_init(self) -> Self::Target {
                // FIXME use Vec::into_raw_parts instead after stablized
                // https://doc.rust-lang.org/std/vec/struct.Vec.html#method.into_raw_parts
                let mut me = std::mem::ManuallyDrop::new(self);
                Vec::from_raw_parts(me.as_mut_ptr() as *mut $e, me.len(), me.capacity())
            }
        }
    };
}

impl_vec_assume_init!(f32);
impl_vec_assume_init!(f64);
impl_vec_assume_init!(c32);
impl_vec_assume_init!(c64);

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
pub enum EigenVectorFlag {
    Calc = b'V',
    Not = b'N',
}

impl EigenVectorFlag {
    pub fn is_calc(&self) -> bool {
        match self {
            EigenVectorFlag::Calc => true,
            EigenVectorFlag::Not => false,
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
        self as *const EigenVectorFlag as *const i8
    }
}

/// Create a vector without initialization
///
/// Safety
/// ------
/// - Memory is not initialized. Do not read the memory before write.
///
unsafe fn vec_uninit<T: Sized>(n: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(n);
    v.set_len(n);
    v
}

/// Create a vector without initialization
///
/// Safety
/// ------
/// - Memory is not initialized. Do not read the memory before write.
///
unsafe fn vec_uninit2<T: Sized>(n: usize) -> Vec<MaybeUninit<T>> {
    let mut v = Vec::with_capacity(n);
    v.set_len(n);
    v
}
