//! Basic types and their methods for linear algebra

use std::ops::*;
use std::fmt::Debug;
use std::iter::Sum;
use num_complex::Complex;
use num_traits::*;
use rand::Rng;
use rand::distributions::*;
use ndarray::LinalgScalar;

use super::lapack_traits::LapackScalar;

pub use num_complex::Complex32 as c32;
pub use num_complex::Complex64 as c64;

macro_rules! trait_alias {
    ($name:ident: $($t:ident),*) => {

pub trait $name : $($t +)* {}

impl<T> $name for T where T: $($t +)* {}

}} // trait_alias!

trait_alias!(Field: LapackScalar,
             LinalgScalar,
             AssociatedReal,
             AssociatedComplex,
             Absolute,
             SquareRoot,
             Conjugate,
             RandNormal,
             Sum,
             Debug);

trait_alias!(RealField: Field, Float);

/// Define associating real float type
pub trait AssociatedReal: Sized {
    type Real: Float + Mul<Self, Output = Self>;
}

/// Define associating complex type
pub trait AssociatedComplex: Sized {
    type Complex;
}

/// Define `abs()` more generally
pub trait Absolute {
    type Output: RealField;
    fn squared(&self) -> Self::Output;
    fn abs(&self) -> Self::Output {
        self.squared().sqrt()
    }
}

/// Define `sqrt()` more generally
pub trait SquareRoot {
    fn sqrt(&self) -> Self;
}

/// Complex conjugate value
pub trait Conjugate: Copy {
    fn conj(self) -> Self;
}

/// Scalars which can be initialized from Gaussian random number
pub trait RandNormal {
    fn randn<R: Rng>(&mut R) -> Self;
}

macro_rules! impl_traits {
    ($real:ty, $complex:ty) => {

impl AssociatedReal for $real {
    type Real = $real;
}

impl AssociatedReal for $complex {
    type Real = $real;
}

impl AssociatedComplex for $real {
    type Complex = $complex;
}

impl AssociatedComplex for $complex {
    type Complex = $complex;
}

impl Absolute for $real {
    type Output = Self;
    fn squared(&self) -> Self::Output {
        *self * *self
    }
    fn abs(&self) -> Self::Output {
        Float::abs(*self)
    }
}

impl Absolute for $complex {
    type Output = $real;
    fn squared(&self) -> Self::Output {
        self.norm_sqr()
    }
    fn abs(&self) -> Self::Output {
        self.norm()
    }
}

impl SquareRoot for $real {
    fn sqrt(&self) -> Self {
        Float::sqrt(*self)
    }
}

impl SquareRoot for $complex {
    fn sqrt(&self) -> Self {
        Complex::sqrt(self)
    }
}

impl Conjugate for $real {
    fn conj(self) -> Self {
        self
    }
}

impl Conjugate for $complex {
    fn conj(self) -> Self {
        Complex::conj(&self)
    }
}

impl RandNormal for $real {
    fn randn<R: Rng>(rng: &mut R) -> Self {
        let dist = Normal::new(0., 1.);
        dist.ind_sample(rng) as $real
    }
}

impl RandNormal for $complex {
    fn randn<R: Rng>(rng: &mut R) -> Self {
        let dist = Normal::new(0., 1.);
        let re = dist.ind_sample(rng) as $real;
        let im = dist.ind_sample(rng) as $real;
        Self::new(re, im)
    }
}

}} // impl_traits!

impl_traits!(f64, c64);
impl_traits!(f32, c32);
