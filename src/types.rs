
pub use num_complex::Complex32 as c32;
pub use num_complex::Complex64 as c64;
use num_complex::Complex;
use num_traits::Float;
use std::ops::*;
use rand::Rng;
use rand::distributions::*;

pub trait AssociatedReal: Sized {
    type Real: Float + Mul<Self, Output = Self>;
}
pub trait AssociatedComplex: Sized {
    type Complex;
}

/// Field with norm
pub trait Absolute {
    type Output: Float;
    fn squared(&self) -> Self::Output;
    fn abs(&self) -> Self::Output {
        self.squared().sqrt()
    }
}

pub trait Conjugate: Copy {
    fn conj(self) -> Self;
}

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
