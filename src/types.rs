
pub use num_complex::Complex32 as c32;
pub use num_complex::Complex64 as c64;
use num_traits::Float;
use std::ops::*;

pub trait AssociatedReal: Sized {
    type Real: Float + Mul<Self, Output = Self>;
}
pub trait AssociatedComplex: Sized {
    type Complex;
}

macro_rules! impl_assoc {
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
}} // impl_assoc!

impl_assoc!(f64, c64);
impl_assoc!(f32, c32);
