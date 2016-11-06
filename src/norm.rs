
extern crate lapack;

use self::lapack::fortran::*;
use num_traits::Zero;

pub trait ImplNorm: Sized {
    fn norm_1(m: usize, n: usize, mut a: Vec<Self>) -> Self;
    fn norm_i(m: usize, n: usize, mut a: Vec<Self>) -> Self;
    fn norm_f(m: usize, n: usize, mut a: Vec<Self>) -> Self;
}

macro_rules! impl_norm {
    ($float:ty, $lange:path) => {
impl ImplNorm for $float {
    fn norm_1(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = Vec::<Self>::new();
        $lange(b'o', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
    fn norm_i(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = vec![Self::zero(); m];
        $lange(b'i', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
    fn norm_f(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = Vec::<Self>::new();
        $lange(b'f', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
}
}} // end macro_rules

impl_norm!(f64, dlange);
impl_norm!(f32, slange);
