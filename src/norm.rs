
extern crate lapack;

use self::lapack::fortran::*;
use num_traits::Zero;

pub trait ImplNorm: Sized {
    fn norm_1(m: usize, n: usize, mut a: Vec<Self>) -> Self;
    fn norm_i(m: usize, n: usize, mut a: Vec<Self>) -> Self;
    fn norm_f(m: usize, n: usize, mut a: Vec<Self>) -> Self;
}

impl ImplNorm for f64 {
    fn norm_1(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = Vec::<Self>::new();
        dlange(b'o', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
    fn norm_i(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = vec![Self::zero(); m];
        dlange(b'i', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
    fn norm_f(m: usize, n: usize, mut a: Vec<Self>) -> Self {
        let mut work = Vec::<Self>::new();
        dlange(b'f', m as i32, n as i32, &mut a, m as i32, &mut work)
    }
}
