//! Eigenvalue decomposition for general matrices

use lapacke;
use num_traits::Zero;

use crate::error::*;
use crate::layout::MatrixLayout;
use crate::types::*;

use super::into_result;

/// Wraps `*geev` for real/complex
pub trait Eig_: Scalar {
    unsafe fn eig(calc_v: bool, l: MatrixLayout, a: &mut [Self]) -> Result<(Vec<Self::Complex>, Vec<Self::Complex>)>;
}

macro_rules! impl_eig_complex {
    ($scalar:ty, $ev:path) => {
        impl Eig_ for $scalar {
            unsafe fn eig(calc_v: bool, l: MatrixLayout, mut a: &mut [Self]) -> Result<(Vec<Self::Complex>, Vec<Self::Complex>)> {
                let (n, _) = l.size();
                let jobvr = if calc_v { b'V' } else { b'N' };
                let mut w = vec![Self::Complex::zero(); n as usize];
                let mut vl = Vec::new();
                let mut vr = vec![Self::Complex::zero(); (n * n) as usize];
                let info = $ev(l.lapacke_layout(), b'N', jobvr, n, &mut a, n, &mut w, &mut vl, n, &mut vr, n);
                into_result(info, (w, vr))
            }
        }
    };
}

macro_rules! impl_eig_real {
    ($scalar:ty, $ev:path) => {
        impl Eig_ for $scalar {
            unsafe fn eig(calc_v: bool, l: MatrixLayout, mut a: &mut [Self]) -> Result<(Vec<Self::Complex>, Vec<Self::Complex>)> {
                let (n, _) = l.size();
                let jobvr = if calc_v { b'V' } else { b'N' };
                let mut wr = vec![Self::Real::zero(); n as usize];
                let mut wi = vec![Self::Real::zero(); n as usize];
                let mut vl = Vec::new();
                let mut vr = vec![Self::Real::zero(); (n * n) as usize];
                let info = $ev(l.lapacke_layout(), b'N', jobvr, n, &mut a, n, &mut wr, &mut wi, &mut vl, n, &mut vr, n);
                let w: Vec<Self::Complex> = wr.iter().zip(wi.iter()).map(|(&r, &i)| Self::Complex::new(r, i)).collect();
                let n = n as usize;
                let mut conj = false;
                let mut v = vec![Self::Complex::zero(); n * n];
                for (i, c) in w.iter().enumerate() {
                    if conj {
                        for j in 0..n {
                            v[n*j+i] = Self::Complex::new(vr[n*j+i-1], -vr[n*j+i]);
                        }
                        conj = false;
                    } else if c.im != 0.0 {
                        conj = true;
                        for j in 0..n {
                            v[n*j+i] = Self::Complex::new(vr[n*j+i], vr[n*j+i+1]);
                        }
                    } else {
                        for j in 0..n {
                            v[n*j+i] = Self::Complex::new(vr[n*j+i], 0.0);
                        }
                    }
                }
                into_result(info, (w, v))
            }
        }
    };
}

impl_eig_real!(f64, lapacke::dgeev);
impl_eig_real!(f32, lapacke::sgeev);
impl_eig_complex!(c64, lapacke::zgeev);
impl_eig_complex!(c32, lapacke::cgeev);