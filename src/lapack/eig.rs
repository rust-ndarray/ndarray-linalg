//! Eigenvalue decomposition for general matrices

use lapacke;
use num_traits::Zero;

use crate::error::*;
use crate::layout::MatrixLayout;
use crate::types::*;

use super::into_result;

/// Wraps `*geev` for real/complex
pub trait Eig_: Scalar {
    unsafe fn eig(calc_v: bool, l: MatrixLayout, a: &mut [Self]) -> Result<(Vec<Self::Complex>, Vec<Self>)>;
}

macro_rules! impl_eig_complex {
    ($scalar:ty, $ev:path) => {
        impl Eig_ for $scalar {
            unsafe fn eig(calc_v: bool, l: MatrixLayout, mut a: &mut [Self]) -> Result<(Vec<Self::Complex>, Vec<Self::Complex>)> {
                let (n, _) = l.size();
                let jobvl = if calc_v { b'V' } else { b'N' };
                let mut w = vec![Self::Complex::zero(); n as usize];
                let mut vl = vec![Self::Complex::zero(); (n * n) as usize];
                let mut vr = Vec::new();
                let info = $ev(l.lapacke_layout(), jobvl, b'N', n, &mut a, n, &mut w, &mut vl, n, &mut vr, n);
                into_result(info, (w, vl))
            }
        }
    };
}

macro_rules! impl_eig_real {
    ($scalar:ty, $ev:path, $cn:ty) => {
        impl Eig_ for $scalar {
            unsafe fn eig(calc_v: bool, l: MatrixLayout, mut a: &mut [Self]) -> Result<(Vec<Self::Complex>, Vec<Self::Real>)> {
                let (n, _) = l.size();
                let jobvl = if calc_v { b'V' } else { b'N' };
                let mut wr = vec![Self::Real::zero(); n as usize];
                let mut wi = vec![Self::Real::zero(); n as usize];
                let mut vl = vec![Self::Real::zero(); (n * n) as usize];
                let mut vr = Vec::new();
                let info = $ev(l.lapacke_layout(), jobvl, b'N', n, &mut a, n, &mut wr, &mut wi, &mut vl, n, &mut vr, n);
                let w: Vec<$cn> = wr.iter().zip(wi.iter()).map(|(&r, &i)| <$cn>::new(r, i)).collect();
                into_result(info, (w, vl))
            }
        }
    };
}

impl_eig_real!(f64, lapacke::dgeev, c64);
impl_eig_real!(f32, lapacke::sgeev, c32);
impl_eig_complex!(c64, lapacke::zgeev);
impl_eig_complex!(c32, lapacke::cgeev);