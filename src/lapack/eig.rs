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
                // If the j-th eigenvalue is real, then
                // eigenvector = [ vr[j], vr[j+n], vr[j+2*n], ... ].
                //
                // If the j-th and (j+1)-st eigenvalues form a complex conjugate pair, 
                // eigenvector(j)   = [ vr[j] + i*vr[j+1], vr[j+n] + i*vr[j+n+1], vr[j+2*n] + i*vr[j+2*n+1], ... ] and
                // eigenvector(j+1) = [ vr[j] - i*vr[j+1], vr[j+n] - i*vr[j+n+1], vr[j+2*n] - i*vr[j+2*n+1], ... ].
                // 
                // Therefore, if eigenvector(j) is written as [ v_{j0}, v_{j1}, v_{j2}, ... ],
                // you have to make 
                // v = vec![ v_{00}, v_{10}, v_{20}, ..., v_{jk}, v_{(j+1)k}, v_{(j+2)k}, ... ] (v.len() = n*n)
                // based on wi and vr.
                // After that, v is converted to Array2 (see ../eig.rs).
                let n = n as usize;
                let mut flg = false;
                let conj: Vec<i8> = wi.iter().map(|&i| {
                    if flg {
                        flg = false;
                        -1
                    } else if i != 0.0 {
                        flg = true;
                        1
                    } else {
                        0
                    }
                }).collect();
                let v: Vec<Self::Complex> = (0..n*n).map(|i| {
                    let j = i % n;
                    match conj[j] {
                         1 => Self::Complex::new(vr[i], vr[i+1]),
                        -1 => Self::Complex::new(vr[i-1], -vr[i]),
                         _ => Self::Complex::new(vr[i], 0.0),
                    }
                }).collect();
                
                into_result(info, (w, v))
            }
        }
    };
}

impl_eig_real!(f64, lapacke::dgeev);
impl_eig_real!(f32, lapacke::sgeev);
impl_eig_complex!(c64, lapacke::zgeev);
impl_eig_complex!(c32, lapacke::cgeev);