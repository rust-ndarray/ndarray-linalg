//! Implement QR decomposition

extern crate lapack;

use std::cmp::min;
use self::lapack::fortran::*;
use num_traits::Zero;

use error::LapackError;

pub trait ImplQR: Sized {
    fn qr(n: usize, m: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>), LapackError>;
    fn lq(n: usize, m: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>), LapackError>;
}


macro_rules! impl_qr {
    ($qr:ident, $geqrf:path, $orgqr:path) => {
fn $qr(n: usize, m: usize, mut a: Vec<Self>) -> Result<(Vec<Self>, Vec<Self>), LapackError> {
    let mut info = 0;
    let n = n as i32;
    let m = m as i32;
    let k = min(m, n);
    let lda = m;
    let lw_default = 1000;
    let mut tau = vec![Self::zero(); k as usize];
    let mut work = vec![Self::zero(); lw_default];
    println!("a = \n{:?}", &a);
    // estimate lwork
    $geqrf(m, n, &mut a, lda, &mut tau, &mut work, -1, &mut info);
    let lwork_r = work[0] as i32;
    if lwork_r > lw_default as i32 {
        work = vec![Self::zero(); lwork_r as usize];
    }
    println!("lwork_r = {:?}", lwork_r);
    // calc R
    $geqrf(m, n, &mut a, lda, &mut tau, &mut work, lwork_r, &mut info);
    if info != 0 {
        return Err(From::from(info));
    }
    println!("r = \n{:?}", &a);
    println!("tau = \n{:?}", &tau);
    let r = a.clone();
    // re-estimate lwork
    $orgqr(k, k, k, &mut a, k, &mut tau, &mut work, -1, &mut info);
    let lwork_q = work[0] as i32;
    if lwork_q > lwork_r {
        work = vec![Self::zero(); lwork_q as usize];
    }
    println!("lwork_q = {:?}", lwork_q);
    println!("m = {:?}", m);
    println!("n = {:?}", n);
    println!("k = {:?}", k);
    // calc Q
    $orgqr(k,
           k,
           k,
           &mut a,
           k,
           &mut tau,
           &mut work,
           lwork_q,
           &mut info);
    if info == 0 {
        Ok((a, r))
    } else {
        Err(From::from(info))
    }
}
}} // endmacro

impl ImplQR for f64 {
    impl_qr!(qr, dgeqrf, dorgqr);
    impl_qr!(lq, dgelqf, dorglq);
}
