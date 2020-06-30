use ndarray::*;
use ndarray_linalg::{krylov::*, *};

#[test]
fn aq_qh_mgs() {
    let a: Array2<f64> = random((5, 5));
    let v: Array1<f64> = random(5);
    let (q, h) = arnoldi_mgs(a.clone(), v, 1e-9);
    println!("A = \n{:?}", &a);
    println!("Q = \n{:?}", &q);
    println!("H = \n{:?}", &h);
    let aq = a.dot(&q);
    let qh = q.dot(&h);
    println!("AQ = \n{:?}", &aq);
    println!("QH = \n{:?}", &qh);
    close_l2(&aq, &qh, 1e-9);
}

#[test]
fn aq_qh_householder() {
    let a: Array2<f64> = random((5, 5));
    let v: Array1<f64> = random(5);
    let (q, h) = arnoldi_mgs(a.clone(), v, 1e-9);
    println!("A = \n{:?}", &a);
    println!("Q = \n{:?}", &q);
    println!("H = \n{:?}", &h);
    let aq = a.dot(&q);
    let qh = q.dot(&h);
    println!("AQ = \n{:?}", &aq);
    println!("QH = \n{:?}", &qh);
    close_l2(&aq, &qh, 1e-9);
}

#[test]
fn aq_qh_mgs_complex() {
    let a: Array2<c64> = random((5, 5));
    let v: Array1<c64> = random(5);
    let (q, h) = arnoldi_mgs(a.clone(), v, 1e-9);
    println!("A = \n{:?}", &a);
    println!("Q = \n{:?}", &q);
    println!("H = \n{:?}", &h);
    let aq = a.dot(&q);
    let qh = q.dot(&h);
    println!("AQ = \n{:?}", &aq);
    println!("QH = \n{:?}", &qh);
    close_l2(&aq, &qh, 1e-9);
}

#[test]
fn aq_qh_householder_complex() {
    let a: Array2<c64> = random((5, 5));
    let v: Array1<c64> = random(5);
    let (q, h) = arnoldi_mgs(a.clone(), v, 1e-9);
    println!("A = \n{:?}", &a);
    println!("Q = \n{:?}", &q);
    println!("H = \n{:?}", &h);
    let aq = a.dot(&q);
    let qh = q.dot(&h);
    println!("AQ = \n{:?}", &aq);
    println!("QH = \n{:?}", &qh);
    close_l2(&aq, &qh, 1e-9);
}
