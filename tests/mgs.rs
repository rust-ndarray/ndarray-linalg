use ndarray::*;
use ndarray_linalg::{mgs::*, *};

fn qr_full<A: Scalar + Lapack>() {
    const N: usize = 5;
    let rtol: A::Real = A::real(1e-9);

    let a: Array2<A> = random((N, N));
    let (q, r) = mgs(a.axis_iter(Axis(1)), N, rtol, Strategy::Terminate);
    assert_close_l2!(&q.dot(&r), &a, rtol);

    let qc: Array2<A> = conjugate(&q);
    assert_close_l2!(&qc.dot(&q), &Array::eye(N), rtol);
}

#[test]
fn qr_full_real() {
    qr_full::<f64>();
}

#[test]
fn qr_full_complex() {
    qr_full::<c64>();
}

fn qr<A: Scalar + Lapack>() {
    const N: usize = 4;
    let rtol: A::Real = A::real(1e-9);

    let a: Array2<A> = random((N, N / 2));
    let (q, r) = mgs(a.axis_iter(Axis(1)), N, rtol, Strategy::Terminate);
    assert_close_l2!(&q.dot(&r), &a, rtol);

    let qc: Array2<A> = conjugate(&q);
    assert_close_l2!(&qc.dot(&q), &Array::eye(N / 2), rtol);
}

#[test]
fn qr_real() {
    qr::<f64>();
}

#[test]
fn qr_complex() {
    qr::<c64>();
}

fn qr_over<A: Scalar + Lapack>() {
    const N: usize = 4;
    let rtol: A::Real = A::real(1e-9);

    let a: Array2<A> = random((N, N * 2));

    // Terminate
    let (q, r) = mgs(a.axis_iter(Axis(1)), N, rtol, Strategy::Terminate);
    let a_sub = a.slice(s![.., 0..N]);
    assert_close_l2!(&q.dot(&r), &a_sub, rtol);
    let qc: Array2<A> = conjugate(&q);
    assert_close_l2!(&qc.dot(&q), &Array::eye(N), rtol);

    // Skip
    let (q, r) = mgs(a.axis_iter(Axis(1)), N, rtol, Strategy::Skip);
    let a_sub = a.slice(s![.., 0..N]);
    assert_close_l2!(&q.dot(&r), &a_sub, rtol);
    let qc: Array2<A> = conjugate(&q);
    assert_close_l2!(&qc.dot(&q), &Array::eye(N), rtol);

    // Full
    let (q, r) = mgs(a.axis_iter(Axis(1)), N, rtol, Strategy::Full);
    assert_close_l2!(&q.dot(&r), &a, rtol);
    let qc: Array2<A> = conjugate(&q);
    assert_close_l2!(&qc.dot(&q), &Array::eye(N), rtol);
}

#[test]
fn qr_over_real() {
    qr_over::<f64>();
}

#[test]
fn qr_over_complex() {
    qr_over::<c64>();
}
