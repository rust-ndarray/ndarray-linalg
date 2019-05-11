use ndarray::*;
use ndarray_linalg::{krylov::*, *};

fn full<A: Scalar + Lapack>(rtol: A::Real) {
    const N: usize = 5;
    let a: Array2<A> = random((N, N));
    let (q, r) = mgs(a.axis_iter(Axis(1)), N, rtol, Strategy::Terminate);
    assert_close_l2!(&q.dot(&r), &a, rtol);
    let qc: Array2<A> = conjugate(&q);
    assert_close_l2!(&qc.dot(&q), &Array::eye(N), rtol);
}

#[test]
fn full_f32() {
    full::<f32>(1e-5);
}
#[test]
fn full_f64() {
    full::<f64>(1e-9);
}
#[test]
fn full_c32() {
    full::<c32>(1e-5);
}
#[test]
fn full_c64() {
    full::<c64>(1e-9);
}

fn half<A: Scalar + Lapack>(rtol: A::Real) {
    const N: usize = 4;
    let a: Array2<A> = random((N, N / 2));
    let (q, r) = mgs(a.axis_iter(Axis(1)), N, rtol, Strategy::Terminate);
    assert_close_l2!(&q.dot(&r), &a, rtol);
    let qc: Array2<A> = conjugate(&q);
    assert_close_l2!(&qc.dot(&q), &Array::eye(N / 2), rtol);
}

#[test]
fn half_f32() {
    half::<f32>(1e-5);
}
#[test]
fn half_f64() {
    half::<f64>(1e-9);
}
#[test]
fn half_c32() {
    half::<c32>(1e-5);
}
#[test]
fn half_c64() {
    half::<c64>(1e-9);
}

fn over<A: Scalar + Lapack>(rtol: A::Real) {
    const N: usize = 4;
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
fn over_f32() {
    over::<f32>(1e-5);
}
#[test]
fn over_f64() {
    over::<f64>(1e-9);
}
#[test]
fn over_c32() {
    over::<c32>(1e-5);
}
#[test]
fn over_c64() {
    over::<c64>(1e-9);
}
