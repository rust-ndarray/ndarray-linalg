use ndarray::*;
use ndarray_linalg::{krylov::*, *};

#[test]
fn mgs_full() {
    fn test<A: Scalar + Lapack>(rtol: A::Real) {
        const N: usize = 5;
        let a: Array2<A> = random((N, N));
        let (q, r) = mgs(a.axis_iter(Axis(1)), N, rtol, Strategy::Terminate);
        assert_close_l2!(&q.dot(&r), &a, rtol);
        let qc: Array2<A> = conjugate(&q);
        assert_close_l2!(&qc.dot(&q), &Array::eye(N), rtol);
    }

    test::<f32>(1e-5);
    test::<f64>(1e-9);
    test::<c32>(1e-5);
    test::<c64>(1e-9);
}

#[test]
fn mgs_half() {
    fn test<A: Scalar + Lapack>(rtol: A::Real) {
        const N: usize = 4;
        let a: Array2<A> = random((N, N / 2));
        let (q, r) = mgs(a.axis_iter(Axis(1)), N, rtol, Strategy::Terminate);
        assert_close_l2!(&q.dot(&r), &a, rtol);
        let qc: Array2<A> = conjugate(&q);
        assert_close_l2!(&qc.dot(&q), &Array::eye(N / 2), rtol);
    }

    test::<f32>(1e-5);
    test::<f64>(1e-9);
    test::<c32>(1e-5);
    test::<c64>(1e-9);
}

#[test]
fn mgs_over() {
    fn test<A: Scalar + Lapack>(rtol: A::Real) {
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

    test::<f32>(1e-5);
    test::<f64>(1e-9);
    test::<c32>(1e-5);
    test::<c64>(1e-9);
}

#[test]
fn householder_full() {
    fn test<A: Scalar + Lapack>(rtol: A::Real) {
        const N: usize = 5;
        let a: Array2<A> = random((N, N));
        let (q, r) = householder(a.axis_iter(Axis(1)), N, rtol, Strategy::Terminate);
        let qc: Array2<A> = conjugate(&q);
        assert_close_l2!(&qc.dot(&q), &Array::eye(N), rtol; "Check Q^H Q = I");
        assert_close_l2!(&q.dot(&r), &a, rtol; "Check A = QR");
    }

    test::<f32>(1e-5);
    test::<f64>(1e-9);
    test::<c32>(1e-5);
    test::<c64>(1e-9);
}

#[test]
fn householder_half() {
    fn test<A: Scalar + Lapack>(rtol: A::Real) {
        const N: usize = 4;
        let a: Array2<A> = random((N, N / 2));
        let (q, r) = householder(a.axis_iter(Axis(1)), N, rtol, Strategy::Terminate);
        let qc: Array2<A> = conjugate(&q);
        assert_close_l2!(&qc.dot(&q), &Array::eye(N / 2), rtol; "Check Q^H Q = I");
        assert_close_l2!(&q.dot(&r), &a, rtol; "Check A = QR");
    }

    test::<f32>(1e-5);
    test::<f64>(1e-9);
    test::<c32>(1e-5);
    test::<c64>(1e-9);
}

#[test]
fn householder_over() {
    fn test<A: Scalar + Lapack>(rtol: A::Real) {
        const N: usize = 4;
        let a: Array2<A> = random((N, N * 2));

        // Terminate
        let (q, r) = householder(a.axis_iter(Axis(1)), N, rtol, Strategy::Terminate);
        let a_sub = a.slice(s![.., 0..N]);
        let qc: Array2<A> = conjugate(&q);
        assert_close_l2!(&qc.dot(&q), &Array::eye(N), rtol; "Check Q^H Q = I");
        assert_close_l2!(&q.dot(&r), &a_sub, rtol; "Check A = QR");

        // Skip
        let (q, r) = householder(a.axis_iter(Axis(1)), N, rtol, Strategy::Skip);
        let a_sub = a.slice(s![.., 0..N]);
        let qc: Array2<A> = conjugate(&q);
        assert_close_l2!(&qc.dot(&q), &Array::eye(N), rtol);
        assert_close_l2!(&q.dot(&r), &a_sub, rtol);

        // Full
        let (q, r) = householder(a.axis_iter(Axis(1)), N, rtol, Strategy::Full);
        let qc: Array2<A> = conjugate(&q);
        assert_close_l2!(&qc.dot(&q), &Array::eye(N), rtol);
        assert_close_l2!(&q.dot(&r), &a, rtol);
    }

    test::<f32>(1e-5);
    test::<f64>(1e-9);
    test::<c32>(1e-5);
    test::<c64>(1e-9);
}
