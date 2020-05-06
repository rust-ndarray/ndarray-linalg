use ndarray::*;
use ndarray_linalg::*;

#[test]
fn dgeev() {
    // https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dgeev_ex.f.htm
    let a: Array2<f64> = arr2(&[
        [-1.01, 0.86, -4.60, 3.31, -4.81],
        [3.98, 0.53, -7.04, 5.29, 3.55],
        [3.30, 8.26, -3.89, 8.20, -1.51],
        [4.43, 4.96, -7.66, -7.33, 6.18],
        [7.31, -6.43, -6.16, 2.47, 5.58],
    ]);
    let (e, vecs): (Array1<_>, Array2<_>) = (&a).eig().unwrap();
    assert_close_l2!(
        &e,
        &arr1(&[
            c64::new(2.86, 10.76),
            c64::new(2.86, -10.76),
            c64::new(-0.69, 4.70),
            c64::new(-0.69, -4.70),
            c64::new(-10.46, 0.00)
        ]),
        1.0e-3
    );

    /*
    let answer = &arr2(&[[c64::new(  0.11,  0.17), c64::new(  0.11, -0.17), c64::new(  0.73,  0.00), c64::new(  0.73,  0.00), c64::new(  0.46,  0.00)],
                         [c64::new(  0.41, -0.26), c64::new(  0.41,  0.26), c64::new( -0.03, -0.02), c64::new( -0.03,  0.02), c64::new(  0.34,  0.00)],
                         [c64::new(  0.10, -0.51), c64::new(  0.10,  0.51), c64::new(  0.19, -0.29), c64::new(  0.19,  0.29), c64::new(  0.31,  0.00)],
                         [c64::new(  0.40, -0.09), c64::new(  0.40,  0.09), c64::new( -0.08, -0.08), c64::new( -0.08,  0.08), c64::new( -0.74,  0.00)],
                         [c64::new(  0.54,  0.00), c64::new(  0.54,  0.00), c64::new( -0.29, -0.49), c64::new( -0.29,  0.49), c64::new(  0.16,  0.00)]]);
    */

    let a_c: Array2<c64> = a.map(|f| c64::new(*f, 0.0));
    for (i, v) in vecs.axis_iter(Axis(1)).enumerate() {
        let av = a_c.dot(&v);
        let ev = v.mapv(|f| e[i] * f);
        assert_close_l2!(&av, &ev, 1.0e-7);
    }
}

#[test]
fn fgeev() {
    // https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dgeev_ex.f.htm
    let a: Array2<f32> = arr2(&[
        [-1.01, 0.86, -4.60, 3.31, -4.81],
        [3.98, 0.53, -7.04, 5.29, 3.55],
        [3.30, 8.26, -3.89, 8.20, -1.51],
        [4.43, 4.96, -7.66, -7.33, 6.18],
        [7.31, -6.43, -6.16, 2.47, 5.58],
    ]);
    let (e, vecs): (Array1<_>, Array2<_>) = (&a).eig().unwrap();
    assert_close_l2!(
        &e,
        &arr1(&[
            c32::new(2.86, 10.76),
            c32::new(2.86, -10.76),
            c32::new(-0.69, 4.70),
            c32::new(-0.69, -4.70),
            c32::new(-10.46, 0.00)
        ]),
        1.0e-3
    );

    /*
    let answer = &arr2(&[[c32::new(  0.11,  0.17), c32::new(  0.11, -0.17), c32::new(  0.73,  0.00), c32::new(  0.73,  0.00), c32::new(  0.46,  0.00)],
                         [c32::new(  0.41, -0.26), c32::new(  0.41,  0.26), c32::new( -0.03, -0.02), c32::new( -0.03,  0.02), c32::new(  0.34,  0.00)],
                         [c32::new(  0.10, -0.51), c32::new(  0.10,  0.51), c32::new(  0.19, -0.29), c32::new(  0.19,  0.29), c32::new(  0.31,  0.00)],
                         [c32::new(  0.40, -0.09), c32::new(  0.40,  0.09), c32::new( -0.08, -0.08), c32::new( -0.08,  0.08), c32::new( -0.74,  0.00)],
                         [c32::new(  0.54,  0.00), c32::new(  0.54,  0.00), c32::new( -0.29, -0.49), c32::new( -0.29,  0.49), c32::new(  0.16,  0.00)]]);
    */

    let a_c: Array2<c32> = a.map(|f| c32::new(*f, 0.0));
    for (i, v) in vecs.axis_iter(Axis(1)).enumerate() {
        let av = a_c.dot(&v);
        let ev = v.mapv(|f| e[i] * f);
        assert_close_l2!(&av, &ev, 1.0e-5);
    }
}

#[test]
fn zgeev() {
    // https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/zgeev_ex.f.htm
    let a: Array2<c64> = arr2(&[
        [
            c64::new(-3.84, 2.25),
            c64::new(-8.94, -4.75),
            c64::new(8.95, -6.53),
            c64::new(-9.87, 4.82),
        ],
        [
            c64::new(-0.66, 0.83),
            c64::new(-4.40, -3.82),
            c64::new(-3.50, -4.26),
            c64::new(-3.15, 7.36),
        ],
        [
            c64::new(-3.99, -4.73),
            c64::new(-5.88, -6.60),
            c64::new(-3.36, -0.40),
            c64::new(-0.75, 5.23),
        ],
        [
            c64::new(7.74, 4.18),
            c64::new(3.66, -7.53),
            c64::new(2.58, 3.60),
            c64::new(4.59, 5.41),
        ],
    ]);
    let (e, vecs): (Array1<_>, Array2<_>) = (&a).eig().unwrap();
    assert_close_l2!(
        &e,
        &arr1(&[
            c64::new(-9.43, -12.98),
            c64::new(-3.44, 12.69),
            c64::new(0.11, -3.40),
            c64::new(5.76, 7.13)
        ]),
        1.0e-3
    );

    /*
    let answer = &arr2(&[[c64::new(  0.43,  0.33), c64::new(  0.83,  0.00), c64::new(  0.60,  0.00), c64::new( -0.31,  0.03)],
                         [c64::new(  0.51, -0.03), c64::new(  0.08, -0.25), c64::new( -0.40, -0.20), c64::new(  0.04,  0.34)],
                         [c64::new(  0.62,  0.00), c64::new( -0.25,  0.28), c64::new( -0.09, -0.48), c64::new(  0.36,  0.06)],
                         [c64::new( -0.23,  0.11), c64::new( -0.10, -0.32), c64::new( -0.43,  0.13), c64::new(  0.81,  0.00)]]);
    */

    for (i, v) in vecs.axis_iter(Axis(1)).enumerate() {
        let av = a.dot(&v);
        let ev = v.mapv(|f| e[i] * f);
        assert_close_l2!(&av, &ev, 1.0e-7);
    }
}

#[test]
fn cgeev() {
    // https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/zgeev_ex.f.htm
    let a: Array2<c32> = arr2(&[
        [
            c32::new(-3.84, 2.25),
            c32::new(-8.94, -4.75),
            c32::new(8.95, -6.53),
            c32::new(-9.87, 4.82),
        ],
        [
            c32::new(-0.66, 0.83),
            c32::new(-4.40, -3.82),
            c32::new(-3.50, -4.26),
            c32::new(-3.15, 7.36),
        ],
        [
            c32::new(-3.99, -4.73),
            c32::new(-5.88, -6.60),
            c32::new(-3.36, -0.40),
            c32::new(-0.75, 5.23),
        ],
        [
            c32::new(7.74, 4.18),
            c32::new(3.66, -7.53),
            c32::new(2.58, 3.60),
            c32::new(4.59, 5.41),
        ],
    ]);
    let (e, vecs): (Array1<_>, Array2<_>) = (&a).eig().unwrap();
    assert_close_l2!(
        &e,
        &arr1(&[
            c32::new(-9.43, -12.98),
            c32::new(-3.44, 12.69),
            c32::new(0.11, -3.40),
            c32::new(5.76, 7.13)
        ]),
        1.0e-3
    );

    /*
    let answer = &arr2(&[[c32::new(  0.43,  0.33), c32::new(  0.83,  0.00), c32::new(  0.60,  0.00), c32::new( -0.31,  0.03)],
                         [c32::new(  0.51, -0.03), c32::new(  0.08, -0.25), c32::new( -0.40, -0.20), c32::new(  0.04,  0.34)],
                         [c32::new(  0.62,  0.00), c32::new( -0.25,  0.28), c32::new( -0.09, -0.48), c32::new(  0.36,  0.06)],
                         [c32::new( -0.23,  0.11), c32::new( -0.10, -0.32), c32::new( -0.43,  0.13), c32::new(  0.81,  0.00)]]);
    */

    for (i, v) in vecs.axis_iter(Axis(1)).enumerate() {
        let av = a.dot(&v);
        let ev = v.mapv(|f| e[i] * f);
        assert_close_l2!(&av, &ev, 1.0e-5);
    }
}
