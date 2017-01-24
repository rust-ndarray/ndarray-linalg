include!("header.rs");

fn assert_almost_eq(a: f64, b: f64) {
    let rel_dev = (a - b).abs() / (a.abs() + b.abs());
    if rel_dev > 1.0e-7 {
        panic!("a={:?}, b={:?} are not almost equal", a, b);
    }
}

#[test]
fn vector_norm() {
    let a = Array::range(1., 10., 1.);
    assert_almost_eq(a.norm(), 285.0.sqrt());
}

#[test]
fn matrix_opnorm_square() {
    let a = Array::range(1., 10., 1.).into_shape((3, 3)).unwrap();
    assert_almost_eq(a.opnorm_1(), 18.0);
    assert_almost_eq(a.opnorm_i(), 24.0);
    assert_almost_eq(a.opnorm_f(), 285.0.sqrt());
}

#[test]
fn matrix_opnorm_square_t() {
    let a = Array::range(1., 10., 1.).into_shape((3, 3)).unwrap().reversed_axes();
    assert_almost_eq(a.opnorm_1(), 24.0);
    assert_almost_eq(a.opnorm_i(), 18.0);
    assert_almost_eq(a.opnorm_f(), 285.0.sqrt());
}

#[test]
fn matrix_opnorm_3x4() {
    let a = Array::range(1., 13., 1.).into_shape((3, 4)).unwrap();
    assert_almost_eq(a.opnorm_1(), 24.0);
    assert_almost_eq(a.opnorm_i(), 42.0);
    assert_almost_eq(a.opnorm_f(), 650.0.sqrt());
}

#[test]
fn matrix_opnorm_3x4_t() {
    let a = Array::range(1., 13., 1.)
        .into_shape((3, 4))
        .unwrap()
        .reversed_axes();
    assert_almost_eq(a.opnorm_1(), 42.0);
    assert_almost_eq(a.opnorm_i(), 24.0);
    assert_almost_eq(a.opnorm_f(), 650.0.sqrt());
}

#[test]
fn matrix_opnorm_4x3() {
    let a = Array::range(1., 13., 1.).into_shape((4, 3)).unwrap();
    assert_almost_eq(a.opnorm_1(), 30.0);
    assert_almost_eq(a.opnorm_i(), 33.0);
    assert_almost_eq(a.opnorm_f(), 650.0.sqrt());
}

#[test]
fn matrix_opnorm_4x3_t() {
    let a = Array::range(1., 13., 1.)
        .into_shape((4, 3))
        .unwrap()
        .reversed_axes();
    assert_almost_eq(a.opnorm_1(), 33.0);
    assert_almost_eq(a.opnorm_i(), 30.0);
    assert_almost_eq(a.opnorm_f(), 650.0.sqrt());
}
