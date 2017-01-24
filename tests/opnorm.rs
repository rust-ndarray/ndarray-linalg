include!("header.rs");

#[test]
fn matrix_opnorm_square() {
    let a = Array::range(1., 10., 1.).into_shape((3, 3)).unwrap();
    a.opnorm_1().assert_close(18.0, 1e-7);
    a.opnorm_i().assert_close(24.0, 1e-7);
    a.opnorm_f().assert_close(285.0.sqrt(), 1e-7);
}

#[test]
fn matrix_opnorm_square_t() {
    let a = Array::range(1., 10., 1.).into_shape((3, 3)).unwrap().reversed_axes();
    a.opnorm_1().assert_close(24.0, 1e-7);
    a.opnorm_i().assert_close(18.0, 1e-7);
    a.opnorm_f().assert_close(285.0.sqrt(), 1e-7);
}

#[test]
fn matrix_opnorm_3x4() {
    let a = Array::range(1., 13., 1.).into_shape((3, 4)).unwrap();
    a.opnorm_1().assert_close(24.0, 1e-7);
    a.opnorm_i().assert_close(42.0, 1e-7);
    a.opnorm_f().assert_close(650.0.sqrt(), 1e-7);
}

#[test]
fn matrix_opnorm_3x4_t() {
    let a = Array::range(1., 13., 1.)
        .into_shape((3, 4))
        .unwrap()
        .reversed_axes();
    a.opnorm_1().assert_close(42.0, 1e-7);
    a.opnorm_i().assert_close(24.0, 1e-7);
    a.opnorm_f().assert_close(650.0.sqrt(), 1e-7);
}

#[test]
fn matrix_opnorm_4x3() {
    let a = Array::range(1., 13., 1.).into_shape((4, 3)).unwrap();
    a.opnorm_1().assert_close(30.0, 1e-7);
    a.opnorm_i().assert_close(33.0, 1e-7);
    a.opnorm_f().assert_close(650.0.sqrt(), 1e-7);
}

#[test]
fn matrix_opnorm_4x3_t() {
    let a = Array::range(1., 13., 1.)
        .into_shape((4, 3))
        .unwrap()
        .reversed_axes();
    a.opnorm_1().assert_close(33.0, 1e-7);
    a.opnorm_i().assert_close(30.0, 1e-7);
    a.opnorm_f().assert_close(650.0.sqrt(), 1e-7);
}
