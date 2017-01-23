include!("header.rs");

#[test]
fn inv_random() {
    let r_dist = RealNormal::new(0., 1.);
    let a = Array::<f64, _>::random((3, 3), r_dist);
    let ai = a.clone().inv().unwrap();
    let id = Array::eye(3);
    ai.dot(&a).assert_allclose_l2(&id, 1e-7);
}

#[test]
fn inv_random_t() {
    let r_dist = RealNormal::new(0., 1.);
    let a = Array::<f64, _>::random((3, 3), r_dist).reversed_axes();
    let ai = a.clone().inv().unwrap();
    let id = Array::eye(3);
    ai.dot(&a).assert_allclose_l2(&id, 1e-7);
}

#[test]
#[should_panic]
fn inv_error() {
    // do not have inverse
    let a = Array::range(1., 10., 1.).into_shape((3, 3)).unwrap();
    let _ = a.clone().inv().unwrap();
}
