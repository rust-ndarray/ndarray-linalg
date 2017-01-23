include!("header.rs");

#[test]
fn ssqrt_symmetric_random() {
    let r_dist = RealNormal::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist);
    a = a.dot(&a.t());
    let ar = a.clone().ssqrt().unwrap();
    ar.clone().reversed_axes().assert_allclose_l2(&ar, 1e-7);
}

#[test]
fn ssqrt_symmetric_random_t() {
    let r_dist = RealNormal::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist);
    a = a.dot(&a.t()).reversed_axes();
    let ar = a.clone().ssqrt().unwrap();
    ar.clone().reversed_axes().assert_allclose_l2(&ar, 1e-7);
}

#[test]
fn ssqrt_sqrt_random() {
    let r_dist = RealNormal::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist);
    a = a.dot(&a.t());
    let ar = a.clone().ssqrt().unwrap();
    ar.clone().reversed_axes().assert_allclose_l2(&ar, 1e-7);
}

#[test]
fn ssqrt_sqrt_random_t() {
    let r_dist = RealNormal::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist);
    a = a.dot(&a.t()).reversed_axes();
    let ar = a.clone().ssqrt().unwrap();
    ar.clone().reversed_axes().assert_allclose_l2(&ar, 1e-7);
}
