include!("header.rs");

#[test]
fn cholesky() {
    let r_dist = RealNormal::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist);
    a = a.dot(&a.t());
    println!("a = \n{:?}", a);
    let c = a.clone().cholesky().unwrap();
    println!("c = \n{:?}", c);
    println!("cc = \n{:?}", c.t().dot(&c));
    c.t().dot(&c).assert_allclose_l2(&a, 1e-7);
}

#[test]
fn cholesky_t() {
    let r_dist = RealNormal::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist);
    a = a.dot(&a.t()).reversed_axes();
    println!("a = \n{:?}", a);
    let c = a.clone().cholesky().unwrap();
    println!("c = \n{:?}", c);
    println!("cc = \n{:?}", c.t().dot(&c));
    c.t().dot(&c).assert_allclose_l2(&a, 1e-7);
}
