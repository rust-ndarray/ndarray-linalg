include!("header.rs");

#[test]
fn trace() {
    let r_dist = RealNormal::new(0., 1.);
    let a = Array::<f64, _>::random((3, 3), r_dist);
    assert_rclose!(a.trace().unwrap(), a[(0, 0)] + a[(1, 1)] + a[(2, 2)], 1e-7);
}
