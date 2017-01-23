include!("header.rs");

fn random_hermite(n: usize) -> Array<f64, Ix2> {
    let r_dist = RealNormal::new(0., 1.);
    let a = Array::<f64, _>::random((n, n), r_dist);
    a.dot(&a.t())
}

#[test]
fn deth() {
    let a = random_hermite(3);
    let (e, _) = a.clone().eigh().unwrap();
    let deth = a.clone().deth().unwrap();
    let det_eig = e.iter().fold(1.0, |x, y| x * y);
    deth.assert_close(det_eig, 1.0e-7);
}
