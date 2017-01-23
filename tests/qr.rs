include!("header.rs");

#[test]
fn qr_square_upper() {
    let r_dist = RealNormal::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist);
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    q.clone().assert_allclose_l2(&Array::eye(3), 1e-7);
    q.dot(&q.t()).assert_allclose_l2(&Array::eye(3), 1e-7);
    r.assert_allclose_l2(&a, 1e-7);
}

#[test]
fn qr_square_upper_t() {
    let r_dist = RealNormal::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 3), r_dist).reversed_axes();
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    q.clone().assert_allclose_l2(&Array::eye(3), 1e-7);
    q.dot(&q.t()).assert_allclose_l2(&Array::eye(3), 1e-7);
    r.assert_allclose_l2(&a, 1e-7);
}

#[test]
fn qr_square() {
    let r_dist = RealNormal::new(0., 1.);
    let a = Array::<f64, _>::random((3, 3), r_dist);
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    q.dot(&q.t()).assert_allclose_l2(&Array::eye(3), 1e-7);
    q.dot(&r).assert_allclose_l2(&a, 1e-7);
}

#[test]
fn qr_square_t() {
    let r_dist = RealNormal::new(0., 1.);
    let a = Array::<f64, _>::random((3, 3), r_dist).reversed_axes();
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    q.dot(&q.t()).assert_allclose_l2(&Array::eye(3), 1e-7);
    q.dot(&r).assert_allclose_l2(&a, 1e-7);
}

#[test]
fn qr_3x4_upper() {
    let r_dist = RealNormal::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 4), r_dist);
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    q.clone().assert_allclose_l2(&Array::eye(3), 1e-7);
    q.dot(&q.t()).assert_allclose_l2(&Array::eye(3), 1e-7);
    q.dot(&r).assert_allclose_l2(&a, 1e-7);
}

#[test]
fn qr_3x4_upper_t() {
    let r_dist = RealNormal::new(0., 1.);
    let mut a = Array::<f64, _>::random((4, 3), r_dist).reversed_axes();
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    q.clone().assert_allclose_l2(&Array::eye(3), 1e-7);
    q.dot(&q.t()).assert_allclose_l2(&Array::eye(3), 1e-7);
    q.dot(&r).assert_allclose_l2(&a, 1e-7);
}

#[test]
fn qr_3x4() {
    let r_dist = RealNormal::new(0., 1.);
    let a = Array::<f64, _>::random((3, 4), r_dist);
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    q.dot(&q.t()).assert_allclose_l2(&Array::eye(3), 1e-7);
    q.dot(&r).assert_allclose_l2(&a, 1e-7);
}

#[test]
fn qr_3x4_t() {
    let r_dist = RealNormal::new(0., 1.);
    let a = Array::<f64, _>::random((4, 3), r_dist).reversed_axes();
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    q.dot(&q.t()).assert_allclose_l2(&Array::eye(3), 1e-7);
    q.dot(&r).assert_allclose_l2(&a, 1e-7);
}

#[test]
fn qr_4x3_upper() {
    let r_dist = RealNormal::new(0., 1.);
    let mut a = Array::<f64, _>::random((4, 3), r_dist);
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    q.t().dot(&q).assert_allclose_l2(&Array::eye(3), 1e-7);
    q.dot(&r).assert_allclose_l2(&a, 1e-7);
}

#[test]
fn qr_4x3_upper_t() {
    let r_dist = RealNormal::new(0., 1.);
    let mut a = Array::<f64, _>::random((3, 4), r_dist).reversed_axes();
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    q.t().dot(&q).assert_allclose_l2(&Array::eye(3), 1e-7);
    q.dot(&r).assert_allclose_l2(&a, 1e-7);
}

#[test]
fn qr_4x3() {
    let r_dist = RealNormal::new(0., 1.);
    let a = Array::<f64, _>::random((4, 3), r_dist);
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    q.t().dot(&q).assert_allclose_l2(&Array::eye(3), 1e-7);
    q.dot(&r).assert_allclose_l2(&a, 1e-7);
}

#[test]
fn qr_4x3_t() {
    let r_dist = RealNormal::new(0., 1.);
    let a = Array::<f64, _>::random((3, 4), r_dist).reversed_axes();
    println!("a = \n{:?}", &a);
    let (q, r) = a.clone().qr().unwrap();
    println!("q = \n{:?}", &q);
    println!("r = \n{:?}", &r);
    q.t().dot(&q).assert_allclose_l2(&Array::eye(3), 1e-7);
    q.dot(&r).assert_allclose_l2(&a, 1e-7);
}
