include!("header.rs");

#[test]
fn solve_upper() {
    let r_dist = RealNormal::new(0.0, 1.0);
    let a = drop_lower(Array::<f64, _>::random((3, 3), r_dist));
    println!("a = \n{:?}", &a);
    let b = Array::<f64, _>::random(3, r_dist);
    println!("b = \n{:?}", &b);
    let x = a.solve_upper(b.clone()).unwrap();
    println!("x = \n{:?}", &x);
    println!("Ax = \n{:?}", a.dot(&x));
    all_close_l2(&a.dot(&x), &b, 1e-7).unwrap();
}

#[test]
fn solve_upper_t() {
    let r_dist = RealNormal::new(0., 1.);
    let a = drop_lower(Array::<f64, _>::random((3, 3), r_dist).reversed_axes());
    println!("a = \n{:?}", &a);
    let b = Array::<f64, _>::random(3, r_dist);
    println!("b = \n{:?}", &b);
    let x = a.solve_upper(b.clone()).unwrap();
    println!("x = \n{:?}", &x);
    println!("Ax = \n{:?}", a.dot(&x));
    all_close_l2(&a.dot(&x), &b, 1e-7).unwrap();
}

#[test]
fn solve_lower() {
    let r_dist = RealNormal::new(0., 1.);
    let a = drop_upper(Array::<f64, _>::random((3, 3), r_dist));
    println!("a = \n{:?}", &a);
    let b = Array::<f64, _>::random(3, r_dist);
    println!("b = \n{:?}", &b);
    let x = a.solve_lower(b.clone()).unwrap();
    println!("x = \n{:?}", &x);
    println!("Ax = \n{:?}", a.dot(&x));
    all_close_l2(&a.dot(&x), &b, 1e-7).unwrap();
}

#[test]
fn solve_lower_t() {
    let r_dist = RealNormal::new(0., 1.);
    let a = drop_upper(Array::<f64, _>::random((3, 3), r_dist).reversed_axes());
    println!("a = \n{:?}", &a);
    let b = Array::<f64, _>::random(3, r_dist);
    println!("b = \n{:?}", &b);
    let x = a.solve_lower(b.clone()).unwrap();
    println!("x = \n{:?}", &x);
    println!("Ax = \n{:?}", a.dot(&x));
    all_close_l2(&a.dot(&x), &b, 1e-7).unwrap();
}
