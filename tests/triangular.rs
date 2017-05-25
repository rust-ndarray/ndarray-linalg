include!("header.rs");

macro_rules! impl_test {
    ($modname:ident, $random:path) => {
mod $modname {
    use ndarray::prelude::*;
    use ndarray_linalg::prelude::*;
    use ndarray_rand::RandomExt;
    use rand_extra::*;
    #[test]
    fn solve_upper() {
        let r_dist = RealNormal::new(0.0, 1.0);
        let a = drop_lower($random((3, 3), r_dist));
        println!("a = \n{:?}", &a);
        let b = $random(3, r_dist);
        println!("b = \n{:?}", &b);
        let x = a.solve_upper(b.clone()).unwrap();
        println!("x = \n{:?}", &x);
        println!("Ax = \n{:?}", a.dot(&x));
        all_close_l2(&a.dot(&x), &b, 1e-7).unwrap();
    }

    #[test]
    fn solve_upper_t() {
        let r_dist = RealNormal::new(0., 1.);
        let a = drop_lower($random((3, 3), r_dist).reversed_axes());
        println!("a = \n{:?}", &a);
        let b = $random(3, r_dist);
        println!("b = \n{:?}", &b);
        let x = a.solve_upper(b.clone()).unwrap();
        println!("x = \n{:?}", &x);
        println!("Ax = \n{:?}", a.dot(&x));
        all_close_l2(&a.dot(&x), &b, 1e-7).unwrap();
    }

    #[test]
    fn solve_lower() {
        let r_dist = RealNormal::new(0., 1.);
        let a = drop_upper($random((3, 3), r_dist));
        println!("a = \n{:?}", &a);
        let b = $random(3, r_dist);
        println!("b = \n{:?}", &b);
        let x = a.solve_lower(b.clone()).unwrap();
        println!("x = \n{:?}", &x);
        println!("Ax = \n{:?}", a.dot(&x));
        all_close_l2(&a.dot(&x), &b, 1e-7).unwrap();
    }

    #[test]
    fn solve_lower_t() {
        let r_dist = RealNormal::new(0., 1.);
        let a = drop_upper($random((3, 3), r_dist).reversed_axes());
        println!("a = \n{:?}", &a);
        let b = $random(3, r_dist);
        println!("b = \n{:?}", &b);
        let x = a.solve_lower(b.clone()).unwrap();
        println!("x = \n{:?}", &x);
        println!("Ax = \n{:?}", a.dot(&x));
        all_close_l2(&a.dot(&x), &b, 1e-7).unwrap();
    }
}
}} // impl_test_opnorm

impl_test!(owned, Array<f64, _>::random);
impl_test!(shared, RcArray<f64, _>::random);

macro_rules! impl_test_2d {
    ($modname:ident, $drop:path, $solve:ident) => {
mod $modname {
    use super::random_owned;
    use ndarray_linalg::prelude::*;
    #[test]
    fn solve_tt() {
        let a = $drop(random_owned(3, 3, true));
        println!("a = \n{:?}", &a);
        let b = random_owned(3, 2, true);
        println!("b = \n{:?}", &b);
        let x = a.$solve(&b).unwrap();
        println!("x = \n{:?}", &x);
        println!("Ax = \n{:?}", a.dot(&x));
        all_close_l2(&a.dot(&x), &b, 1e-7).unwrap();
    }
    #[test]
    fn solve_tf() {
        let a = $drop(random_owned(3, 3, true));
        println!("a = \n{:?}", &a);
        let b = random_owned(3, 2, false);
        println!("b = \n{:?}", &b);
        let x = a.$solve(&b).unwrap();
        println!("x = \n{:?}", &x);
        println!("Ax = \n{:?}", a.dot(&x));
        all_close_l2(&a.dot(&x), &b, 1e-7).unwrap();
    }
    #[test]
    fn solve_ft() {
        let a = $drop(random_owned(3, 3, false));
        println!("a = \n{:?}", &a);
        let b = random_owned(3, 2, true);
        println!("b = \n{:?}", &b);
        let x = a.$solve(&b).unwrap();
        println!("x = \n{:?}", &x);
        println!("Ax = \n{:?}", a.dot(&x));
        all_close_l2(&a.dot(&x), &b, 1e-7).unwrap();
    }
    #[test]
    fn solve_ff() {
        let a = $drop(random_owned(3, 3, false));
        println!("a = \n{:?}", &a);
        let b = random_owned(3, 2, false);
        println!("b = \n{:?}", &b);
        let x = a.$solve(&b).unwrap();
        println!("x = \n{:?}", &x);
        println!("Ax = \n{:?}", a.dot(&x));
        all_close_l2(&a.dot(&x), &b, 1e-7).unwrap();
    }
}
}} // impl_test_2d

impl_test_2d!(lower2d, drop_upper, solve_lower);
impl_test_2d!(upper2d, drop_lower, solve_upper);
