use ndarray::prelude::*;
use ndarray_linalg::{
    assert_aclose, assert_close_l2, c32, c64, random_hpd_using, random_using, solve::*,
    OperationNorm, Scalar,
};

macro_rules! test_solve {
    (
        [$($elem_type:ty => $rtol:expr),*],
        $a_ident:ident = $a:expr,
        $x_ident:ident = $x:expr,
        b = $b:expr,
        $solve:ident,
    ) => {
        $({
            let $a_ident: Array2<$elem_type> = $a;
            let $x_ident: Array1<$elem_type> = $x;
            let b: Array1<$elem_type> = $b;
            let a = $a_ident;
            let x = $x_ident;
            let rtol = $rtol;
            assert_close_l2!(&a.$solve(&b).unwrap(), &x, rtol);
            assert_close_l2!(&a.factorize().unwrap().$solve(&b).unwrap(), &x, rtol);
            assert_close_l2!(&a.factorize_into().unwrap().$solve(&b).unwrap(), &x, rtol);
        })*
    };
}

macro_rules! test_solve_into {
    (
        [$($elem_type:ty => $rtol:expr),*],
        $a_ident:ident = $a:expr,
        $x_ident:ident = $x:expr,
        b = $b:expr,
        $solve_into:ident,
    ) => {
        $({
            let $a_ident: Array2<$elem_type> = $a;
            let $x_ident: Array1<$elem_type> = $x;
            let b: Array1<$elem_type> = $b;
            let a = $a_ident;
            let x = $x_ident;
            let rtol = $rtol;
            assert_close_l2!(&a.$solve_into(b.clone()).unwrap(), &x, rtol);
            assert_close_l2!(&a.factorize().unwrap().$solve_into(b.clone()).unwrap(), &x, rtol);
            assert_close_l2!(&a.factorize_into().unwrap().$solve_into(b.clone()).unwrap(), &x, rtol);
        })*
    };
}

macro_rules! test_solve_inplace {
    (
        [$($elem_type:ty => $rtol:expr),*],
        $a_ident:ident = $a:expr,
        $x_ident:ident = $x:expr,
        b = $b:expr,
        $solve_inplace:ident,
    ) => {
        $({
            let $a_ident: Array2<$elem_type> = $a;
            let $x_ident: Array1<$elem_type> = $x;
            let b: Array1<$elem_type> = $b;
            let a = $a_ident;
            let x = $x_ident;
            let rtol = $rtol;
            {
                let mut b = b.clone();
                assert_close_l2!(&a.$solve_inplace(&mut b).unwrap(), &x, rtol);
                assert_close_l2!(&b, &x, rtol);
            }
            {
                let mut b = b.clone();
                assert_close_l2!(&a.factorize().unwrap().$solve_inplace(&mut b).unwrap(), &x, rtol);
                assert_close_l2!(&b, &x, rtol);
            }
            {
                let mut b = b.clone();
                assert_close_l2!(&a.factorize_into().unwrap().$solve_inplace(&mut b).unwrap(), &x, rtol);
                assert_close_l2!(&b, &x, rtol);
            }
        })*
    };
}

macro_rules! test_solve_all {
    (
        [$($elem_type:ty => $rtol:expr),*],
        $a_ident:ident = $a:expr,
        $x_ident:ident = $x:expr,
        b = $b:expr,
        [$solve:ident, $solve_into:ident, $solve_inplace:ident],
    ) => {
        test_solve!([$($elem_type => $rtol),*], $a_ident = $a, $x_ident = $x, b = $b, $solve,);
        test_solve_into!([$($elem_type => $rtol),*], $a_ident = $a, $x_ident = $x, b = $b, $solve_into,);
        test_solve_inplace!([$($elem_type => $rtol),*], $a_ident = $a, $x_ident = $x, b = $b, $solve_inplace,);
    };
}

#[test]
fn solve_random_float() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    for n in 0..=8 {
        for &set_f in &[false, true] {
            test_solve_all!(
                [f32 => 1e-3, f64 => 1e-9],
                a = random_using([n; 2].set_f(set_f), &mut rng),
                x = random_using(n, &mut rng),
                b = a.dot(&x),
                [solve, solve_into, solve_inplace],
            );
        }
    }
}

#[test]
fn solve_random_complex() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    for n in 0..=8 {
        for &set_f in &[false, true] {
            test_solve_all!(
                [c32 => 1e-3, c64 => 1e-9],
                a = random_using([n; 2].set_f(set_f), &mut rng),
                x = random_using(n, &mut rng),
                b = a.dot(&x),
                [solve, solve_into, solve_inplace],
            );
        }
    }
}

#[should_panic]
#[test]
fn solve_shape_mismatch() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a: Array2<f64> = random_using((3, 3), &mut rng);
    let b: Array1<f64> = random_using(2, &mut rng);
    let _ = a.solve_into(b);
}

#[test]
fn solve_t_random_float() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    for n in 0..=8 {
        for &set_f in &[false, true] {
            test_solve_all!(
                [f32 => 1e-3, f64 => 1e-9],
                a = random_using([n; 2].set_f(set_f), &mut rng),
                x = random_using(n, &mut rng),
                b = a.t().dot(&x),
                [solve_t, solve_t_into, solve_t_inplace],
            );
        }
    }
}

#[should_panic]
#[test]
fn solve_t_shape_mismatch() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a: Array2<f64> = random_using((3, 3).f(), &mut rng);
    let b: Array1<f64> = random_using(4, &mut rng);
    let _ = a.solve_into(b);
}

#[test]
fn solve_t_random_complex() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    for n in 0..=8 {
        for &set_f in &[false, true] {
            test_solve_all!(
                [c32 => 1e-3, c64 => 1e-9],
                a = random_using([n; 2].set_f(set_f), &mut rng),
                x = random_using(n, &mut rng),
                b = a.t().dot(&x),
                [solve_t, solve_t_into, solve_t_inplace],
            );
        }
    }
}

#[should_panic]
#[test]
fn solve_factorized_shape_mismatch() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a: Array2<f64> = random_using((3, 3), &mut rng);
    let b: Array1<f64> = random_using(4, &mut rng);
    let f = a.factorize_into().unwrap();
    let _ = f.solve_into(b);
}

#[test]
fn solve_h_random_float() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    for n in 0..=8 {
        for &set_f in &[false, true] {
            test_solve_all!(
                [f32 => 1e-3, f64 => 1e-9],
                a = random_using([n; 2].set_f(set_f), &mut rng),
                x = random_using(n, &mut rng),
                b = a.t().mapv(|x| x.conj()).dot(&x),
                [solve_h, solve_h_into, solve_h_inplace],
            );
        }
    }
}

#[should_panic]
#[test]
fn solve_factorized_t_shape_mismatch() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a: Array2<f64> = random_using((3, 3).f(), &mut rng);
    let b: Array1<f64> = random_using(4, &mut rng);
    let f = a.factorize_into().unwrap();
    let _ = f.solve_into(b);
}

#[test]
fn solve_h_random_complex() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    for n in 0..=8 {
        for &set_f in &[false, true] {
            test_solve_all!(
                [c32 => 1e-3, c64 => 1e-9],
                a = random_using([n; 2].set_f(set_f), &mut rng),
                x = random_using(n, &mut rng),
                b = a.t().mapv(|x| x.conj()).dot(&x),
                [solve_h, solve_h_into, solve_h_inplace],
            );
        }
    }
}

#[test]
fn rcond() {
    macro_rules! rcond {
        ($elem:ty, $rows:expr, $atol:expr) => {
            let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
            let a: Array2<$elem> = random_hpd_using($rows, &mut rng);
            let rcond = 1. / (a.opnorm_one().unwrap() * a.inv().unwrap().opnorm_one().unwrap());
            assert_aclose!(a.rcond().unwrap(), rcond, $atol);
            assert_aclose!(a.rcond_into().unwrap(), rcond, $atol);
        };
    }
    for rows in 1..6 {
        rcond!(f64, rows, 0.2);
        rcond!(f32, rows, 0.5);
        rcond!(c64, rows, 0.2);
        rcond!(c32, rows, 0.5);
    }
}

#[test]
fn rcond_hilbert() {
    macro_rules! rcond_hilbert {
        ($elem:ty, $rows:expr, $atol:expr) => {
            let a = Array2::<$elem>::from_shape_fn(($rows, $rows), |(i, j)| {
                1. / (i as $elem + j as $elem + 1.)
            });
            assert_aclose!(a.rcond().unwrap(), 0., $atol);
            assert_aclose!(a.rcond_into().unwrap(), 0., $atol);
        };
    }
    rcond_hilbert!(f64, 10, 1e-9);
    rcond_hilbert!(f32, 10, 1e-3);
}

#[test]
fn rcond_identity() {
    macro_rules! rcond_identity {
        ($elem:ty, $rows:expr, $atol:expr) => {
            let a = Array2::<$elem>::eye($rows);
            assert_aclose!(a.rcond().unwrap(), 1., $atol);
            assert_aclose!(a.rcond_into().unwrap(), 1., $atol);
        };
    }
    for rows in 1..6 {
        rcond_identity!(f64, rows, 1e-9);
        rcond_identity!(f32, rows, 1e-3);
        rcond_identity!(c64, rows, 1e-9);
        rcond_identity!(c32, rows, 1e-3);
    }
}
