use ndarray::*;
use ndarray_linalg::*;

#[should_panic]
#[test]
fn solve_shape_mismatch() {
    let a: Array2<f64> = random((3, 3));
    let b: Array1<f64> = random(2);
    let _ = a.solve_into(b);
}

#[test]
fn solve_random() {
    let a: Array2<f64> = random((3, 3));
    let x: Array1<f64> = random(3);
    let b = a.dot(&x);
    let y = a.solve_into(b).unwrap();
    assert_close_l2!(&x, &y, 1e-7);
}

#[should_panic]
#[test]
fn solve_t_shape_mismatch() {
    let a: Array2<f64> = random((3, 3).f());
    let b: Array1<f64> = random(4);
    let _ = a.solve_into(b);
}

#[test]
fn solve_random_t() {
    let a: Array2<f64> = random((3, 3).f());
    let x: Array1<f64> = random(3);
    let b = a.dot(&x);
    let y = a.solve_into(b).unwrap();
    assert_close_l2!(&x, &y, 1e-7);
}

#[should_panic]
#[test]
fn solve_factorized_shape_mismatch() {
    let a: Array2<f64> = random((3, 3));
    let b: Array1<f64> = random(4);
    let f = a.factorize_into().unwrap();
    let _ = f.solve_into(b);
}

#[test]
fn solve_factorized() {
    let a: Array2<f64> = random((3, 3));
    let ans: Array1<f64> = random(3);
    let b = a.dot(&ans);
    let f = a.factorize_into().unwrap();
    let x = f.solve_into(b).unwrap();
    assert_close_l2!(&x, &ans, 1e-7);
}

#[should_panic]
#[test]
fn solve_factorized_t_shape_mismatch() {
    let a: Array2<f64> = random((3, 3).f());
    let b: Array1<f64> = random(4);
    let f = a.factorize_into().unwrap();
    let _ = f.solve_into(b);
}

#[test]
fn solve_factorized_t() {
    let a: Array2<f64> = random((3, 3).f());
    let ans: Array1<f64> = random(3);
    let b = a.dot(&ans);
    let f = a.factorize_into().unwrap();
    let x = f.solve_into(b).unwrap();
    assert_close_l2!(&x, &ans, 1e-7);
}

#[test]
fn rcond() {
    macro_rules! rcond {
        ($elem:ty, $rows:expr, $atol:expr) => {
            let a: Array2<$elem> = random_hpd($rows);
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
                1. / (i as $elem + j as $elem - 1.)
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
