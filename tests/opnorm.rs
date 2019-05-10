use ndarray::*;
use ndarray_linalg::*;
use num_traits::Float;

fn test(a: Array2<f64>, one: f64, inf: f64, fro: f64) {
    println!("ONE = {:?}", a.opnorm_one());
    println!("INF = {:?}", a.opnorm_inf());
    println!("FRO = {:?}", a.opnorm_fro());
    assert_rclose!(a.opnorm_one().unwrap(), one, 1e-7);
    assert_rclose!(a.opnorm_inf().unwrap(), inf, 1e-7);
    assert_rclose!(a.opnorm_fro().unwrap(), fro, 1e-7);
}

fn gen(i: usize, j: usize, rev: bool) -> Array2<f64> {
    let n = (i * j + 1) as f64;
    if rev {
        Array::range(1., n, 1.).into_shape((j, i)).unwrap().reversed_axes()
    } else {
        Array::range(1., n, 1.).into_shape((i, j)).unwrap()
    }
}

#[test]
fn opnorm_square() {
    test(gen(3, 3, false), 18.0, 24.0, 285.0.sqrt());
}

#[test]
fn opnorm_square_t() {
    test(gen(3, 3, true), 24.0, 18.0, 285.0.sqrt());
}

#[test]
fn opnorm_3x4() {
    test(gen(3, 4, false), 24.0, 42.0, 650.0.sqrt());
}

#[test]
fn opnorm_3x4_t() {
    test(gen(3, 4, true), 33.0, 30.0, 650.0.sqrt());
}

#[test]
fn opnorm_4x3() {
    test(gen(4, 3, false), 30.0, 33.0, 650.0.sqrt());
}

#[test]
fn opnorm_4x3_t() {
    test(gen(4, 3, true), 42.0, 24.0, 650.0.sqrt());
}
