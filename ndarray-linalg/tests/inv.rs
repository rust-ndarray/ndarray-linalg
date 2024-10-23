use ndarray::*;
use ndarray_linalg::*;

fn test_inv_random<A>(n: usize, set_f: bool, rtol: A::Real)
where
    A: Scalar + Lapack,
{
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a: Array2<A> = random_using([n; 2].set_f(set_f), &mut rng);
    let identity = Array2::eye(n);
    assert_close_l2!(&a.inv().unwrap().dot(&a), &identity, rtol);
    assert_close_l2!(
        &a.factorize().unwrap().inv().unwrap().dot(&a),
        &identity,
        rtol
    );
    assert_close_l2!(
        &a.clone().factorize_into().unwrap().inv().unwrap().dot(&a),
        &identity,
        rtol
    );
}

fn test_inv_into_random<A>(n: usize, set_f: bool, rtol: A::Real)
where
    A: Scalar + Lapack,
{
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a: Array2<A> = random_using([n; 2].set_f(set_f), &mut rng);
    let identity = Array2::eye(n);
    assert_close_l2!(&a.clone().inv_into().unwrap().dot(&a), &identity, rtol);
    assert_close_l2!(
        &a.factorize().unwrap().inv_into().unwrap().dot(&a),
        &identity,
        rtol
    );
    assert_close_l2!(
        &a.clone()
            .factorize_into()
            .unwrap()
            .inv_into()
            .unwrap()
            .dot(&a),
        &identity,
        rtol
    );
}

#[test]
fn inv_empty() {
    test_inv_random::<f32>(0, false, 0.);
    test_inv_random::<f64>(0, false, 0.);
    test_inv_random::<c32>(0, false, 0.);
    test_inv_random::<c64>(0, false, 0.);
}

#[test]
fn inv_random_float() {
    for n in 1..=8 {
        for &set_f in &[false, true] {
            test_inv_random::<f32>(n, set_f, 1e-3);
            test_inv_random::<f64>(n, set_f, 1e-9);
        }
    }
}

#[test]
fn inv_random_complex() {
    for n in 1..=8 {
        for &set_f in &[false, true] {
            test_inv_random::<c32>(n, set_f, 1e-3);
            test_inv_random::<c64>(n, set_f, 1e-9);
        }
    }
}

#[test]
fn inv_into_empty() {
    test_inv_into_random::<f32>(0, false, 0.);
    test_inv_into_random::<f64>(0, false, 0.);
    test_inv_into_random::<c32>(0, false, 0.);
    test_inv_into_random::<c64>(0, false, 0.);
}

#[test]
fn inv_into_random_float() {
    for n in 1..=8 {
        for &set_f in &[false, true] {
            test_inv_into_random::<f32>(n, set_f, 1e-3);
            test_inv_into_random::<f64>(n, set_f, 1e-9);
        }
    }
}

#[test]
fn inv_into_random_complex() {
    for n in 1..=8 {
        for &set_f in &[false, true] {
            test_inv_into_random::<c32>(n, set_f, 1e-3);
            test_inv_into_random::<c64>(n, set_f, 1e-9);
        }
    }
}

#[test]
#[should_panic]
fn inv_error() {
    // do not have inverse
    let a = Array::<f64, _>::zeros((3, 3));
    let a_inv = a.inv().unwrap();
    println!("{:?}", a_inv);
}

#[test]
fn inv_2x2() {
    // Related to issue #123 where this problem led to a wrongly computed inverse when using the
    // `openblas` backend.
    let a: Array2<f64> = array!([1.0, 2.0], [3.0, 4.0]);
    let a_inv = a.inv().unwrap();
    assert_close_l2!(&a_inv, &array!([-2.0, 1.0], [1.5, -0.5]), 1e-7);
}
