use ndarray::*;
use ndarray_linalg::*;

#[should_panic]
#[test]
fn size_shoter() {
    let a: Array1<f32> = Array::zeros(3);
    let b = Array::zeros(4);
    a.inner(&b);
}

#[should_panic]
#[test]
fn size_longer() {
    let a: Array1<f32> = Array::zeros(3);
    let b = Array::zeros(4);
    b.inner(&a);
}

#[test]
fn abs() {
    let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
    let a: Array1<c32> = random_using(1, &mut rng);
    let aa = a.inner(&a);
    assert_aclose!(aa.re(), a.norm().powi(2), 1e-5);
    assert_aclose!(aa.im(), 0.0, 1e-5);
}
