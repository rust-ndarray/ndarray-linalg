use ndarray::*;
use ndarray_linalg::*;

#[should_panic]
#[test]
fn eigh_generalized_shape_mismatch() {
    let a = Array2::<f64>::eye(3);
    let b = Array2::<f64>::eye(2);
    let _ = (a, b).eigh_inplace(UPLO::Upper);
}

#[test]
fn fixed() {
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
    let (e, vecs): (Array1<_>, Array2<_>) = (&a).eigh(UPLO::Upper).unwrap();
    assert_close_l2!(&e, &arr1(&[2.0, 2.0, 5.0]), 1.0e-7);

    // Check eigenvectors are orthogonalized
    let s = vecs.t().dot(&vecs);
    assert_close_l2!(&s, &Array::eye(3), 1.0e-7);

    for (i, v) in vecs.axis_iter(Axis(1)).enumerate() {
        let av = a.dot(&v);
        let ev = v.mapv(|x| e[i] * x);
        assert_close_l2!(&av, &ev, 1.0e-7);
    }
}

#[test]
fn fixed_t() {
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]).reversed_axes();
    let (e, vecs): (Array1<_>, Array2<_>) = (&a).eigh(UPLO::Upper).unwrap();
    assert_close_l2!(&e, &arr1(&[2.0, 2.0, 5.0]), 1.0e-7);

    // Check eigenvectors are orthogonalized
    let s = vecs.t().dot(&vecs);
    assert_close_l2!(&s, &Array::eye(3), 1.0e-7);

    for (i, v) in vecs.axis_iter(Axis(1)).enumerate() {
        let av = a.dot(&v);
        let ev = v.mapv(|x| e[i] * x);
        assert_close_l2!(&av, &ev, 1.0e-7);
    }
}

#[test]
fn fixed_lower() {
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]);
    let (e, vecs): (Array1<_>, Array2<_>) = (&a).eigh(UPLO::Lower).unwrap();
    assert_close_l2!(&e, &arr1(&[2.0, 2.0, 5.0]), 1.0e-7);

    // Check eigenvectors are orthogonalized
    let s = vecs.t().dot(&vecs);
    assert_close_l2!(&s, &Array::eye(3), 1.0e-7);

    for (i, v) in vecs.axis_iter(Axis(1)).enumerate() {
        let av = a.dot(&v);
        let ev = v.mapv(|x| e[i] * x);
        assert_close_l2!(&av, &ev, 1.0e-7);
    }
}

#[test]
fn fixed_t_lower() {
    let a = arr2(&[[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]).reversed_axes();
    let (e, vecs): (Array1<_>, Array2<_>) = (&a).eigh(UPLO::Lower).unwrap();
    assert_close_l2!(&e, &arr1(&[2.0, 2.0, 5.0]), 1.0e-7);

    // Check eigenvectors are orthogonalized
    let s = vecs.t().dot(&vecs);
    assert_close_l2!(&s, &Array::eye(3), 1.0e-7);

    for (i, v) in vecs.axis_iter(Axis(1)).enumerate() {
        let av = a.dot(&v);
        let ev = v.mapv(|x| e[i] * x);
        assert_close_l2!(&av, &ev, 1.0e-7);
    }
}

#[test]
fn ssqrt() {
    let a: Array2<f64> = random_hpd(3);
    let ans = a.clone();
    let s = a.ssqrt(UPLO::Upper).unwrap();
    println!("a = {:?}", &ans);
    println!("s = {:?}", &s);
    assert_close_l2!(&s.t(), &s, 1e-7);
    let ss = s.dot(&s);
    println!("ss = {:?}", &ss);
    assert_close_l2!(&ss, &ans, 1e-7);
}

#[test]
fn ssqrt_t() {
    let a: Array2<f64> = random_hpd(3).reversed_axes();
    let ans = a.clone();
    let s = a.ssqrt(UPLO::Upper).unwrap();
    println!("a = {:?}", &ans);
    println!("s = {:?}", &s);
    assert_close_l2!(&s.t(), &s, 1e-7);
    let ss = s.dot(&s);
    println!("ss = {:?}", &ss);
    assert_close_l2!(&ss, &ans, 1e-7);
}

#[test]
fn ssqrt_lower() {
    let a: Array2<f64> = random_hpd(3);
    let ans = a.clone();
    let s = a.ssqrt(UPLO::Lower).unwrap();
    println!("a = {:?}", &ans);
    println!("s = {:?}", &s);
    assert_close_l2!(&s.t(), &s, 1e-7);
    let ss = s.dot(&s);
    println!("ss = {:?}", &ss);
    assert_close_l2!(&ss, &ans, 1e-7);
}

#[test]
fn ssqrt_t_lower() {
    let a: Array2<f64> = random_hpd(3).reversed_axes();
    let ans = a.clone();
    let s = a.ssqrt(UPLO::Lower).unwrap();
    println!("a = {:?}", &ans);
    println!("s = {:?}", &s);
    assert_close_l2!(&s.t(), &s, 1e-7);
    let ss = s.dot(&s);
    println!("ss = {:?}", &ss);
    assert_close_l2!(&ss, &ans, 1e-7);
}
