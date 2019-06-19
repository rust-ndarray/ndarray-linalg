use ndarray::*;
use ndarray_linalg::*;
use std::cmp::min;

fn test(a: &Array2<f64>) {
    let (n, m) = a.dim();
    let answer = a.clone();
    println!("a = \n{:?}", a);
    let (u, s, vt): (_, Array1<_>, _) = a.svd(true, true).unwrap();
    let u: Array2<_> = u.unwrap();
    let vt: Array2<_> = vt.unwrap();
    println!("u = \n{:?}", &u);
    println!("s = \n{:?}", &s);
    println!("v = \n{:?}", &vt);
    let mut sm = Array::zeros((n, m));
    for i in 0..min(n, m) {
        sm[(i, i)] = s[i];
    }
    assert_close_l2!(&u.dot(&sm).dot(&vt), &answer, 1e-7);
}

fn test_no_vt(a: &Array2<f64>) {
    let (n, _m) = a.dim();
    println!("a = \n{:?}", a);
    let (u, _s, vt): (_, Array1<_>, _) = a.svd(true, false).unwrap();
    assert!(u.is_some());
    assert!(vt.is_none());
    let u = u.unwrap();
    assert_eq!(u.dim().0, n);
    assert_eq!(u.dim().1, n);
}

fn test_no_u(a: &Array2<f64>) {
    let (_n, m) = a.dim();
    println!("a = \n{:?}", a);
    let (u, _s, vt): (_, Array1<_>, _) = a.svd(false, true).unwrap();
    assert!(u.is_none());
    assert!(vt.is_some());
    let vt = vt.unwrap();
    assert_eq!(vt.dim().0, m);
    assert_eq!(vt.dim().1, m);
}

macro_rules! test_svd_impl {
    ($test:ident, $n:expr, $m:expr) => {
        paste::item! {
            #[test]
            fn [<svd_ $test _ $n x $m>]() {
                let a = random(($n, $m));
                $test(&a);
            }

            #[test]
            fn [<svd_ $test _ $n x $m _t>]() {
                let a = random(($n, $m).f());
                $test(&a);
            }
        }
    };
}

test_svd_impl!(test, 3, 3);
test_svd_impl!(test_no_vt, 3, 3);
test_svd_impl!(test_no_u, 3, 3);
test_svd_impl!(test, 4, 3);
test_svd_impl!(test_no_vt, 4, 3);
test_svd_impl!(test_no_u, 4, 3);
test_svd_impl!(test, 3, 4);
test_svd_impl!(test_no_vt, 3, 4);
test_svd_impl!(test_no_u, 3, 4);
