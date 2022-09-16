use ndarray::*;
use ndarray_linalg::*;
use std::cmp::min;

fn test<T: Scalar + Lapack>(a: &Array2<T>) {
    let (n, m) = a.dim();
    let answer = a.clone();
    println!("a = \n{:?}", a);
    let (u, s, vt): (_, Array1<_>, _) = a.svd(true, true).unwrap();
    let u: Array2<_> = u.unwrap();
    let vt: Array2<_> = vt.unwrap();
    println!("u = \n{:?}", &u);
    println!("s = \n{:?}", &s);
    println!("v = \n{:?}", &vt);
    let mut sm = Array::<T, _>::zeros((n, m));
    for i in 0..min(n, m) {
        sm[(i, i)] = T::from(s[i]).unwrap();
    }
    assert_close_l2!(&u.dot(&sm).dot(&vt), &answer, T::real(1e-7));
}

fn test_no_vt<T: Scalar + Lapack>(a: &Array2<T>) {
    let (n, _m) = a.dim();
    println!("a = \n{:?}", a);
    let (u, _s, vt): (_, Array1<_>, _) = a.svd(true, false).unwrap();
    assert!(u.is_some());
    assert!(vt.is_none());
    let u = u.unwrap();
    assert_eq!(u.dim().0, n);
    assert_eq!(u.dim().1, n);
}

fn test_no_u<T: Scalar + Lapack>(a: &Array2<T>) {
    let (_n, m) = a.dim();
    println!("a = \n{:?}", a);
    let (u, _s, vt): (_, Array1<_>, _) = a.svd(false, true).unwrap();
    assert!(u.is_none());
    assert!(vt.is_some());
    let vt = vt.unwrap();
    assert_eq!(vt.dim().0, m);
    assert_eq!(vt.dim().1, m);
}

fn test_diag_only<T: Scalar + Lapack>(a: &Array2<T>) {
    println!("a = \n{:?}", a);
    let (u, _s, vt): (_, Array1<_>, _) = a.svd(false, false).unwrap();
    assert!(u.is_none());
    assert!(vt.is_none());
}

macro_rules! test_svd_impl {
    ($type:ty, $test:ident, $n:expr, $m:expr) => {
        paste::item! {
            #[test]
            fn [<svd_ $type _ $test _ $n x $m>]() {
                let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
                let a = random_using(($n, $m), &mut rng);
                $test::<$type>(&a);
            }

            #[test]
            fn [<svd_ $type _ $test _ $n x $m _t>]() {
                let mut rng = rand_pcg::Mcg128Xsl64::new(0xcafef00dd15ea5e5);
                let a = random_using(($n, $m).f(), &mut rng);
                $test::<$type>(&a);
            }
        }
    };
}

test_svd_impl!(f64, test, 3, 3);
test_svd_impl!(f64, test_no_vt, 3, 3);
test_svd_impl!(f64, test_no_u, 3, 3);
test_svd_impl!(f64, test_diag_only, 3, 3);
test_svd_impl!(f64, test, 4, 3);
test_svd_impl!(f64, test_no_vt, 4, 3);
test_svd_impl!(f64, test_no_u, 4, 3);
test_svd_impl!(f64, test_diag_only, 4, 3);
test_svd_impl!(f64, test, 3, 4);
test_svd_impl!(f64, test_no_vt, 3, 4);
test_svd_impl!(f64, test_no_u, 3, 4);
test_svd_impl!(f64, test_diag_only, 3, 4);
test_svd_impl!(c64, test, 3, 3);
test_svd_impl!(c64, test_no_vt, 3, 3);
test_svd_impl!(c64, test_no_u, 3, 3);
test_svd_impl!(c64, test_diag_only, 3, 3);
test_svd_impl!(c64, test, 4, 3);
test_svd_impl!(c64, test_no_vt, 4, 3);
test_svd_impl!(c64, test_no_u, 4, 3);
test_svd_impl!(c64, test_diag_only, 4, 3);
test_svd_impl!(c64, test, 3, 4);
test_svd_impl!(c64, test_no_vt, 3, 4);
test_svd_impl!(c64, test_no_u, 3, 4);
test_svd_impl!(c64, test_diag_only, 3, 4);
