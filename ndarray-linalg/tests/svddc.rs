use ndarray::*;
use ndarray_linalg::*;

fn test<T: Scalar + Lapack>(a: &Array2<T>, flag: UVTFlag) {
    let (n, m) = a.dim();
    let k = n.min(m);
    let answer = a.clone();
    println!("a = \n{:?}", a);
    let (u, s, vt): (_, Array1<_>, _) = a.svddc(flag).unwrap();
    let mut sm: Array2<T> = match flag {
        UVTFlag::Full => Array::zeros((n, m)),
        UVTFlag::Some => Array::zeros((k, k)),
        UVTFlag::None => {
            assert!(u.is_none());
            assert!(vt.is_none());
            return;
        }
    };
    let u: Array2<_> = u.unwrap();
    let vt: Array2<_> = vt.unwrap();
    println!("u = \n{:?}", &u);
    println!("s = \n{:?}", &s);
    println!("v = \n{:?}", &vt);
    for i in 0..k {
        sm[(i, i)] = T::from_real(s[i]);
    }
    assert_close_l2!(&u.dot(&sm).dot(&vt), &answer, T::real(1e-7));
}

macro_rules! test_svd_impl {
    ($scalar:ty, $n:expr, $m:expr) => {
        paste::item! {
            #[test]
            fn [<svddc_ $scalar _full_ $n x $m>]() {
                let a = random(($n, $m));
                test::<$scalar>(&a, UVTFlag::Full);
            }

            #[test]
            fn [<svddc_ $scalar _some_ $n x $m>]() {
                let a = random(($n, $m));
                test::<$scalar>(&a, UVTFlag::Some);
            }

            #[test]
            fn [<svddc_ $scalar _none_ $n x $m>]() {
                let a = random(($n, $m));
                test::<$scalar>(&a, UVTFlag::None);
            }

            #[test]
            fn [<svddc_ $scalar _full_ $n x $m _t>]() {
                let a = random(($n, $m).f());
                test::<$scalar>(&a, UVTFlag::Full);
            }

            #[test]
            fn [<svddc_ $scalar _some_ $n x $m _t>]() {
                let a = random(($n, $m).f());
                test::<$scalar>(&a, UVTFlag::Some);
            }

            #[test]
            fn [<svddc_ $scalar _none_ $n x $m _t>]() {
                let a = random(($n, $m).f());
                test::<$scalar>(&a, UVTFlag::None);
            }
        }
    };
}

test_svd_impl!(f64, 3, 3);
test_svd_impl!(f64, 4, 3);
test_svd_impl!(f64, 3, 4);
test_svd_impl!(c64, 3, 3);
test_svd_impl!(c64, 4, 3);
test_svd_impl!(c64, 3, 4);
