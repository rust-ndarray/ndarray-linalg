use ndarray::*;
use ndarray_linalg::*;

fn test(a: &Array2<f64>, flag: UVTFlag) {
    let (n, m) = a.dim();
    let k = n.min(m);
    let answer = a.clone();
    println!("a = \n{:?}", a);
    let (u, s, vt): (_, Array1<_>, _) = a.svddc(flag).unwrap();
    let mut sm = match flag {
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
        sm[(i, i)] = s[i];
    }
    assert_close_l2!(&u.dot(&sm).dot(&vt), &answer, 1e-7);
}

macro_rules! test_svd_impl {
    ($n:expr, $m:expr) => {
        paste::item! {
            #[test]
            fn [<svddc_full_ $n x $m>]() {
                let a = random(($n, $m));
                test(&a, UVTFlag::Full);
            }

            #[test]
            fn [<svddc_some_ $n x $m>]() {
                let a = random(($n, $m));
                test(&a, UVTFlag::Some);
            }

            #[test]
            fn [<svddc_none_ $n x $m>]() {
                let a = random(($n, $m));
                test(&a, UVTFlag::None);
            }

            #[test]
            fn [<svddc_full_ $n x $m _t>]() {
                let a = random(($n, $m).f());
                test(&a, UVTFlag::Full);
            }

            #[test]
            fn [<svddc_some_ $n x $m _t>]() {
                let a = random(($n, $m).f());
                test(&a, UVTFlag::Some);
            }

            #[test]
            fn [<svddc_none_ $n x $m _t>]() {
                let a = random(($n, $m).f());
                test(&a, UVTFlag::None);
            }
        }
    };
}

test_svd_impl!(3, 3);
test_svd_impl!(4, 3);
test_svd_impl!(3, 4);
