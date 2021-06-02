use ndarray::arr2;
use ndarray::*;
use ndarray_linalg::*;
use rand::{thread_rng, Rng};

/// create a zero rank array
pub fn zero_rank<A, Sh>(sh: Sh) -> Array2<A>
where
    A: Scalar + Lapack,
    Sh: ShapeBuilder<Dim = Ix2> + Clone,
{
    random_with_rank(sh, 0)
}

/// create a random matrix with a random partial rank.
pub fn partial_rank<A, Sh>(sh: Sh) -> Array2<A>
where
    A: Scalar + Lapack,
    Sh: ShapeBuilder<Dim = Ix2> + Clone,
{
    let mut rng = thread_rng();
    let (m, n) = sh.clone().into_shape().raw_dim().into_pattern();
    let min_dim = n.min(m);
    let rank = rng.gen_range(1..min_dim);
    println!("desired rank = {}", rank);
    random_with_rank(sh, rank)
}

/// create a random matrix and ensures it is full rank.
pub fn full_rank<A, Sh>(sh: Sh) -> Array2<A>
where
    A: Scalar + Lapack,
    Sh: ShapeBuilder<Dim = Ix2> + Clone,
{
    let (m, n) = sh.clone().into_shape().raw_dim().into_pattern();
    let min_dim = n.min(m);
    random_with_rank(sh, min_dim)
}

fn test<T: Scalar + Lapack>(a: &Array2<T>, tolerance: T::Real) {
    println!("a = \n{:?}", &a);
    let a_plus: Array2<_> = a.pinv(None).unwrap();
    println!("a_plus = \n{:?}", &a_plus);
    let ident = a.dot(&a_plus);
    assert_close_l2!(&ident.dot(a), &a, tolerance);
    assert_close_l2!(&a_plus.dot(&ident), &a_plus, tolerance);
}

macro_rules! test_both_impl {
    ($type:ty, $test:tt, $n:expr, $m:expr, $t:expr) => {
        paste::item! {
            #[test]
            fn [<pinv_test_ $type _ $test _ $n x $m _r>]() {
                let a: Array2<$type> = $test(($n, $m));
                test::<$type>(&a, $t);
            }

            #[test]
            fn [<pinv_test_ $type _ $test _ $n x $m _c>]() {
                let a = $test(($n, $m).f());
                test::<$type>(&a, $t);
            }
        }
    };
}

macro_rules! test_pinv_impl {
    ($type:ty, $n:expr, $m:expr, $a:expr) => {
        test_both_impl!($type, zero_rank, $n, $m, $a);
        test_both_impl!($type, partial_rank, $n, $m, $a);
        test_both_impl!($type, full_rank, $n, $m, $a);
    };
}

test_pinv_impl!(f32, 3, 3, 1e-4);
test_pinv_impl!(f32, 4, 3, 1e-4);
test_pinv_impl!(f32, 3, 4, 1e-4);

test_pinv_impl!(c32, 3, 3, 1e-4);
test_pinv_impl!(c32, 4, 3, 1e-4);
test_pinv_impl!(c32, 3, 4, 1e-4);

test_pinv_impl!(f64, 3, 3, 1e-12);
test_pinv_impl!(f64, 4, 3, 1e-12);
test_pinv_impl!(f64, 3, 4, 1e-12);

test_pinv_impl!(c64, 3, 3, 1e-12);
test_pinv_impl!(c64, 4, 3, 1e-12);
test_pinv_impl!(c64, 3, 4, 1e-12);

//
// This matrix was taken from 7.1.1 Test1 in
// "On Moore-Penrose Pseudoinverse Computation for Stiffness Matrices Resulting
//  from Higher Order Approximation" by Marek Klimczak
// https://doi.org/10.1155/2019/5060397
//
#[test]
fn pinv_test_single_value_less_then_threshold_3x3() {
    #[rustfmt::skip]
    let a: Array2<f64> = arr2(&[
            [ 1., -1.,  0.],
            [-1.,  2., -1.],
            [ 0., -1.,  1.]
        ],
    );
    #[rustfmt::skip]
    let a_plus_actual: Array2<f64> = arr2(&[
            [ 5. / 9., -1. / 9., -4. / 9.],
            [-1. / 9.,  2. / 9., -1. / 9.],
            [-4. / 9., -1. / 9.,  5. / 9.],
        ],
    );
    let a_plus: Array2<_> = a.pinv(None).unwrap();
    println!("a_plus -> {:?}", &a_plus);
    println!("a_plus_actual -> {:?}", &a_plus);
    assert_close_l2!(&a_plus, &a_plus_actual, 1e-15);
}
