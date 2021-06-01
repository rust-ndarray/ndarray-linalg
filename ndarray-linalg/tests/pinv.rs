use ndarray::arr2;
use ndarray::*;
use ndarray_linalg::rank::Rank;
use ndarray_linalg::*;
use rand::{seq::SliceRandom, thread_rng};

/// creates a zero matrix which always has rank zero
pub fn zero_rank<A, S, Sh, D>(sh: Sh) -> ArrayBase<S, D>
where
    A: Scalar,
    S: DataOwned<Elem = A>,
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    ArrayBase::zeros(sh)
}

/// creates a random matrix and repeatedly creates a linear dependency between rows until the
/// rank drops.
pub fn partial_rank<A, Sh>(sh: Sh) -> Array2<A>
where
    A: Scalar + Lapack,
    Sh: ShapeBuilder<Dim = Ix2>,
{
    let mut rng = thread_rng();
    let mut result: Array2<A> = random(sh);
    println!("before: {:?}", result);

    let (n, m) = result.dim();
    println!("(n, m) => ({:?},{:?})", n, m);

    // create randomized row iterator
    let min_dim = n.min(m);
    let mut row_indexes = (0..min_dim).into_iter().collect::<Vec<usize>>();
    row_indexes.as_mut_slice().shuffle(&mut rng);
    let mut row_index_iter = row_indexes.iter().cycle();

    for count in 1..=10 {
        println!("count: {}", count);
        let (&x, &y) = (
            row_index_iter.next().unwrap(),
            row_index_iter.next().unwrap(),
        );
        let (from_row_index, to_row_index) = if x < y { (x, y) } else { (y, x) };
        println!("(r_f, r_t) => ({:?},{:?})", from_row_index, to_row_index);

        let mut it = result.outer_iter_mut();
        let from_row = it.nth(from_row_index).unwrap();
        let mut to_row = it.nth(to_row_index - (from_row_index + 1)).unwrap();

        // set the to_row with the value of the from_row multiplied by rand_multiple
        let rand_multiple = A::rand(&mut rng);
        println!("rand_multiple: {:?}", rand_multiple);
        Zip::from(&mut to_row)
            .and(&from_row)
            .for_each(|r1, r2| *r1 = *r2 * rand_multiple);

        if let Ok(rank) = result.rank() {
            println!("result: {:?}", result);
            println!("rank: {:?}", rank);
            if rank > 0 && rank < min_dim {
                return result;
            }
        }
    }
    unreachable!("unable to generate random partial rank matrix after making 10 mutations")
}

/// creates a random matrix and insures it is full rank.
pub fn full_rank<A, Sh>(sh: Sh) -> Array2<A>
where
    A: Scalar + Lapack,
    Sh: ShapeBuilder<Dim = Ix2> + Clone,
{
    for _ in 0..10 {
        let r: Array2<A> = random(sh.clone());
        let (n, m) = r.dim();
        let n = n.min(m);
        if let Ok(rank) = r.rank() {
            println!("result: {:?}", r);
            println!("rank: {:?}", rank);
            if rank == n {
                return r;
            }
        }
    }
    unreachable!("unable to generate random full rank matrix in 10 tries")
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
