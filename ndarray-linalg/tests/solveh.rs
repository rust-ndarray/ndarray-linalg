use ndarray::*;
use ndarray_linalg::*;
use num_complex::{Complex32, Complex64};

fn test_solveh<A>(n: usize, transpose: bool, rtol: A::Real)
where
    A: Scalar + Lapack,
{
    let a: Array2<A> = if transpose {
        random_hpd(n).reversed_axes()
    } else {
        random_hpd(n)
    };
    let x: Array1<A> = random(n);
    let b = a.dot(&x);
    let mut solutions = Vec::new();
    solutions.push(a.solveh(&b).unwrap());
    solutions.push(a.factorizeh().unwrap().solveh(&b).unwrap());
    solutions.push(a.factorizeh_into().unwrap().solveh(&b).unwrap());
    for solution in solutions {
        assert_close_l2!(&x, &solution, rtol);
    }
}

fn test_solveh_into<A>(n: usize, transpose: bool, rtol: A::Real)
where
    A: Scalar + Lapack,
{
    let a: Array2<A> = if transpose {
        random_hpd(n).reversed_axes()
    } else {
        random_hpd(n)
    };
    let x: Array1<A> = random(n);
    let b = a.dot(&x);
    let mut solutions = Vec::new();
    solutions.push(a.solveh_into(b.clone()).unwrap());
    solutions.push(a.factorizeh().unwrap().solveh_into(b.clone()).unwrap());
    solutions.push(a.factorizeh_into().unwrap().solveh_into(b.clone()).unwrap());
    for solution in solutions {
        assert_close_l2!(&x, &solution, rtol);
    }
}

#[test]
fn solveh_empty() {
    test_solveh::<f32>(0, false, 0.);
    test_solveh::<f64>(0, false, 0.);
    test_solveh::<Complex32>(0, false, 0.);
    test_solveh::<Complex64>(0, false, 0.);
}

#[test]
fn solveh_random_float() {
    for n in 1..=8 {
        test_solveh::<f32>(n, false, 1e-6);
        test_solveh::<f64>(n, false, 1e-9);
    }
}

#[test]
fn solveh_random_complex() {
    for n in 1..=8 {
        test_solveh::<Complex32>(n, false, 1e-6);
        test_solveh::<Complex64>(n, false, 1e-9);
    }
}

#[test]
fn solveh_into_random_float() {
    for n in 1..=8 {
        test_solveh_into::<f32>(n, false, 1e-6);
        test_solveh_into::<f64>(n, false, 1e-9);
    }
}

#[test]
fn solveh_into_random_complex() {
    for n in 1..=8 {
        test_solveh_into::<Complex32>(n, false, 1e-6);
        test_solveh_into::<Complex64>(n, false, 1e-9);
    }
}

#[test]
fn solveh_random_float_t() {
    for n in 1..=8 {
        test_solveh::<f32>(n, true, 1e-6);
        test_solveh::<f64>(n, true, 1e-9);
    }
}

#[test]
fn solveh_random_complex_t() {
    for n in 1..=8 {
        test_solveh::<Complex32>(n, true, 1e-6);
        test_solveh::<Complex64>(n, true, 1e-9);
    }
}

#[test]
fn solveh_into_random_float_t() {
    for n in 1..=8 {
        test_solveh_into::<f32>(n, true, 1e-6);
        test_solveh_into::<f64>(n, true, 1e-9);
    }
}

#[test]
fn solveh_into_random_complex_t() {
    for n in 1..=8 {
        test_solveh_into::<Complex32>(n, true, 1e-6);
        test_solveh_into::<Complex64>(n, true, 1e-9);
    }
}
