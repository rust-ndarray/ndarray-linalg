include!("header.rs");

fn test_lu(a: Array<f64, Ix2>) {
    println!("a = \n{:?}", &a);
    let (p, l, u) = a.clone().lu().unwrap();
    println!("P = \n{:?}", &p);
    println!("L = \n{:?}", &l);
    println!("U = \n{:?}", &u);
    println!("LU = \n{:?}", l.dot(&u));
    all_close_l2(&l.dot(&u).permutated(&p), &a, 1e-7).unwrap();
}

macro_rules! test_lu_upper {
    ($testname:ident, $testname_t:ident, $n:expr, $m:expr) => {
#[test]
fn $testname() {
    let r_dist = RealNormal::new(0., 1.);
    let mut a = Array::<f64, _>::random(($n, $m), r_dist);
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    test_lu(a);
}
#[test]
fn $testname_t() {
    let r_dist = RealNormal::new(0., 1.);
    let mut a = Array::<f64, _>::random(($m, $n), r_dist).reversed_axes();
    for ((i, j), val) in a.indexed_iter_mut() {
        if i > j {
            *val = 0.0;
        }
    }
    test_lu(a);
}
}} // end test_lu_upper
test_lu_upper!(lu_square_upper, lu_square_upper_t, 3, 3);
test_lu_upper!(lu_3x4_upper, lu_3x4_upper_t, 3, 4);
test_lu_upper!(lu_4x3_upper, lu_4x3_upper_t, 4, 3);

macro_rules! test_lu_lower {
    ($testname:ident, $testname_t:ident, $n:expr, $m:expr) => {
#[test]
fn $testname() {
    let r_dist = RealNormal::new(0., 1.);
    let mut a = Array::<f64, _>::random(($n, $m), r_dist);
    for ((i, j), val) in a.indexed_iter_mut() {
        if i < j {
            *val = 0.0;
        }
    }
    test_lu(a);
}
#[test]
fn $testname_t() {
    let r_dist = RealNormal::new(0., 1.);
    let mut a = Array::<f64, _>::random(($m, $n), r_dist).reversed_axes();
    for ((i, j), val) in a.indexed_iter_mut() {
        if i < j {
            *val = 0.0;
        }
    }
    test_lu(a);
}
}} // end test_lu_lower
test_lu_lower!(lu_square_lower, lu_square_lower_t, 3, 3);
test_lu_lower!(lu_3x4_lower, lu_3x4_lower_t, 3, 4);
test_lu_lower!(lu_4x3_lower, lu_4x3_lower_t, 4, 3);

macro_rules! test_lu {
    ($testname:ident, $testname_t:ident, $n:expr, $m:expr) => {
#[test]
fn $testname() {
    let r_dist = RealNormal::new(0., 1.);
    let a = Array::<f64, _>::random(($n, $m), r_dist);
    test_lu(a);
}
#[test]
fn $testname_t() {
    let r_dist = RealNormal::new(0., 1.);
    let a = Array::<f64, _>::random(($m, $n), r_dist).reversed_axes();
    test_lu(a);
}
}} // end test_lu
test_lu!(lu_square, lu_square_t, 3, 3);
test_lu!(lu_3x4, lu_3x4_t, 3, 4);
test_lu!(lu_4x3, lu_4x3_t, 4, 3);
