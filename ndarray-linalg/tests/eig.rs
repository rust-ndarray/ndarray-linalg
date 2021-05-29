use ndarray::*;
use ndarray_linalg::*;

// Test Av_i = e_i v_i for i = 0..n
fn test_eig<T: Scalar>(a: Array2<T>, eigs: Array1<T::Complex>, vecs: Array2<T::Complex>)
where
    T::Complex: Lapack,
{
    println!("a\n{:+.4}", &a);
    println!("eigs\n{:+.4}", &eigs);
    println!("vec\n{:+.4}", &vecs);
    let a: Array2<T::Complex> = a.map(|v| v.as_c());
    for (&e, v) in eigs.iter().zip(vecs.axis_iter(Axis(1))) {
        let av = a.dot(&v);
        let ev = v.mapv(|val| val * e);
        println!("av = {:+.4}", &av);
        println!("ev = {:+.4}", &ev);
        assert_close_l2!(&av, &ev, T::real(1e-3));
    }
}

// Test case for real Eigenvalue problem
//
//  -1.01   0.86  -4.60   3.31  -4.81
//   3.98   0.53  -7.04   5.29   3.55
//   3.30   8.26  -3.89   8.20  -1.51
//   4.43   4.96  -7.66  -7.33   6.18
//   7.31  -6.43  -6.16   2.47   5.58
//
// Eigenvalues
// (  2.86, 10.76) (  2.86,-10.76) ( -0.69,  4.70) ( -0.69, -4.70) -10.46
//
// Left eigenvectors
// (  0.04,  0.29) (  0.04, -0.29) ( -0.13, -0.33) ( -0.13,  0.33)   0.04
// (  0.62,  0.00) (  0.62,  0.00) (  0.69,  0.00) (  0.69,  0.00)   0.56
// ( -0.04, -0.58) ( -0.04,  0.58) ( -0.39, -0.07) ( -0.39,  0.07)  -0.13
// (  0.28,  0.01) (  0.28, -0.01) ( -0.02, -0.19) ( -0.02,  0.19)  -0.80
// ( -0.04,  0.34) ( -0.04, -0.34) ( -0.40,  0.22) ( -0.40, -0.22)   0.18
//
// Right eigenvectors
// (  0.11,  0.17) (  0.11, -0.17) (  0.73,  0.00) (  0.73,  0.00)   0.46
// (  0.41, -0.26) (  0.41,  0.26) ( -0.03, -0.02) ( -0.03,  0.02)   0.34
// (  0.10, -0.51) (  0.10,  0.51) (  0.19, -0.29) (  0.19,  0.29)   0.31
// (  0.40, -0.09) (  0.40,  0.09) ( -0.08, -0.08) ( -0.08,  0.08)  -0.74
// (  0.54,  0.00) (  0.54,  0.00) ( -0.29, -0.49) ( -0.29,  0.49)   0.16
//
// - https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dgeev_ex.f.htm
//
fn test_matrix_real<T: Scalar>() -> Array2<T::Real> {
    array![
        [
            T::real(-1.01),
            T::real(0.86),
            T::real(-4.60),
            T::real(3.31),
            T::real(-4.81)
        ],
        [
            T::real(3.98),
            T::real(0.53),
            T::real(-7.04),
            T::real(5.29),
            T::real(3.55)
        ],
        [
            T::real(3.30),
            T::real(8.26),
            T::real(-3.89),
            T::real(8.20),
            T::real(-1.51)
        ],
        [
            T::real(4.43),
            T::real(4.96),
            T::real(-7.66),
            T::real(-7.33),
            T::real(6.18)
        ],
        [
            T::real(7.31),
            T::real(-6.43),
            T::real(-6.16),
            T::real(2.47),
            T::real(5.58)
        ],
    ]
}

fn test_matrix_real_t<T: Scalar>() -> Array2<T::Real> {
    let orig = test_matrix_real::<T>();
    let mut out = Array2::zeros(orig.raw_dim().f());
    out.assign(&orig);
    out
}

fn answer_eig_real<T: Scalar>() -> Array1<T::Complex> {
    array![
        T::complex(-10.46, 0.00),
        T::complex(-0.69, 4.70),
        T::complex(-0.69, -4.70),
        T::complex(2.86, 10.76),
        T::complex(2.86, -10.76),
    ]
}

// Test case for {c,z}geev
//
// ( -3.84,  2.25) ( -8.94, -4.75) (  8.95, -6.53) ( -9.87,  4.82)
// ( -0.66,  0.83) ( -4.40, -3.82) ( -3.50, -4.26) ( -3.15,  7.36)
// ( -3.99, -4.73) ( -5.88, -6.60) ( -3.36, -0.40) ( -0.75,  5.23)
// (  7.74,  4.18) (  3.66, -7.53) (  2.58,  3.60) (  4.59,  5.41)
//
// Eigenvalues
// ( -9.43,-12.98) ( -3.44, 12.69) (  0.11, -3.40) (  5.76,  7.13)
//
// Left eigenvectors
// (  0.24, -0.18) (  0.61,  0.00) ( -0.18, -0.33) (  0.28,  0.09)
// (  0.79,  0.00) ( -0.05, -0.27) (  0.82,  0.00) ( -0.55,  0.16)
// (  0.22, -0.27) ( -0.21,  0.53) ( -0.37,  0.15) (  0.45,  0.09)
// ( -0.02,  0.41) (  0.40, -0.24) (  0.06,  0.12) (  0.62,  0.00)
//
// Right eigenvectors
// (  0.43,  0.33) (  0.83,  0.00) (  0.60,  0.00) ( -0.31,  0.03)
// (  0.51, -0.03) (  0.08, -0.25) ( -0.40, -0.20) (  0.04,  0.34)
// (  0.62,  0.00) ( -0.25,  0.28) ( -0.09, -0.48) (  0.36,  0.06)
// ( -0.23,  0.11) ( -0.10, -0.32) ( -0.43,  0.13) (  0.81,  0.00)
//
// - https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/zgeev_ex.f.htm
//
fn test_matrix_complex<T: Scalar>() -> Array2<T::Complex> {
    array![
        [
            T::complex(-3.84, 2.25),
            T::complex(-8.94, -4.75),
            T::complex(8.95, -6.53),
            T::complex(-9.87, 4.82)
        ],
        [
            T::complex(-0.66, 0.83),
            T::complex(-4.40, -3.82),
            T::complex(-3.50, -4.26),
            T::complex(-3.15, 7.36)
        ],
        [
            T::complex(-3.99, -4.73),
            T::complex(-5.88, -6.60),
            T::complex(-3.36, -0.40),
            T::complex(-0.75, 5.23)
        ],
        [
            T::complex(7.74, 4.18),
            T::complex(3.66, -7.53),
            T::complex(2.58, 3.60),
            T::complex(4.59, 5.41)
        ]
    ]
}

fn test_matrix_complex_t<T: Scalar>() -> Array2<T::Complex> {
    let orig = test_matrix_complex::<T>();
    let mut out = Array2::zeros(orig.raw_dim().f());
    out.assign(&orig);
    out
}

fn answer_eig_complex<T: Scalar>() -> Array1<T::Complex> {
    array![
        T::complex(-9.43, -12.98),
        T::complex(-3.44, 12.69),
        T::complex(0.11, -3.40),
        T::complex(5.76, 7.13)
    ]
}

// re-evaluated eigenvalues in f64 accuracy
fn answer_eigvectors_complex<T: Scalar>() -> Array2<T::Complex> {
    array![
        [
            T::complex(0.4308565200776108, 0.32681273781262143),
            T::complex(0.8256820507672813, 0.),
            T::complex(0.5983959785539453, 0.),
            T::complex(-0.30543190348437826, 0.03333164861799901)
        ],
        [
            T::complex(0.5087414602970965, -0.02883342170692809),
            T::complex(0.07502916788141115, -0.2487285045091665),
            T::complex(-0.40047616275207687, -0.2014492227625603),
            T::complex(0.03978282815783273, 0.34450765221546126)
        ],
        [
            T::complex(0.6198496527657755, 0.),
            T::complex(-0.24575578997801528, 0.27887240221169646),
            T::complex(-0.09008001907594984, -0.4752646215391732),
            T::complex(0.3583254365159844, 0.06064506988524665)
        ],
        [
            T::complex(-0.22692824331926856, 0.11043927846403584),
            T::complex(-0.10343406372814358, -0.3192014653632327),
            T::complex(-0.43484029549540404, 0.13372491785816037),
            T::complex(0.8082432893178352, 0.)
        ]
    ]
}

macro_rules! impl_test_real {
    ($real:ty) => {
        paste::item! {
            #[test]
            fn [<$real _eigvals >]() {
                let a = test_matrix_real::<$real>();
                let (e, _vecs) = a.eig().unwrap();
                assert_close_l2!(&e, &answer_eig_real::<$real>(), 1.0e-3);
            }

            #[test]
            fn [<$real _eigvals_t>]() {
                let a = test_matrix_real_t::<$real>();
                let (e, _vecs) = a.eig().unwrap();
                assert_close_l2!(&e, &answer_eig_real::<$real>(), 1.0e-3);
            }

            #[test]
            fn [<$real _eig>]() {
                let a = test_matrix_real::<$real>();
                let (e, vecs) = a.eig().unwrap();
                test_eig(a, e, vecs);
            }

            #[test]
            fn [<$real _eig_t>]() {
                let a = test_matrix_real_t::<$real>();
                let (e, vecs) = a.eig().unwrap();
                test_eig(a, e, vecs);
            }

        } // paste::item!
    };
}

impl_test_real!(f32);
impl_test_real!(f64);

macro_rules! impl_test_complex {
    ($complex:ty) => {
        paste::item! {
            #[test]
            fn [<$complex _eigvals >]() {
                let a = test_matrix_complex::<$complex>();
                let (e, _vecs) = a.eig().unwrap();
                assert_close_l2!(&e, &answer_eig_complex::<$complex>(), 1.0e-3);
            }

            #[test]
            fn [<$complex _eigvals_t>]() {
                let a = test_matrix_complex_t::<$complex>();
                let (e, _vecs) = a.eig().unwrap();
                assert_close_l2!(&e, &answer_eig_complex::<$complex>(), 1.0e-3);
            }

            #[test]
            fn [<$complex _eigvector>]() {
                let a = test_matrix_complex::<$complex>();
                let (_e, vecs) = a.eig().unwrap();
                assert_close_l2!(&vecs, &answer_eigvectors_complex::<$complex>(), 1.0e-3);
            }

            #[test]
            fn [<$complex _eigvector_t>]() {
                let a = test_matrix_complex_t::<$complex>();
                let (_e, vecs) = a.eig().unwrap();
                assert_close_l2!(&vecs, &answer_eigvectors_complex::<$complex>(), 1.0e-3);
            }

            #[test]
            fn [<$complex _eig>]() {
                let a = test_matrix_complex::<$complex>();
                let (e, vecs) = a.eig().unwrap();
                test_eig(a, e, vecs);
            }

            #[test]
            fn [<$complex _eig_t>]() {
                let a = test_matrix_complex_t::<$complex>();
                let (e, vecs) = a.eig().unwrap();
                test_eig(a, e, vecs);
            }
        } // paste::item!
    };
}

impl_test_complex!(c32);
impl_test_complex!(c64);
