use ndarray::*;
use ndarray_linalg::*;

#[test]
fn generalized_eigenvalue_fmt() {
    let ge0 = GeneralizedEigenvalue::Finite(0.1, (1.0, 10.0));
    assert_eq!(ge0.to_string(), "1.000e-1 (1.000e0/1.000e1)".to_string());

    let ge1 = GeneralizedEigenvalue::Indeterminate((1.0, 0.0));
    assert_eq!(ge1.to_string(), "∞ (1.000e0/0.000e0)".to_string());
}

#[test]
fn real_a_real_b_3x3_full_rank() {
    #[rustfmt::skip]
    let a = array![
        [ 2.0, 1.0, 8.0],
        [-2.0, 0.0, 3.0],
        [ 7.0, 6.0, 5.0],
    ];
    #[rustfmt::skip]
    let b = array![
        [ 1.0,  2.0, -7.0],
        [-3.0,  1.0,  6.0],
        [ 4.0, -5.0,  1.0],
    ];
    let (geneigvals, eigvecs) = (a.clone(), b.clone()).eig_generalized(None).unwrap();

    let a = a.map(|v| v.as_c());
    let b = b.map(|v| v.as_c());
    for (ge, vec) in geneigvals.iter().zip(eigvecs.columns()) {
        if let GeneralizedEigenvalue::Finite(e, _) = ge {
            let ebv = b.dot(&vec).map(|v| v * e);
            let av = a.dot(&vec);
            assert_close_l2!(&av, &ebv, 1e-7);
        }
    }

    let mut eigvals = geneigvals
        .iter()
        .filter_map(|ge: &GeneralizedEigenvalue<c64>| match ge {
            GeneralizedEigenvalue::Finite(e, _) => Some(e.clone()),
            GeneralizedEigenvalue::Indeterminate(_) => None,
        })
        .collect::<Vec<_>>();
    eigvals.sort_by(|a, b| a.re().partial_cmp(&b.re()).unwrap());
    let eigvals = Array1::from_vec(eigvals);
    // Reference eigenvalues from Mathematica
    assert_close_l2!(
        &eigvals,
        &array![-0.4415795111, 0.5619249537, 50.87965456].map(c64::from),
        1e-7
    );
}

#[test]
fn real_a_real_b_3x3_nullity_1() {
    #[rustfmt::skip]
    let a = array![
        [ 2.0, 1.0, 8.0],
        [-2.0, 0.0, 3.0],
        [ 7.0, 6.0, 5.0],
    ];
    #[rustfmt::skip]
    let b = array![
        [1.0,  2.0, 3.0],
        [0.0,  1.0, 1.0],
        [1.0, -1.0, 0.0],
    ];
    let (geneigvals, eigvecs) = (a.clone(), b.clone()).eig_generalized(Some(1e-4)).unwrap();

    let a = a.map(|v| v.as_c());
    let b = b.map(|v| v.as_c());
    for (ge, vec) in geneigvals.iter().zip(eigvecs.columns()) {
        if let GeneralizedEigenvalue::Finite(e, _) = ge {
            let ebv = b.dot(&vec).map(|v| v * e);
            let av = a.dot(&vec);
            assert_close_l2!(&av, &ebv, 1e-7);
        }
    }

    let mut eigvals = geneigvals
        .iter()
        .filter_map(|ge: &GeneralizedEigenvalue<c64>| match ge {
            GeneralizedEigenvalue::Finite(e, _) => Some(e.clone()),
            GeneralizedEigenvalue::Indeterminate(_) => None,
        })
        .collect::<Vec<_>>();
    eigvals.sort_by(|a, b| a.re().partial_cmp(&b.re()).unwrap());
    let eigvals = Array1::from_vec(eigvals);
    // Reference eigenvalues from Mathematica
    assert_close_l2!(
        &eigvals,
        &array![-12.91130192, 3.911301921].map(c64::from),
        1e-7
    );
}

#[test]
fn complex_a_complex_b_3x3_full_rank() {
    #[rustfmt::skip]
    let a = array![
        [c64::new(1.0,  2.0), c64::new(-3.0,  0.5), c64::new( 0.0, -1.0)],
        [c64::new(2.5, -4.0), c64::new( 1.0,  1.0), c64::new(-1.5,  2.5)],
        [c64::new(0.0,  0.0), c64::new( 3.0, -2.0), c64::new( 4.0,  4.0)],
    ];
    #[rustfmt::skip]
    let b = array![
        [c64::new(-2.0,  1.0), c64::new( 3.5, -1.0), c64::new( 1.0,  1.0)],
        [c64::new( 0.0, -3.0), c64::new( 2.0,  2.0), c64::new(-4.0,  0.0)],
        [c64::new( 5.0,  5.0), c64::new(-1.5,  1.5), c64::new( 0.0, -2.0)],
    ];
    let (geneigvals, eigvecs) = (a.clone(), b.clone()).eig_generalized(None).unwrap();

    let a = a.map(|v| v.as_c());
    let b = b.map(|v| v.as_c());
    for (ge, vec) in geneigvals.iter().zip(eigvecs.columns()) {
        if let GeneralizedEigenvalue::Finite(e, _) = ge {
            let ebv = b.dot(&vec).map(|v| v * e);
            let av = a.dot(&vec);
            assert_close_l2!(&av, &ebv, 1e-7);
        }
    }

    let mut eigvals = geneigvals
        .iter()
        .filter_map(|ge: &GeneralizedEigenvalue<c64>| match ge {
            GeneralizedEigenvalue::Finite(e, _) => Some(e.clone()),
            GeneralizedEigenvalue::Indeterminate(_) => None,
        })
        .collect::<Vec<_>>();
    eigvals.sort_by(|a, b| a.re().partial_cmp(&b.re()).unwrap());
    let eigvals = Array1::from_vec(eigvals);
    // Reference eigenvalues from Mathematica
    assert_close_l2!(
        &eigvals,
        &array![
            c64::new(-0.701598, -1.71262),
            c64::new(-0.67899, -0.0172468),
            c64::new(0.59059, 0.276034)
        ],
        1e-5
    );
}

#[test]
fn complex_a_complex_b_3x3_nullity_1() {
    #[rustfmt::skip]
    let a = array![
        [c64::new(1.0,  2.0), c64::new(-3.0,  0.5), c64::new( 0.0, -1.0)],
        [c64::new(2.5, -4.0), c64::new( 1.0,  1.0), c64::new(-1.5,  2.5)],
        [c64::new(0.0,  0.0), c64::new( 3.0, -2.0), c64::new( 4.0,  4.0)],
    ];
    #[rustfmt::skip]
    let b = array![
        [c64::new(-2.55604, -4.10176), c64::new(9.03944,  3.745000), c64::new(35.4641,  21.1704)],
        [c64::new( 7.85029,  7.02144), c64::new(9.23225, -0.479451), c64::new(13.9507, -16.5402)],
        [c64::new(-4.47803,  3.98981), c64::new(9.44434, -4.519970), c64::new(40.9006, -23.5060)],
    ];
    let (geneigvals, eigvecs) = (a.clone(), b.clone()).eig_generalized(Some(1e-4)).unwrap();

    let a = a.map(|v| v.as_c());
    let b = b.map(|v| v.as_c());
    for (ge, vec) in geneigvals.iter().zip(eigvecs.columns()) {
        if let GeneralizedEigenvalue::Finite(e, _) = ge {
            let ebv = b.dot(&vec).map(|v| v * e);
            let av = a.dot(&vec);
            assert_close_l2!(&av, &ebv, 1e-7);
        }
    }

    let mut eigvals = geneigvals
        .iter()
        .filter_map(|ge: &GeneralizedEigenvalue<c64>| match ge {
            GeneralizedEigenvalue::Finite(e, _) => Some(e.clone()),
            GeneralizedEigenvalue::Indeterminate(_) => None,
        })
        .collect::<Vec<_>>();
    eigvals.sort_by(|a, b| a.re().partial_cmp(&b.re()).unwrap());
    let eigvals = Array1::from_vec(eigvals);
    // Reference eigenvalues from Mathematica
    assert_close_l2!(
        &eigvals,
        &array![
            c64::new(-0.0620674, -0.270016),
            c64::new(0.0218236, 0.0602709),
        ],
        1e-5
    );
}
