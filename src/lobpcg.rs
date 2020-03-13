use num_traits::NumCast;
use ndarray::prelude::*;
use ndarray::OwnedRepr;
use crate::{cholesky::*, triangular::*, eigh::*, norm::*, close_l2};
//use sprs::CsMat;
use crate::{Scalar, Lapack};
use crate::error::{Result, LinalgError};

pub enum Order {
    Largest,
    Smallest
}

#[derive(Debug)]
pub enum EigResult<A> {
    Ok(Array1<A>, Array2<A>, Vec<A>),
    Err(Array1<A>, Array2<A>, Vec<A>, LinalgError),
    NoResult(LinalgError)
}

fn sorted_eig<A: Scalar + Lapack>(a: ArrayView2<A>, b: Option<ArrayView2<A>>, size: usize, order: &Order) -> Result<(Array1<A>, Array2<A>)> {
    //close_l2(&input, &input.t(), 1e-4);

    let (vals, vecs) = match b {
        Some(b) => (a, b).eigh(UPLO::Upper).map(|x| (x.0, (x.1).0))?,
        _ => a.eigh(UPLO::Upper)?
    };

    let n = a.len_of(Axis(0));

    Ok(match order {
        Order::Largest => (vals.slice_move(s![n-size..; -1]).mapv(|x| Scalar::from_real(x)), vecs.slice_move(s![.., n-size..; -1])),
        Order::Smallest => (vals.slice_move(s![..size]).mapv(|x| Scalar::from_real(x)), vecs.slice_move(s![.., ..size]))
    })
}

fn ndarray_mask<A: Scalar>(matrix: ArrayView2<A>, mask: &[bool]) -> Array2<A> {
    let (rows, cols) = (matrix.nrows(), matrix.ncols());

    assert_eq!(mask.len(), cols);

    let n_positive = mask.iter().filter(|x| **x).count();

    let matrix = matrix.gencolumns().into_iter().zip(mask.iter())
        .filter(|(_,x)| **x)
        .map(|(x,_)| x.to_vec())
        .flatten()
        .collect::<Vec<A>>();

    Array2::from_shape_vec((n_positive, rows), matrix).unwrap().reversed_axes()
}

fn apply_constraints<A: Scalar + Lapack>(
    mut v: ArrayViewMut<A, Ix2>,
    fact_yy: &CholeskyFactorized<OwnedRepr<A>>,
    y: ArrayView2<A>
) {
    let gram_yv = y.t().dot(&v);

    let u = gram_yv.genrows().into_iter()
        .map(|x| fact_yy.solvec(&x).unwrap().to_vec())
        .flatten()
        .collect::<Vec<A>>();

    let u = Array2::from_shape_vec((5, 5), u).unwrap();

    v -= &(y.dot(&u));
}

fn orthonormalize<T: Scalar + Lapack>(
    v: Array2<T>
) -> Result<(Array2<T>, Array2<T>)> {
    let gram_vv = v.t().dot(&v);
    let gram_vv_fac = gram_vv.cholesky(UPLO::Lower)?;

    close_l2(&gram_vv, &gram_vv_fac.dot(&gram_vv_fac.t()), NumCast::from(1e-5).unwrap());

    let v_t = v.reversed_axes();
    let u = gram_vv_fac.solve_triangular(UPLO::Lower, Diag::NonUnit, &v_t)?
        .reversed_axes();

    Ok((u, gram_vv_fac))
}

pub fn lobpcg<A: Scalar + Lapack + PartialOrd + Default, F: Fn(ArrayView2<A>) -> Array2<A>>(
    a: F,
    x: Array2<A>,
    m: Option<Array2<A>>,
    y: Option<Array2<A>>,
    tol: A::Real, maxiter: usize,
    order: Order
) -> EigResult<A> {
    // the target matrix should be symmetric and quadratic
    //assert!(sprs::is_symmetric(&A));

    // the initital approximation should be maximal square
    // n is the dimensionality of the problem
    let (n, size_x) = (x.nrows(), x.ncols());
    assert!(size_x <= n);

    let size_y = match y {
        Some(ref y) => y.ncols(),
        _ => 0
    };

    if (n - size_y) < 5 * size_x {
        panic!("Please use a different approach, the LOBPCG method only supports the calculation of a couple of eigenvectors!");
    }

    // cap the number of iteration
    let mut iter = usize::min(n, maxiter);

    // factorize yy for later use
    let fact_yy = y.as_ref().map(|x| x.t().dot(x).factorizec(UPLO::Upper).unwrap());

    // orthonormalize the initial guess and calculate matrices AX and XAX
    let (x, _) = match orthonormalize(x) {
        Ok(x) => x,
        Err(err) => return EigResult::NoResult(err)
    };

    let ax = a(x.view());
    let xax = x.t().dot(&ax);

    // perform eigenvalue decomposition on XAX
    let (mut lambda, eig_block) = match sorted_eig(xax.view(), None, size_x, &order) {
        Ok(x) => x,
        Err(err) => return EigResult::NoResult(err)
    };

    //dbg!(&lambda, &eig_block);

    // initiate X and AX with eigenvector
    let mut x = x.dot(&eig_block);
    let mut ax = ax.dot(&eig_block);

    //dbg!(&X, &AX);
    let mut activemask = vec![true; size_x];
    let mut residual_norms = Vec::new();
    let mut results = vec![(lambda.clone(), x.clone())];

    let mut previous_block_size = size_x;

    let mut ident: Array2<A> = Array2::eye(size_x);
    let ident0: Array2<A> = Array2::eye(size_x);

    let mut ap: Option<(Array2<A>, Array2<A>)> = None;

    let final_norm = loop {
        // calculate residual
        let lambda_tmp = lambda.clone().insert_axis(Axis(0));
        let tmp = &x * &lambda_tmp;

        let r = &ax - &tmp;

        // calculate L2 norm of error for every eigenvalue
        let tmp = r.gencolumns().into_iter().map(|x| x.norm()).collect::<Vec<A::Real>>();
        residual_norms.push(tmp.clone());

        // disable eigenvalues which are below the tolerance threshold
        activemask = tmp.iter().zip(activemask.iter()).map(|(x, a)| *x > tol && *a).collect();

        // resize identity block if necessary
        let current_block_size = activemask.iter().filter(|x| **x).count();
        if current_block_size != previous_block_size {
            previous_block_size = current_block_size;
            ident = Array2::eye(current_block_size);
        }

        // if we are below the threshold for all eigenvalue or exceeded the number of iteration,
        // abort
        if current_block_size == 0 || iter == 0 {
            break Ok(tmp);
        }

        // select active eigenvalues, apply pre-conditioner, orthogonalize to Y and orthonormalize
        let mut active_block_r = ndarray_mask(r.view(), &activemask);
        if let Some(ref m) = m {
            active_block_r = m.dot(&active_block_r);
        }
        if let (Some(ref y), Some(ref fact_yy)) = (&y, &fact_yy) {
            apply_constraints(active_block_r.view_mut(), fact_yy, y.view());
        }

        let (r,_) = match orthonormalize(active_block_r) {
            Ok(x) => x,
            Err(err) => break Err(err)
        };

        let ar = a(r.view());
        
        // perform the Rayleigh Ritz procedure
        let xaw = x.t().dot(&ar);
        let waw = r.t().dot(&ar);
        let xw = x.t().dot(&r);

        // compute symmetric gram matrices
        let (gram_a, gram_b, active_p, active_ap) = if let Some((ref p, ref ap)) = ap {
            let active_p = ndarray_mask(p.view(), &activemask);
            let active_ap = ndarray_mask(ap.view(), &activemask);

            let (active_p, p_r) = orthonormalize(active_p).unwrap();
            //dbg!(&active_P, &P_R);
            let active_ap = match p_r.solve_triangular(UPLO::Lower, Diag::NonUnit, &active_ap.reversed_axes()) {
                Ok(x) => x,
                Err(err) => break Err(err)
            };

            let active_ap = active_ap.reversed_axes();

            //dbg!(&active_AP);
            //dbg!(&R);

            let xap = x.t().dot(&active_ap);
            let wap = r.t().dot(&active_ap);
            let pap = active_p.t().dot(&active_ap);
            let xp = x.t().dot(&active_p);
            let wp = r.t().dot(&active_p);

            (
                stack![Axis(0),
                    stack![Axis(1), Array2::from_diag(&lambda), xaw, xap],
                    stack![Axis(1), xaw.t(), waw, wap],
                    stack![Axis(1), xap.t(), wap.t(), pap]
                ],

                stack![Axis(0),
                    stack![Axis(1), ident0, xw, xp],
                    stack![Axis(1), xw.t(), ident, wp],
                    stack![Axis(1), xp.t(), wp.t(), ident]
                ],
                Some(active_p),
                Some(active_ap)
            )
        } else {
            (
                stack![Axis(0), 
                    stack![Axis(1), Array2::from_diag(&lambda), xaw],
                    stack![Axis(1), xaw.t(), waw]
                ],
                stack![Axis(0),
                    stack![Axis(1), ident0, xw],
                    stack![Axis(1), xw.t(), ident]
                ],
                None,
                None
            )
        };

        //assert!(is_symmetric(gramA.view()));
        //assert!(is_symmetric(gramB.view()));

        //dbg!(&gramA, &gramB);
        let (new_lambda, eig_vecs) = match sorted_eig(gram_a.view(), Some(gram_b.view()), size_x, &order) {
            Ok(x) => x,
            Err(err) => break Err(err)
        };
        lambda = new_lambda;

        //dbg!(&lambda, &eig_vecs);
        let (pp, app, eig_x) = if let (Some(_), (Some(ref active_p), Some(ref active_ap))) = (ap, (active_p, active_ap)) {

            let eig_x = eig_vecs.slice(s![..size_x, ..]);
            let eig_r = eig_vecs.slice(s![size_x..size_x+current_block_size, ..]);
            let eig_p = eig_vecs.slice(s![size_x+current_block_size.., ..]);

            //dbg!(&eig_X);
            //dbg!(&eig_R);
            //dbg!(&eig_P);

            //dbg!(&R, &AR, &active_P, &active_AP);

            let pp = r.dot(&eig_r) + active_p.dot(&eig_p);
            let app = ar.dot(&eig_r) + active_ap.dot(&eig_p);

            //dbg!(&pp);
            //dbg!(&app);

            (pp, app, eig_x)
        } else {
            let eig_x = eig_vecs.slice(s![..size_x, ..]);
            let eig_r = eig_vecs.slice(s![size_x.., ..]);

            let pp = r.dot(&eig_r);
            let app = ar.dot(&eig_r);

            (pp, app, eig_x)
        };

        x = x.dot(&eig_x) + &pp;
        ax = ax.dot(&eig_x) + &app;

        results.push((lambda.clone(), x.clone()));

        //dbg!(&X);
        //dbg!(&AX);

        ap = Some((pp, app));

        //dbg!(&ap);

        iter -= 1;
    };

    let best_idx = residual_norms.iter()
        .enumerate()
        .min_by(|&(_, item1): &(usize, &Vec<A::Real>), &(_, item2): &(usize, &Vec<A::Real>)| {
            let norm1: A::Real = item1.iter().map(|x| (*x)*(*x)).sum();
            let norm2: A::Real = item2.iter().map(|x| (*x)*(*x)).sum();
            norm1.partial_cmp(&norm2).unwrap()
        });

    match best_idx {
        Some((idx, norms)) => {
            let (vals, vecs) = results[idx].clone();
            let norms = norms.iter().map(|x| Scalar::from_real(*x)).collect();

            match final_norm {
                Ok(_) => EigResult::Ok(vals, vecs, norms),
                Err(err) => EigResult::Err(vals, vecs, norms, err)
            }
        },
        None => {
            match final_norm {
                Ok(_) => panic!("Not error available!"),
                Err(err) => EigResult::NoResult(err)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sorted_eig;
    use super::orthonormalize;
    use super::ndarray_mask;
    use super::Order;
    use super::lobpcg;
    use super::EigResult;
    use crate::close_l2;
    use ndarray::prelude::*;
    use crate::qr::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    
    #[test]
    fn test_sorted_eigen() {
        let matrix = Array2::random((10, 10), Uniform::new(0., 10.));
        let matrix = matrix.t().dot(&matrix);

        // return all eigenvectors with largest first
        let (vals, vecs) = sorted_eig(matrix.view(), None, 10, &Order::Largest).unwrap();

        // calculate V * A * V' and compare to original matrix
        let diag = Array2::from_diag(&vals);
        let rec = (vecs.dot(&diag)).dot(&vecs.t());

        close_l2(&matrix, &rec, 1e-5);
    }

    #[test]
    fn test_masking() {
        let matrix = Array2::random((10, 5), Uniform::new(0., 10.));
        let masked_matrix = ndarray_mask(matrix.view(), &[true, true, false, true, false]);
        close_l2(&masked_matrix.slice(s![.., 2]), &matrix.slice(s![.., 3]), 1e-12);
    }

    #[test]
    fn test_orthonormalize() {
        let matrix = Array2::random((10, 10), Uniform::new(-10., 10.));

        let (n, l) = orthonormalize(matrix.clone()).unwrap();

        // check for orthogonality
        let identity = n.dot(&n.t());
        close_l2(&identity, &Array2::eye(10), 1e-2);

        // compare returned factorization with QR decomposition
        let (_, r) = matrix.qr().unwrap();
        close_l2(&r.mapv(|x: f32| x.abs()) , &l.t().mapv(|x| x.abs()), 1e-2);
    }

    #[test]
    fn test_eigsolver_diag() {
        let diag = arr1(&[1.,2.,3.,4.,5.,6.,7.,8.,9.,10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.]);
        let a = Array2::from_diag(&diag);
        let x: Array2<f64> = Array2::random((20, 3), Uniform::new(0.0, 1.0));

        let result = lobpcg(|y| a.dot(&y), x, None, None, 1e-10, 20, Order::Smallest);
        match result {
            EigResult::Ok(vals, _, _) => close_l2(&vals, &arr1(&[1.0, 2.0, 3.0]), 1e-5),
            EigResult::Err(vals, _,_,_) => close_l2(&vals, &arr1(&[1.0, 2.0, 3.0]), 1e-5),
            EigResult::NoResult(err) => panic!("Did not converge: {:?}", err)
        }

        let x: Array2<f64> = Array2::random((20, 3), Uniform::new(0.0, 1.0));
        let result = lobpcg(|y| a.dot(&y), x, None, None, 1e-10, 20, Order::Largest);
        match result {
            EigResult::Ok(vals, _, _) => close_l2(&vals, &arr1(&[20.0, 19.0, 18.0]), 1e-5),
            EigResult::Err(vals, _,_,_) => close_l2(&vals, &arr1(&[20.0, 19.0, 18.0]), 1e-5),
            EigResult::NoResult(err) => panic!("Did not converge: {:?}", err)
        }
    }

    #[test]
    fn test_eigsolver() {
        let n = 50;
        let tmp = Array2::random((n, n), Uniform::new(0.0, 1.0));
        let (v, _) = orthonormalize(tmp).unwrap();

        // set eigenvalues in decreasing order
        let t = Array2::from_diag(&Array1::linspace(n as f64, 1.0, n));
        let a = v.dot(&t.dot(&v.t()));

        let x: Array2<f64> = Array2::random((n, 5), Uniform::new(0.0, 1.0));

        let result = lobpcg(|y| a.dot(&y), x, None, None, 1e-10, 20, Order::Largest);
        match result {
            EigResult::Ok(vals, _, _) | EigResult::Err(vals, _, _, _) => {
                close_l2(&vals, &arr1(&[50.0, 49.0, 48.0, 47.0, 46.0]), 1e-5);
            },
            EigResult::NoResult(err) => panic!("Did not converge: {:?}", err)
        }
    }
}
