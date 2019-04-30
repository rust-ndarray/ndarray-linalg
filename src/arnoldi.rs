use crate::{generate::*, inner::*, norm::Norm, types::*};
use ndarray::*;

/// Iterative orthogonalizer using modified Gram-Schmit procedure
#[derive(Debug, Clone)]
pub struct MGS<A> {
    /// Dimension of base space
    dimension: usize,
    /// Basis of spanned space
    q: Vec<Array1<A>>,
}

pub type Residual<S> = ArrayBase<S, Ix1>;
pub type Coefficient<A> = Array1<A>;

impl<A: Scalar + Lapack> MGS<A> {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            q: Vec::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.dimension
    }

    pub fn len(&self) -> usize {
        self.q.len()
    }

    /// Orthogonalize given vector using current basis
    ///
    /// Panic
    /// -------
    /// - if the size of the input array mismaches to the dimension
    pub fn orthogonalize<S>(&self, mut a: ArrayBase<S, Ix1>) -> (Residual<S>, Coefficient<A>)
    where
        S: DataMut<Elem = A>,
    {
        assert_eq!(a.len(), self.dim());
        let mut coef = Array1::zeros(self.len() + 1);
        for i in 0..self.len() {
            let q = &self.q[i];
            let c = q.inner(&a);
            azip!(mut a, q (q) in { *a = *a - c * q } );
            coef[i] = c;
        }
        let nrm = a.norm_l2();
        coef[self.len()] = A::from_real(nrm);
        (a, coef)
    }

    /// Add new vector if the residual is larger than relative tolerance
    ///
    /// Panic
    /// -------
    /// - if the size of the input array mismaches to the dimension
    pub fn append_if_independent<S>(&mut self, a: ArrayBase<S, Ix1>, rtol: A::Real) -> Option<Coefficient<A>>
    where
        S: Data<Elem = A>,
    {
        let a = a.into_owned();
        let (mut a, coef) = self.orthogonalize(a);
        let nrm = coef[coef.len()].re();
        if nrm < rtol {
            // Linearly dependent
            return None;
        }
        azip!(mut a in { *a = *a / A::from_real(nrm) });
        self.q.push(a);
        Some(coef)
    }

    /// Get orthogonal basis as Q matrix
    pub fn get_q(&self) -> Array2<A> {
        hstack(&self.q).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert::*;

    const N: usize = 5;

    #[test]
    fn new() {
        let mgs: MGS<f32> = MGS::new(N);
        assert_eq!(mgs.dim(), N);
        assert_eq!(mgs.len(), 0);
    }

    fn test_append<A: Scalar + Lapack>(rtol: A::Real) {
        let mut mgs: MGS<A> = MGS::new(N);
        let a: Array2<A> = crate::generate::random((N, 3));
        dbg!(&a);
        for col in a.axis_iter(Axis(1)) {
            let res = mgs.append(col);
            dbg!(res);
        }
        let q = mgs.get_q();
        dbg!(&q);
        let r = mgs.get_r();
        dbg!(&r);

        dbg!(q.dot(&r));
        close_l2(&q.dot(&r), &a, rtol).unwrap();

        let qt: Array2<_> = conjugate(&q);
        dbg!(qt.dot(&q));
        close_l2(&qt.dot(&q), &Array2::eye(3), rtol).unwrap();
    }

    #[test]
    fn append() {
        test_append::<f32>(1e-5);
        test_append::<c32>(1e-5);
        test_append::<f64>(1e-9);
        test_append::<c64>(1e-9);
    }

    fn test_append_if<A: Scalar + Lapack>(rtol: A::Real) {
        let mut mgs: MGS<A> = MGS::new(N);
        let a: Array2<A> = crate::generate::random((N, 8));
        dbg!(&a);
        for col in a.axis_iter(Axis(1)) {
            match mgs.append_if(col, rtol) {
                Some(res) => {
                    dbg!(res);
                }
                None => break,
            }
        }
        let q = mgs.get_q();
        dbg!(&q);
        let r = mgs.get_r();
        dbg!(&r);

        dbg!(q.dot(&r));
        close_l2(&q.dot(&r), &a.slice(s![.., 0..N]), rtol).unwrap();

        let qt: Array2<_> = conjugate(&q);
        dbg!(qt.dot(&q));
        close_l2(&qt.dot(&q), &Array2::eye(N), rtol).unwrap();
    }

    #[test]
    fn append_if() {
        test_append_if::<f32>(1e-5);
        test_append_if::<c32>(1e-5);
        test_append_if::<f64>(1e-9);
        test_append_if::<c64>(1e-9);
    }
}
