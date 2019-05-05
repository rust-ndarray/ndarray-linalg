use super::*;
use crate::{generate::*, inner::*, norm::Norm};

/// Iterative orthogonalizer using modified Gram-Schmit procedure
#[derive(Debug, Clone)]
pub struct MGS<A> {
    /// Dimension of base space
    dim: usize,
    /// Basis of spanned space
    q: Vec<Array1<A>>,
}

impl<A: Scalar> MGS<A> {
    fn ortho<S>(&self, a: &mut ArrayBase<S, Ix1>) -> Array1<A>
    where
        A: Lapack,
        S: DataMut<Elem = A>,
    {
        assert_eq!(a.len(), self.dim);
        let mut coef = Array1::zeros(self.q.len() + 1);
        for i in 0..self.q.len() {
            let q = &self.q[i];
            let c = q.inner(&a);
            azip!(mut a (&mut *a), q (q) in { *a = *a - c * q } );
            coef[i] = c;
        }
        let nrm = a.norm_l2();
        coef[self.q.len()] = A::from(nrm).unwrap();
        coef
    }
}

impl<A: Scalar + Lapack> Orthogonalizer for MGS<A> {
    type Elem = A;

    fn new(dim: usize) -> Self {
        Self { dim, q: Vec::new() }
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.q.len()
    }

    fn orthogonalize<S>(&self, a: &mut ArrayBase<S, Ix1>) -> A::Real
    where
        S: DataMut<Elem = A>,
    {
        let coef = self.ortho(a);
        // Write coefficients into `a`
        azip!(mut a (a.slice_mut(s![0..self.len()])), coef in { *a = coef });
        // 0-fill for remaining
        azip!(mut a (a.slice_mut(s![self.len()..])) in { *a = A::zero() });
        coef[self.len()].re()
    }

    fn append<S>(&mut self, mut a: ArrayBase<S, Ix1>, rtol: A::Real) -> Result<Array1<A>, Array1<A>>
    where
        S: DataMut<Elem = A>,
    {
        let coef = self.ortho(&mut a);
        let nrm = coef[self.len()].re();
        if nrm < rtol {
            // Linearly dependent
            return Err(coef);
        }
        azip!(mut a in { *a = *a / A::from_real(nrm) });
        self.q.push(a.into_owned());
        Ok(coef)
    }

    fn get_q(&self) -> Q<A> {
        hstack(&self.q).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert::*;

    #[test]
    fn mgs_append() {
        let mut mgs = MGS::new(3);
        let coef = mgs.append(array![0.0, 1.0, 0.0], 1e-9).unwrap();
        close_l2(&coef, &array![1.0], 1e-9).unwrap();

        let coef = mgs.append(array![1.0, 1.0, 0.0], 1e-9).unwrap();
        close_l2(&coef, &array![1.0, 1.0], 1e-9).unwrap();

        assert!(mgs.append(array![1.0, 2.0, 0.0], 1e-9).is_err());

        if let Err(coef) = mgs.append(array![1.0, 2.0, 0.0], 1e-9) {
            close_l2(&coef, &array![2.0, 1.0, 0.0], 1e-9).unwrap();
        }
    }

}
