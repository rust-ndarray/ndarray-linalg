use crate::{generate::*, norm::Norm, types::*};
use ndarray::*;

#[derive(Debug, Clone)]
pub struct MGS<A> {
    dim: usize,
    q: Vec<Array1<A>>,
    r: Vec<Array1<A>>,
}

impl<A: Scalar + Lapack> MGS<A> {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            q: Vec::new(),
            r: Vec::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.q.len()
    }

    pub fn append<S>(&mut self, a: ArrayBase<S, Ix1>) -> A::Real
    where
        S: Data<Elem = A>,
    {
        assert_eq!(a.len(), self.dim());
        let mut a = a.into_owned();
        let mut coef = Array1::zeros(self.len() + 1);
        for i in 0..self.len() {
            let q = &self.q[i];
            let c = a.dot(q);
            azip!(mut a, q (q) in { *a = *a - c * q } );
            coef[i] = c;
        }
        let nrm = a.norm_l2();
        coef[self.len()] = A::from_real(nrm);
        self.r.push(coef);
        azip!(mut a in { *a = *a / A::from_real(nrm) });
        self.q.push(a);
        nrm
    }

    pub fn get_q(&self) -> Array2<A> {
        hstack(&self.q).unwrap()
    }

    pub fn get_r(&self) -> Array2<A> {
        let len = self.len();
        let mut r = Array2::zeros((len, len));
        for i in 0..len {
            for j in 0..=i {
                r[(j, i)] = self.r[i][j];
            }
        }
        r
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

    #[test]
    fn append_random() {
        let mut mgs: MGS<f64> = MGS::new(N);
        let a: Array2<f64> = random((N, 3));
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
        close_l2(&q.dot(&r), &a, 1e-9).unwrap();

        dbg!(q.t().dot(&q));
        close_l2(&q.t().dot(&q), &Array2::eye(3), 1e-9).unwrap();
    }
}
