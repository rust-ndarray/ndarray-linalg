use super::*;

pub struct Arnoldi<A, S, F, Ortho>
where
    A: Scalar,
    S: DataMut<Elem = A>,
    F: Fn(&mut ArrayBase<S, Ix1>),
    Ortho: Orthogonalizer<Elem = A>,
{
    a: F,
    v: ArrayBase<S, Ix1>,
    ortho: Ortho,
}

impl<A, S, F, Ortho> Arnoldi<A, S, F, Ortho>
where
    A: Scalar,
    S: DataMut<Elem = A>,
    F: Fn(&mut ArrayBase<S, Ix1>),
    Ortho: Orthogonalizer<Elem = A>,
{
    pub fn new(a: F, v: ArrayBase<S, Ix1>, ortho: Ortho) -> Self {
        Arnoldi { a, v, ortho }
    }
}

impl<A, S, F, Ortho> Iterator for Arnoldi<A, S, F, Ortho>
where
    A: Scalar,
    S: DataMut<Elem = A> + DataClone,
    F: Fn(&mut ArrayBase<S, Ix1>),
    Ortho: Orthogonalizer<Elem = A>,
{
    type Item = Array1<A>;

    fn next(&mut self) -> Option<Self::Item> {
        (self.a)(&mut self.v);
        match self.ortho.div_append(&mut self.v) {
            AppendResult::Added(coef) => {
                let norm = coef[coef.len() - 1].abs();
                azip!(mut a(&mut self.v) in { *a = a.div_real(norm) });
                Some(coef)
            }
            AppendResult::Dependent(_) => None,
        }
    }
}
