//! Moore-Penrose pseudo-inverse of a Matrices
//!
//! [](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.9-The-Moore-Penrose-Pseudoinverse/)

use crate::{error::*, svd::SVDInplace, types::*};
use ndarray::*;
use num_traits::Float;

/// pseudo-inverse of a matrix reference
pub trait Pinv {
    type E;
    type C;
    fn pinv(&self, threshold: Option<Self::E>) -> Result<Self::C>;
}

/// pseudo-inverse
pub trait PInvInto {
    type E;
    type C;
    fn pinv_into(self, rcond: Option<Self::E>) -> Result<Self::C>;
}

/// pseudo-inverse for a mutable reference of a matrix
pub trait PInvInplace {
    type E;
    type C;
    fn pinv_inplace(&mut self, rcond: Option<Self::E>) -> Result<Self::C>;
}

impl<A, S> PInvInto for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type E = A::Real;
    type C = Array2<A>;

    fn pinv_into(mut self, rcond: Option<Self::E>) -> Result<Self::C> {
        self.pinv_inplace(rcond)
    }
}

impl<A, S> Pinv for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: Data<Elem = A>,
{
    type E = A::Real;
    type C = Array2<A>;

    fn pinv(&self, rcond: Option<Self::E>) -> Result<Self::C> {
        let a = self.to_owned();
        a.pinv_into(rcond)
    }
}

impl<A, S> PInvInplace for ArrayBase<S, Ix2>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
{
    type E = A::Real;
    type C = Array2<A>;

    fn pinv_inplace(&mut self, rcond: Option<Self::E>) -> Result<Self::C> {
        if let (Some(u), s, Some(v_h)) = self.svd_inplace(true, true)? {
            // threshold = ε⋅max(m, n)⋅max(Σ)
            // NumPy defaults rcond to 1e-15 which is about 10 * f64 machine epsilon
            let rcond = rcond.unwrap_or_else(|| {
                let (n, m) = self.dim();
                Self::E::epsilon() * Self::E::real(n.max(m))
            });
            let threshold = rcond * s[0];

            // Determine how many singular values to keep and compute the
            // values of `V Σ⁺` (up to `num_keep` columns).
            let (num_keep, v_s_inv) = {
                let mut v_h_t = v_h.reversed_axes();
                let mut num_keep = 0;
                for (&sing_val, mut v_h_t_col) in s.iter().zip(v_h_t.columns_mut()) {
                    if sing_val > threshold {
                        let sing_val_recip = sing_val.recip();
                        v_h_t_col.map_inplace(|v_h_t| {
                            *v_h_t = A::from_real(sing_val_recip) * v_h_t.conj()
                        });
                        num_keep += 1;
                    } else {
                        /*
                        if sing_val != Self::E::real(0.0) {
                            panic!(
                                "for {:#?} singular value {:?} smaller then threshold {:?}",
                                &self, &sing_val, &threshold
                            );
                        }
                        */
                        break;
                    }
                }
                v_h_t.slice_axis_inplace(Axis(1), Slice::from(..num_keep));
                (num_keep, v_h_t)
            };

            // Compute `U^H` (up to `num_keep` rows).
            let u_h = {
                let mut u_t = u.reversed_axes();
                u_t.slice_axis_inplace(Axis(0), Slice::from(..num_keep));
                u_t.map_inplace(|x| *x = x.conj());
                u_t
            };

            Ok(v_s_inv.dot(&u_h))
        } else {
            unreachable!()
        }
    }
}
