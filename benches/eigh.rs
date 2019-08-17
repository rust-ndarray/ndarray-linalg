#[macro_use]
extern crate criterion;

use criterion::Criterion;
use ndarray::*;
use ndarray_linalg::*;

macro_rules! impl_eigh {
    ($n:expr) => {
        paste::item! {
            fn [<eigh $n>](c: &mut Criterion) {
                c.bench_function(&format!("eigh{}", $n), |b| {
                    let a: Array2<f64> = random(($n, $n));
                    b.iter(|| {
                        let (_e, _vecs) = a.eigh(UPLO::Upper).unwrap();
                    })
                });
                c.bench_function(&format!("eigh{}_t", $n), |b| {
                    let a: Array2<f64> = random(($n, $n).f());
                    b.iter(|| {
                        let (_e, _vecs) = a.eigh(UPLO::Upper).unwrap();
                    })
                });
            }
        }
    };
}

impl_eigh!(4);
impl_eigh!(8);
impl_eigh!(16);
impl_eigh!(32);
impl_eigh!(64);
impl_eigh!(128);

criterion_group!(eigh, eigh4, eigh8, eigh16, eigh32, eigh64, eigh128);
criterion_main!(eigh);
