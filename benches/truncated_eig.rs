#[macro_use]
extern crate criterion;

use criterion::Criterion;
use ndarray::*;
use ndarray_linalg::*;

macro_rules! impl_teig {
    ($n:expr) => {
        paste::item! {
            fn [<teig $n>](c: &mut Criterion) {
                c.bench_function(&format!("truncated_eig{}", $n), |b| {
                    let a: Array2<f64> = random(($n, $n));
                    let a = a.t().dot(&a);

                    b.iter(move || {
                        let _result = TruncatedEig::new(a.clone(), TruncatedOrder::Largest).decompose(1);
                    })
                });
                c.bench_function(&format!("truncated_eig{}_t", $n), |b| {
                    let a: Array2<f64> = random(($n, $n).f());
                    let a = a.t().dot(&a);

                    b.iter(|| {
                        let _result = TruncatedEig::new(a.clone(), TruncatedOrder::Largest).decompose(1);
                    })
                });
            }
        }
    };
}

impl_teig!(4);
impl_teig!(8);
impl_teig!(16);
impl_teig!(32);
impl_teig!(64);
impl_teig!(128);

criterion_group!(teig, teig4, teig8, teig16, teig32, teig64, teig128);
criterion_main!(teig);
