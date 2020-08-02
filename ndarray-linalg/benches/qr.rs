use criterion::*;
use ndarray::*;
use ndarray_linalg::*;

fn qr_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr");
    for &n in &[4, 8, 16, 32, 64, 128] {
        group.bench_with_input(BenchmarkId::new("C", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n));
            b.iter(|| {
                let _result = a.qr().unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("F", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n).f());
            b.iter(|| {
                let _result = a.qr().unwrap();
            })
        });
    }
}

criterion_group!(qr, qr_small);
criterion_main!(qr);
