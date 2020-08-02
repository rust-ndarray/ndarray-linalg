use criterion::*;
use ndarray::*;
use ndarray_linalg::*;

fn truncated_eigh_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("truncated_eigh");
    for &n in &[4, 8, 16, 32, 64, 128] {
        group.bench_with_input(BenchmarkId::new("C", n), &n, |b, &n| {
            let a: Array2<f64> = random((n, n));
            let a = a.t().dot(&a);
            b.iter(move || {
                let _result = TruncatedEig::new(a.clone(), TruncatedOrder::Largest).decompose(1);
            })
        });
        group.bench_with_input(BenchmarkId::new("F", n), &n, |b, &n| {
            let a: Array2<f64> = random((n, n).f());
            let a = a.t().dot(&a);
            b.iter(|| {
                let _result = TruncatedEig::new(a.clone(), TruncatedOrder::Largest).decompose(1);
            })
        });
    }
}

criterion_group!(truncated_eigh, truncated_eigh_small);
criterion_main!(truncated_eigh);
