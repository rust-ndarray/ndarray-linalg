use criterion::*;
use ndarray::*;
use ndarray_linalg::*;

fn factorize_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorize");
    for &n in &[4, 8, 16, 32, 64, 128] {
        group.bench_with_input(BenchmarkId::new("C", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n));
            b.iter(|| {
                let _lu = a.factorize().unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("F", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n).f());
            b.iter(|| {
                let _lu = a.factorize().unwrap();
            })
        });
    }
}

criterion_group!(factorize, factorize_small);
criterion_main!(factorize);
