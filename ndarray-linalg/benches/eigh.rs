use criterion::*;
use ndarray::*;
use ndarray_linalg::*;

fn eigh_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigh");
    for &n in &[4, 8, 16, 32, 64, 128] {
        group.bench_with_input(BenchmarkId::new("vals/C", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n));
            b.iter(|| {
                let _result = a.eigvalsh(UPLO::Upper).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("vals/F", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n).f());
            b.iter(|| {
                let _result = a.eigvalsh(UPLO::Upper).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("vecs/C", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n));
            b.iter(|| {
                let _result = a.eigvalsh(UPLO::Upper).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("vecs/F", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n).f());
            b.iter(|| {
                let _result = a.eigvalsh(UPLO::Upper).unwrap();
            })
        });
    }
}

criterion_group!(eigh, eigh_small);
criterion_main!(eigh);
