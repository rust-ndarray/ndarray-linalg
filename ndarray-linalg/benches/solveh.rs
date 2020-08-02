use criterion::*;
use ndarray::*;
use ndarray_linalg::*;

fn solveh_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("solveh");
    for &n in &[4, 8, 16, 32, 64, 128] {
        group.bench_with_input(BenchmarkId::new("factorizeh/C", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n));
            b.iter(|| {
                let _lu = a.factorizeh().unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("factorizeh/F", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n).f());
            b.iter(|| {
                let _lu = a.factorizeh().unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("invh/C", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n));
            b.iter(|| {
                let _inv = a.invh().unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("invh/F", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n).f());
            b.iter(|| {
                let _inv = a.invh().unwrap();
            })
        });
    }
}

criterion_group!(solveh, solveh_small);
criterion_main!(solveh);
