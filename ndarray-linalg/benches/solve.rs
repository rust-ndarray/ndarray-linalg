use criterion::*;
use ndarray::*;
use ndarray_linalg::*;

fn solve_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve");
    for &n in &[4, 8, 16, 32, 64, 128] {
        group.bench_with_input(BenchmarkId::new("factorize/C", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n));
            b.iter(|| {
                let _lu = a.factorize().unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("factorize/F", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n).f());
            b.iter(|| {
                let _lu = a.factorize().unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("inv/C", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n));
            b.iter(|| {
                let _inv = a.inv().unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("inv/F", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n).f());
            b.iter(|| {
                let _inv = a.inv().unwrap();
            })
        });
    }
}

criterion_group!(solve, solve_small);
criterion_main!(solve);
