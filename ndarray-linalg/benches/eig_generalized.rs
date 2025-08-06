use criterion::*;
use ndarray::*;
use ndarray_linalg::*;

fn eig_generalized_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("eig");
    for &n in &[4, 8, 16, 32, 64, 128] {
        group.bench_with_input(BenchmarkId::new("vecs/C/r", n), &n, |c, n| {
            let a: Array2<f64> = random((*n, *n));
            let b: Array2<f64> = random((*n, *n));
            c.iter(|| {
                let (_e, _vecs) = (a.clone(), b.clone()).eig_generalized(None).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("vecs/F/r", n), &n, |c, n| {
            let a: Array2<f64> = random((*n, *n).f());
            let b: Array2<f64> = random((*n, *n).f());
            c.iter(|| {
                let (_e, _vecs) = (a.clone(), b.clone()).eig_generalized(None).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("vecs/C/c", n), &n, |c, n| {
            let a: Array2<c64> = random((*n, *n));
            let b: Array2<c64> = random((*n, *n));
            c.iter(|| {
                let (_e, _vecs) = (a.clone(), b.clone()).eig_generalized(None).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("vecs/F/c", n), &n, |c, n| {
            let a: Array2<c64> = random((*n, *n).f());
            let b: Array2<c64> = random((*n, *n).f());
            c.iter(|| {
                let (_e, _vecs) = (a.clone(), b.clone()).eig_generalized(None).unwrap();
            })
        });
    }
}

criterion_group!(eig, eig_generalized_small);
criterion_main!(eig);
