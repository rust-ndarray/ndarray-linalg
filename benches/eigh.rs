use criterion::*;
use ndarray::*;
use ndarray_linalg::*;

fn eigh_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigh");

    for &n in &[4, 8, 16, 32, 64, 128] {
        group.bench_function(&format!("eigh{:03}", n), |b| {
            let a: Array2<f64> = random((n, n));
            b.iter(|| {
                let (_e, _vecs) = a.eigh(UPLO::Upper).unwrap();
            })
        });
        group.bench_function(&format!("eigh{:03}_t", n), |b| {
            let a: Array2<f64> = random((n, n).f());
            b.iter(|| {
                let (_e, _vecs) = a.eigh(UPLO::Upper).unwrap();
            })
        });
    }
}

criterion_group!(eigh, eigh_small);
criterion_main!(eigh);
