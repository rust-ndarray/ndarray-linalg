use criterion::*;
use ndarray::*;
use ndarray_linalg::*;

fn svd_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd");
    for &n in &[4, 8, 16, 32, 64, 128] {
        group.bench_with_input(BenchmarkId::new("C", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n));
            b.iter(|| {
                let _ = a.svd(false, false).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("F", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n).f());
            b.iter(|| {
                let _ = a.svd(false, false).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("u/C", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n));
            b.iter(|| {
                let _ = a.svd(true, false).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("u/F", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n).f());
            b.iter(|| {
                let _ = a.svd(true, false).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("vt/C", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n));
            b.iter(|| {
                let _ = a.svd(false, true).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("vt/F", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n).f());
            b.iter(|| {
                let _ = a.svd(false, true).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("uvt/C", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n));
            b.iter(|| {
                let _ = a.svd(false, true).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("uvt/F", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n).f());
            b.iter(|| {
                let _ = a.svd(false, true).unwrap();
            })
        });
    }
}

fn svddc_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("svddc");
    for &n in &[4, 8, 16, 32, 64, 128] {
        group.bench_with_input(BenchmarkId::new("C", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n));
            b.iter(|| {
                let _ = a.svddc(UVTFlag::None).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("F", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n).f());
            b.iter(|| {
                let _ = a.svddc(UVTFlag::None).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("some/C", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n));
            b.iter(|| {
                let _ = a.svddc(UVTFlag::Some).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("some/F", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n).f());
            b.iter(|| {
                let _ = a.svddc(UVTFlag::Some).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("full/C", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n));
            b.iter(|| {
                let _ = a.svddc(UVTFlag::Full).unwrap();
            })
        });
        group.bench_with_input(BenchmarkId::new("full/F", n), &n, |b, n| {
            let a: Array2<f64> = random((*n, *n).f());
            b.iter(|| {
                let _ = a.svddc(UVTFlag::Full).unwrap();
            })
        });
    }
}

criterion_group!(svd, svd_small, svddc_small);
criterion_main!(svd);
