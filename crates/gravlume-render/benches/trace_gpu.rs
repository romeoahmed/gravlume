use criterion::{criterion_group, criterion_main};
use gravlume_render::benchmark;

criterion_group!(benches, benchmark::register);
criterion_main!(benches);
