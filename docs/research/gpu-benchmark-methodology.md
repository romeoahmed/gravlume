# 原生 GPU 基准方法

本文定义研究 benchmark 的测量与解释方法，不是用户性能承诺、跨平台结果或 production interface；实际 workload 以 benchmark source 为准，历史结果只进入对应决策账本。

**状态：当前方法。** 仓库只保留一个针对现行 production trace pipeline 的固定原生 GPU benchmark。历史 shader 变体、配对 runner、CSV artifact generator 与 `wgpu-profiler` 集成已经完成决策使命，不再作为永久代码维护；实验结果归档在 [GPU 测地线加速决策账本](gpu-geodesic-acceleration.md)。

## 1. 测量对象

固定 workload 为默认 Kerr surface scene、`1280×720`、full Cartesian KS、equatorial event 与
直接 bolometric transport。资源、pipeline 和 scene 在 Criterion 计时闭包外创建；每次迭代重新
编码并提交一张完整 candidate。该 workload 随 `surface-observable-v1` 取代旧 sky accelerator
workload，因此两者的绝对时间不能串成同一历史序列。

benchmark 复用 renderer 自己的 `GpuTimings`，以一对 timestamp 包围 production trace pass。
resolved GPU duration 通过 Criterion
`Bencher::iter_custom` 返回。命令编码、queue submit、等待和 readback 不计入该 duration，因此
这个指标用于比较 kernel throughput，不代表 resize-to-publish 或点击到像素的端到端延迟。

Criterion 明确把 `iter_custom` 定义为由被测 workload 自行提供总 `Duration` 的接口；wgpu
则规定 pass timestamp 由 `TIMESTAMP_QUERY` feature 提供，resolved tick 需要按
`Queue::get_timestamp_period` 缩放：

- [Criterion 0.8.2 `Bencher::iter_custom`](https://docs.rs/criterion/0.8.2/criterion/struct.Bencher.html#method.iter_custom)
- [wgpu 30 `TIMESTAMP_QUERY`](https://docs.rs/wgpu/30.0.1/wgpu/struct.Features.html#associatedconstant.TIMESTAMP_QUERY)
- [wgpu 30 `ComputePassTimestampWrites`](https://docs.rs/wgpu/30.0.1/wgpu/struct.ComputePassTimestampWrites.html)

## 2. 为什么不保留第二套 profiler

production 已有固定一对 query、单 readback buffer 和明确 submission 生命周期的 `GpuTimings`。
永久 benchmark 再引入 `wgpu-profiler` 会复制 query ownership、frame lifecycle 和错误面；而
一个 compute pass 的 descriptor 只能安装一组 timestamp writes。当前 benchmark 不需要
scope tree、动态 query pool 或多帧 profiler，因此直接复用 production abstraction 更小，
也保证测到的 pass 边界与运行时一致。

以后若某项实验确实需要细分 pass，可在短期研究分支使用 profiler；只有产生持续消费者后才
考虑把它提升为仓库依赖。

## 3. 同步与噪声边界

每个 trace submission 后等待该 submission，完成 timestamp map 后才开始下一次。这样不会把
多个 trace 排队造成的 queue depth 当作单帧性能，也不会让 readback buffer 在 GPU 使用期间
重新映射。Criterion 的具名 warm-up 阶段用于避开 pipeline/resource 冷启动。

Criterion 使用 flat sampling、30 samples、5 秒 warm-up、15 秒 measurement。它给出的区间是
sample mean 的统计区间，不是单帧 p95。Apple SoC 的 thermal/DVFS 漂移曾让同一 binary 的相邻
run 相差约 6%，所以：

- named baseline 只作历史回归信号，不单独决定 shader 取舍；
- 小优化需要反转顺序或同 submission 配对的临时实验，并复测符号；
- correctness/observable gate 必须先于 timing；
- 约 4% 的稳定、低维护收益可以保留，5% noise threshold 不是机械淘汰线。

临时 A/B 完成后，只把结论与必要数值写入研究报告；不保留没有生产消费者的 shader selector、
环境变量协议、CSV schema、bootstrap 实现或专用 example。

## 4. 运行与解释

```text
cargo bench -p gravlume-render --bench trace_gpu --features gpu-benchmarks --locked
```

benchmark feature 默认关闭，Criterion 不进入普通 runtime dependency closure。报告至少记录
revision/dirty state、OS、adapter/backend、power mode、scene、extent、build profile、样本配置、
mean/CI 和 output gate。实际 GPU memory peak 只能来自具名原生工具；`width × height × 8` 只是
logical FP16 texel payload，不能冒充 driver allocation。

端到端交互性能另用 native invalidation→atomic-publish latency、batch timings 和 smoke 验证。
永久 Criterion benchmark 的职责只有一个：以同一 production pipeline 给出可重复的 GPU
throughput regression signal。
