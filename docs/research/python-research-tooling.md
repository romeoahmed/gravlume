# Python 研究工具链

本文定义 `docs/research/scripts` 的开发、依赖解析、执行与测试约定；它只约束可复算研究工具，不定义 production 行为或 Cargo runtime dependency closure。各项数学结论仍由对应研究记录说明，精确依赖快照以 [`uv.lock`](scripts/uv.lock) 为唯一来源。

## 设计结论

研究代码以 Python 3.14 为最低版本，采用普通 `src` package 和单一 `gravlume-research` CLI。模块边界按证明对象划分；CLI 只负责选择检查，不把内部函数扩展成通用 scientific API。

依赖职责保持窄而明确：

| 依赖 | 职责 |
| --- | --- |
| SymPy | exact symbolic algebra、polynomial identity 与 expression equivalence |
| mpmath | arbitrary-precision quadrature、root finding、precision doubling 与数值 oracle |
| NumPy | IEEE 754 binary32 cast、`nextafter`、`uint32` view 与浮点错误处理 scope |
| pytest | test collection、`parametrize`、`raises` 与显式 `approx` tolerance |
| Hypothesis | `given` strategy、显式 boundary `example` 与可复现的生成式性质测试 |
| Ruff | 面向 Python 3.14 的 lint、import ordering 与 format gate |

源码直接导入 `mpmath`，因此它仍是 direct dependency；但 `pyproject.toml` 不独立钉住其版本，uv 必须同时满足 SymPy 发布元数据对 mpmath 的约束。这样不会依赖偶然的 transitive import，也不会绕开 SymPy/mpmath 的兼容关系。

截至 2026-08-28，[SymPy 1.14.0 的 PyPI wheel metadata](https://pypi.org/pypi/sympy/1.14.0/json) 声明 `Requires-Dist: mpmath<1.4,>=1.1.0`，而 [PyPI 上最新正式版 mpmath 1.4.1](https://pypi.org/pypi/mpmath/1.4.1/json) 已越过该上界。因此当前 lock 选择 mpmath 1.3.0 是约束交集的结果；额外要求 `mpmath>=1.4.1` 会使解析无解，dependency override 则会忽略发布方约束，不能在没有独立兼容性证据时用来“强升”。这里的版本只解释本次解析证据，精确快照仍以 `uv.lock` 为准。

没有引入 SciPy、Pydantic 或 Typer：当前没有矩阵/优化 runtime、外部数据模型或复杂 CLI seam；为这些假设增加框架只会扩大依赖和 public surface。`argparse`、frozen dataclass、SymPy 与 mpmath 已覆盖真实边界。

## 解析与升级策略

[`pyproject.toml`](scripts/pyproject.toml) 表达 Python 基线、直接依赖和稳定版政策，不复制当前解析版本；[`uv.lock`](scripts/uv.lock) 保存跨平台精确快照。正常升级命令是：

```text
uv lock --project docs/research/scripts --upgrade
```

已有 lock 时，uv 默认偏好既有解析；`--upgrade` 才会重新选择约束内的最新兼容版本。`prerelease = "disallow"` 排除全部 prerelease candidate，而不是仅降低其优先级。升级后必须审查 lock diff，并运行本文全部检查；不得为追逐单个包而覆盖其依赖方声明的兼容范围。

Build backend 使用 uv 官方建议的同一 minor compatibility range。它是由 `[build-system].requires` 控制的隔离构建依赖，不属于项目运行依赖图：`uv build` 会优先复用版本兼容的 bundled backend，否则解析该范围内的 `uv_build`；其他 build frontend 通常选择范围内最新版本。因此不能把 build backend 的实际版本表述为由项目 `uv.lock` 固定。

## 统一命令

可用检查名：

| CLI 参数 | 研究对象 |
| --- | --- |
| `bl-mino-surface` | 具名 BL/Mino surface 与 outer-edge witness |
| `kerr-capture` | Kerr quartic、Bernstein 与 binary32 interval |
| `kerr-schild-map` | Kerr–Schild ↔ Boyer–Lindquist/Mino seam |
| `kerr-schild-rhs` | Kerr–Schild Hamiltonian/RHS 约化 |
| `mino-step` | RK4 与 cubic Hermite 局部阶数 |
| `scalar-transport` | invariant transfer、Planck bands 与 LUT oracle |

从仓库根目录复算单项：

```text
uv run --isolated --project docs/research/scripts --locked \
  gravlume-research <check>
```

完整 Python gate 必须逐项执行六个 scientific witness；pytest 只验证可独立表述的行为与性质，不能替代这些端到端复算：

```text
uv run --isolated --project docs/research/scripts --locked \
  gravlume-research bl-mino-surface
uv run --isolated --project docs/research/scripts --locked \
  gravlume-research kerr-capture
uv run --isolated --project docs/research/scripts --locked \
  gravlume-research kerr-schild-map
uv run --isolated --project docs/research/scripts --locked \
  gravlume-research kerr-schild-rhs
uv run --isolated --project docs/research/scripts --locked \
  gravlume-research mino-step
uv run --isolated --project docs/research/scripts --locked \
  gravlume-research scalar-transport
uv run --isolated --project docs/research/scripts --locked \
  pytest docs/research/scripts/tests
uv run --isolated --project docs/research/scripts --locked \
  ruff format --check docs/research/scripts/src docs/research/scripts/tests
uv run --isolated --project docs/research/scripts --locked \
  ruff check docs/research/scripts/src docs/research/scripts/tests
```

Named scientific cases 使用 deterministic example 和独立高精度 expectation；性质测试只覆盖可表述的不变量，不生成“期望物理答案”。离散 identity 与 SymPy exact equality 使用严格相等；浮点 observable 使用 `pytest.approx` 并显式给出 `rel`/`abs` budget。mpmath 常量必须在目标 `workdps` 内由十进制文本构造，避免先按默认精度舍入。

## 官方依据

- [uv project layout](https://docs.astral.sh/uv/concepts/projects/layout/) 与 [uv build backend](https://docs.astral.sh/uv/concepts/build-backend/)：`src` package、entry point 和 build compatibility range；
- [uv locking and upgrading](https://docs.astral.sh/uv/concepts/projects/sync/) 与 [dependency resolution](https://docs.astral.sh/uv/concepts/resolution/)：抽象要求、精确 lock、既有版本偏好、升级、prerelease 与 override 语义；
- [pytest `approx`](https://docs.pytest.org/en/stable/reference/reference.html#pytest-approx)、[`raises`](https://docs.pytest.org/en/stable/reference/reference.html#pytest-raises) 与 [parametrization](https://docs.pytest.org/en/stable/how-to/parametrize.html)：数值、异常和参数化 assertion；
- [Hypothesis strategy adaptation](https://hypothesis.readthedocs.io/en/latest/tutorial/adapting-strategies.html)、[`example`](https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.example) 与 [settings](https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.settings)：由输入域和具名边界构造性质测试；
- [NumPy scalar types](https://numpy.org/doc/stable/user/basics.types.html)、[`ndarray.view`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.view.html)、[`nextafter`](https://numpy.org/doc/stable/reference/generated/numpy.nextafter.html) 与 [`errstate`](https://numpy.org/doc/stable/reference/generated/numpy.errstate.html)：固定宽度 binary32 和浮点错误处理 scope；
- [SymPy best practices](https://docs.sympy.org/latest/explanation/best-practices.html) 与 [mpmath documentation](https://mpmath.org/doc/1.3.0/)：exact construction、符号/数值边界和 arbitrary precision；
- [Ruff configuration](https://docs.astral.sh/ruff/configuration/) 与 [formatter](https://docs.astral.sh/ruff/formatter/)：单一 lint/format 配置。
