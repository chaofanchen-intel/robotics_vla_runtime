# build：M\* Rust 组件的编译产物

M\* 的最新版（`external/mstar`，2c8ffde8）有两个 Rust 组件。两者都要从 `external/mstar/rust` 的源码编译，
编译产物不在 git 中跟踪（见 `.gitignore`）。这里保存的是可直接使用的副本与构建日志。

| 路径 | 内容 | 用途 |
|---|---|---|
| `mstar_rust/` | `mstar_rust` 扩展（pyo3）解包后的目录，含 `mstar_rust/*.so` 与 `mstar_rust-0.1.1.dist-info` | 与 wheel 内容相同，供查看；运行时用的是 `envs/mstar` 中已安装的副本 |
| `mstar_rust_wheel/mstar_rust-0.1.1-cp312-cp312-manylinux_2_38_x86_64.whl` | 同一扩展的 wheel | 安装到 `envs/mstar`：`envs/mstar/bin/python -m pip install --no-deps --force-reinstall <wheel>` |
| `bin/mstar-server` | Rust 前端的服务二进制（默认前端） | `src/backends/mstar.py` 通过 `--rust-frontend-bin` 使用（`scripts/run_probes.py --framework mstar-rust`） |
| `logs/` | 两次构建的完整日志 | 来源记录 |

## 源码哈希

`mstar_rust` 的图运行时在导入时检查：编译时记录的 `BUILD_SOURCE_HASH` 必须与当前 `external/mstar/rust` 的内容哈希一致，
否则报 `ImportError`。`mstar-server --source-hash` 同样输出源码哈希。当前值：

- `mstar_rust.BUILD_SOURCE_HASH` = `5651ef713b0c03f5`
- `mstar-server --source-hash` = `5651ef713b0c03f5 a9e6eb590375b03c`

哈希按内容计算，与文件修改时间无关，因此复制目录不会改变它。`rust/target`（构建缓存）不参与这个哈希；本目录没有复制它。

## 复现

```bash
cd /home/chaofan/workspaces/robotics_vla_runtime/external/mstar
# mstar_rust 扩展：wheel 输出到 build/mstar_rust_wheel/，再解包到 build/mstar_rust/
maturin build --release -m rust/Cargo.toml -o ../../build/mstar_rust_wheel
unzip -q -o ../../build/mstar_rust_wheel/mstar_rust-*.whl -d ../../build/mstar_rust
# 装入 envs/mstar（wheel 装入后，导入时检查的哈希与 external/mstar/rust 一致）
cd ../../ && envs/mstar/bin/python -m pip install --no-deps --force-reinstall build/mstar_rust_wheel/mstar_rust-*.whl
# mstar-server 二进制（需要能拉取 crates 或已有缓存）
cd external/mstar && cargo build --release --manifest-path rust/server/Cargo.toml
cp rust/target/release/mstar-server ../../build/bin/mstar-server
```

`envs/mstar` 中的 `mstar_rust` 是通过上面的 `pip install` 装入的，不再需要 `PYTHONPATH`。
重新编译后要再次执行 `pip install`，否则 `envs/mstar` 里仍是旧的扩展。

`CARGO_TARGET_DIR` 可以指向别处，只要最后把二进制复制到 `build/bin/`。`external/mstar/rust/target` 是构建缓存，不纳入本目录的副本。
`external/mstar` 的源码改动后需要重新构建，否则服务启动时会因哈希不一致而报错。
