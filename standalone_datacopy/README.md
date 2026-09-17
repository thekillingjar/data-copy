# 三种分块转置搬运：256KiB UB / 768KiB 输入

本实验比较同一个输入 `A[M,N]` 到计算所需的紧密布局 `B[N,M]`，
所有路径满足 `B[c,r] = A[r,c]`，dtype 为 int32，单 AIV 核。
使用 DataCopyPad，不使用 NDDMA；旧的 padded 元素布局已移除。

没有指定后续计算算子，因此当前测量**转置、搬运和 tile 同步开销**，不包含实际计算，
也没有双缓冲或搬运/计算重叠。结果用于选取搬运方案，不是完整计算算子的性能结论。

## 三种路径

默认输入 `256×768×4 = 786432B = 768KiB`，是256KiB UB的三倍。
默认保留16KiB，单缓冲最大240KiB；该保留量是实验预算，不是实际计算工作区的估算。

| 命令 mode | UB tile | 默认 tile 数 | 操作 |
|---|---|---|---|
| small | 8KiB：输入256×8，UB8×256 | 96 | 每个tile按列收集，Compact转置为紧密UB |
| large | 240KiB：输入256×240，UB240×256 | 4 | 与small相同，增大tile；最后48列为48KiB |
| pretranspose | 两个阶段均240KiB | 4 + 4 | 完整转置到GM临时B，再从B连续搬入UB |

这里的“大tile合理”指利用较多UB、减少tile边界，**并不保证240KiB最优**。
尤其当前small/large每个输入列仍调用一次Compact，两者均为768次搬入API，
变化的是tile循环/同步次数（96 vs 4）、UB地址范围和工作集。
不要把tile数减少24倍解释成DMA指令减少24倍或性能必然提高24倍。

## 地址映射与同步

只沿输入N轴切块，输入M轴完整保留。每次tile起点c0，实际宽度n：

```text
ub[c*M+r] = A[r*N+c0+c]     0 <= c < n, 0 <= r < M
```

每个输入列使用 `blockCount=M, blockLen=4, srcStride=(N-1)*4`，
Compact紧密写到 `ub[c*M]`。要求M为8的倍数，因此每一输出行都32B对齐且无padding。
每个tile末尾的MTE2 barrier保证搬入结束，再复用UB；未实现计算消费者。

预转置阶段用相同large tile形成UB转置块后，一次写出 `n*M*4` 字节到 `B[c0*M]`。
由于tile覆盖B完整尾轴，该块在GM上也完全连续。
写出前执行MTE2→MTE3同步，写出后执行MTE3→MTE2同步，避免下一块覆盖尚未写完的UB。
完整B生成后，同一个stream启动第二个kernel，每个tile只用一次连续GM→UB搬运。

不支持M尾轴分块或非8倍数M；N末尾不足一个tile已处理。
M限制8..4088，N限制1..65535，总输入不超过64MiB。
不使用多核；B不是原地转置，额外占用完整输入大小的GM。

## 构建

将本目录复制到具有CANN ASC编译器、dav-3510/950 simulator的Linux环境。
先source实际安装的CANN环境脚本，确保 `ASCEND_HOME_PATH`、ASC CMake package及 `msprof` 可用。

```bash
cd standalone_datacopy
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # 按实际安装位置修改
export SOC_VERSION=Ascend950PR_9599              # 按已安装simulator的真实名称修改
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/$(uname -m)-linux/simulator/$SOC_VERSION/lib:${LD_LIBRARY_PATH:-}"

cmake -S . -B build -DNPU_ARCH=dav-3510 -DRUN_MODE=sim -DSOC_VERSION="$SOC_VERSION"
cmake --build build -j8
```

`dav-3510` 是编译架构，不是 `--soc-version` 的替代值。
本CMake沿用HW_GE_ATT的ASC语言构建方式，不支持普通macOS编译器或CPU调试模式。

## 一键校验并采集 cycles

如果msprof不在PATH中，或PATH指向另一套安装，显式指定可执行文件：

```bash
export MSPROF_BIN="$ASCEND_HOME_PATH/tools/profiler/bin/msprof"
test -x "$MSPROF_BIN"
"$MSPROF_BIN" op simulator --help
```

请按实际安装位置修改该路径。脚本打印实际使用的msprof路径，并写入每次运行的command.json。
不传 `--msprof` 时从PATH查找。指定路径只解决程序选择，不保证解决所有校验失败；
如果仍缺少 `verification=PASS`，需要继续检查该次run.log。

```bash
python3 profile.py \
  --binary build/demo_datacopy \
  --msprof "$MSPROF_BIN" \
  --soc-version "$SOC_VERSION" \
  --rows 256 --cols 768 \
  --small-kib 8 --large-kib 240 \
  --trials 3 --device 0 \
  --output results_8_240
```

输出目录必须不存在。脚本共运行3次正确性校验和9次性能采集：

1. 每条路径STORE=1单独执行，把所有tile写回诊断GM输出，逐元素检查转置结果。
2. 性能采集STORE=0，关闭诊断写回。预转置阶段写GM临时B始终保留，绝不能关闭。
3. 每次进程执行一遍完整输入；pretranspose包含先后执行的两个kernel。
4. 三个trial轮换方案顺序。L2 cache hint设为disable；不把这当作全部硬件缓存清空保证。
5. 每个kernel只匹配一条 `aiv_total_cycles`，缺列、重复匹配或无有效数字会报错。

也可以手动运行（末尾分别为TILE_KIB、STORE、DEVICE_ID）：

```bash
# 正确性检查，应输出 verification=PASS
msprof op simulator --soc-version="$SOC_VERSION" --output=check_small \
  ./build/demo_datacopy small 256 768 8 1 0

# 正式性能采集
msprof op simulator --soc-version="$SOC_VERSION" --output=perf_small \
  ./build/demo_datacopy small 256 768 8 0 0
msprof op simulator --soc-version="$SOC_VERSION" --output=perf_large \
  ./build/demo_datacopy large 256 768 240 0 0
msprof op simulator --soc-version="$SOC_VERSION" --output=perf_pretranspose \
  ./build/demo_datacopy pretranspose 256 768 240 0 0
```

## 如何比较

在 `results_8_240/summary.json` 中查看三个方案的 `total_cycles` 中位数及min/max：

```text
small.total_cycles        = datacopy_small.aiv_total_cycles
large.total_cycles        = datacopy_large.aiv_total_cycles
pretranspose.total_cycles = datacopy_pretranspose.aiv_total_cycles
                          + datacopy_contiguous.aiv_total_cycles
```

**第三项必须计入预转置成本。** `load_cycles` 仅显示GM转换后搬入有多快，不能单独当成方案三总成本。
`small_over_large > 1` 表示large更快；`pretranspose_over_large > 1` 表示完整预转置方案更慢。

这些是单核各kernel的执行cycles之和，包括kernel初始化及同步，
不包括两个kernel间的调度间隙、host开销和内存分配，并非端到端墙钟时间。
目前按一次使用计费，没有预转置复用摊销，也不使用旧版repeat差分。

输出文件：

- `config.json`：shape、预算、实际tile形状/数量及运行参数。
- `raw.csv`：各trial的预转置cycles、搬入cycles与总cycles。
- `summary.json`：汇总与性能比值。
- 各运行子目录中的 `command.json`、`run.log`、`prof/`：原始命令、日志、profiler数据。

重新解析已有结果：使用相同命令与全部相同参数，追加 `--analyze-only`。

扫描预算，寻找实际合适的tile，而不是预先认定最大tile最好：

```bash
for kib in 32 64 128 192 240; do
  python3 profile.py --binary build/demo_datacopy --soc-version "$SOC_VERSION" \
    --rows 256 --cols 768 --small-kib 8 --large-kib "$kib" \
    --trials 3 --output "results_8_${kib}"
done
```

## 验证状态

本机macOS，没有ASC/msprof，**未完成目标编译、simulator正确性验证或真实cycles采集**。
已通过 `python3 -m unittest -v test_address_model.py`：三路径地址模型、边界tile、
预算拒绝、CSV解析和模拟采集流程测试。测试使用的合成cycles不代表性能测量。

## 接口参考

- [DataCopyPad GMToUB](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_GMToUB.html)：Compact、stride及地址对齐约束。
- [msprof simulator官方文档](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_simulator_user_guide.md)：采集入口。
