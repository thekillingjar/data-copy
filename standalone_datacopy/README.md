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

## 构建：与 HW_GE_ATT/NDDMA 相同的设备采集路径

已核对参考工程：

- `NDDMA/Modeling/Data_collection/round2/scripts/run_round2_collection.py` 中 `build_msprof_command`：
  使用 `msprof --output=... --application="..."`。
- `NDDMA/Modeling/Data_collection/common/executables/standalone_nddma/analyze_profiling_with_params.py`：
  MTE2 cycles 使用 `aiv_total_cycles × MTE2 ratio`。

本脚本不再调用 `msprof op simulator`，不会主动进入依赖msopprof的子命令。
之前据此报错判断CANN缺少组件不足以解释NDDMA能运行的环境；这里的问题是选错了采集入口。

在运行NDDMA的同一台昇腾机器、同一套CANN环境中执行：

```bash
cd standalone_datacopy
# 使用跑通NDDMA时加载的那套环境脚本，以下路径仅为示例
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export MSPROF_BIN="$ASCEND_HOME_PATH/tools/profiler/bin/msprof"

# 新建设备构建目录，避免旧的sim缓存/链接配置
cmake -S . -B build_npu -DNPU_ARCH=dav-3510 -DRUN_MODE=npu
cmake --build build_npu -j8
```

不要为这次运行额外添加simulator目录到LD_LIBRARY_PATH；如果当前shell已经添加过，
建议在新shell中重新加载跑通NDDMA的设备环境。
`SOC_VERSION` 对此采集入口不是必需参数；`--soc-version` 仅保留为可选元数据，不会传给msprof。
`CMakeLists.txt`仍保留显式sim构建分支，但这里的采集流程使用npu版本。

## 一键校验并采集 MTE2 cycles

必须重编译更新后的ASC：本版程序新增可选校验结果文件参数。

```bash
python3 profile.py \
  --binary build_npu/demo_datacopy \
  --msprof "$MSPROF_BIN" \
  --rows 256 --cols 768 \
  --small-kib 8 --large-kib 240 \
  --trials 3 --device 0 \
  --output results_mte2_8_240
```

`--msprof-bin` 是 `--msprof` 的别名。可以直接指定跑通NDDMA使用的msprof绝对路径。
输出目录必须不存在，以防混入旧日志/校验结果。
脚本先执行3次校验，再执行9次性能采集：

1. 校验STORE=1，把每个tile写回诊断GM并逐元素验证；运行成功后程序写出 `verification.txt`。
   不再依赖profiler的run.log是否包含程序stdout。
2. 性能采集STORE=0，关闭诊断写回；预转置阶段写GM临时B始终保留。
3. 每次进程遍历一次完整输入；pretranspose包含两个顺序kernel，均被采集。
4. 每个kernel必须匹配一条op_summary记录；缺少MTE2占比/时间字段会明确报错，不把总cycles冒充MTE2 cycles。

底层命令与NDDMA一致。例如：

```bash
"$MSPROF_BIN" --output=perf_small \
  --application="$(pwd)/build_npu/demo_datacopy small 256 768 8 0 0"
"$MSPROF_BIN" --output=perf_large \
  --application="$(pwd)/build_npu/demo_datacopy large 256 768 240 0 0"
"$MSPROF_BIN" --output=perf_pretranspose \
  --application="$(pwd)/build_npu/demo_datacopy pretranspose 256 768 240 0 0"
```

路径含空格时优先使用Python脚本，它通过shlex.join引用application参数。
直接检查正确性也可以运行：

```bash
./build_npu/demo_datacopy small 256 768 8 1 0
```

## 指标口径

```text
单kernel mte2_cycles = aiv_total_cycles × normalized_mte2_ratio
```

沿用NDDMA占比字段优先顺序：

1. `aiv_mte2_ratio`
2. `mte2_exe_ratio`
3. `aic_mte2_ratio`（兼容参考实现的回退字段，会记录字段来源）
4. 缺少正占比时，使用 `aiv_mte2_time(us)/aiv_time(us)` 等时间比值。

纯数值占比大于1时除以100，否则视为0..1的小数，与NDDMA约定一致。
显式包含 `%` 的值始终除以100（包括 `0.5%`）。裸值1存在百分比/小数歧义，
按参考实现解释为1.0；遇到这类导出格式请核对字段单位。
有效零值保留为0；字段缺失不填0。当前blockDim=1，无需额外除以核数。
这是按占比推导的流水线忙碌cycles，不是指令trace逐条累加，也不是完整kernel延迟。

主要比较：

```text
small.mte2_cycles        = 小tile完整遍历的MTE2 cycles
large.mte2_cycles        = 大tile完整遍历的MTE2 cycles
pretranspose.mte2_cycles = 预转置阶段MTE2 cycles + 后续连续搬入MTE2 cycles
```

第三种还会单独输出 `pretranspose_mte2_cycles` 与 `load_mte2_cycles`。
**MTE2不包括写回B的MTE3代价。** 因此同时保留 `mte3_cycles` 和 `aiv_total_cycles` 作为辅助指标。
缺少MTE3字段时为null/空，不伪造为0；MTE2与MTE3可能重叠，不能简单相加当作端到端耗时。
`aiv_total_cycles`在双kernel方案中是两个kernel之和，不包括kernel间调度空隙。

输出：

- `stages.csv`：每个kernel的MTE2占比、来源字段、原始AIV总cycles、MTE2/MTE3 cycles、源CSV路径/行号。
- `raw.csv`：每个trial按方案汇总的cycles。
- `summary.json`：中位数、MTE2 min/max、`small_over_large_mte2` 和 `pretranspose_over_large_mte2`。
  比值大于1表示分子方案MTE2 cycles更多；分母为0时输出null。
- `config.json`、各目录 `command.json`/`run.log`/`prof/`：完整运行配置与原始数据。

原始profiling文件已在输出目录时，可以相同参数追加 `--analyze-only` 重新解析。

扫描tile预算：

```bash
for kib in 32 64 128 192 240; do
  python3 profile.py --binary build_npu/demo_datacopy --msprof "$MSPROF_BIN" \
    --rows 256 --cols 768 --small-kib 8 --large-kib "$kib" \
    --trials 3 --output "results_mte2_8_${kib}"
done
```

## 验证状态

本机macOS，没有ASC/msprof，尚未完成目标编译、设备正确性检查或实际cycles采集。
主机测试覆盖三路径地址模型/尾块、预算检查、占比及时间回退、双kernel累加、
application命令引用和stdout不含PASS时的独立结果文件校验。
测试中的合成cycles仅用于验证解析，不代表测量结果。

```bash
python3 -m unittest -v test_address_model.py
```

[DataCopyPad接口参考](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_GMToUB.html)
