# ZKDecGNNs




## 完整文件树

```text
ZKDecGNNs/
├── README.md  # 本说明文档
└── spartan-zeno-zk/
    ├── benchmarks/
    │   ├── graph_classification_runner.py  # 单图预处理、证明/验证调度与结果统计
    │   ├── graph_dataset_loader.py  # 加载图数据、归一化、定点量化、填充和边表提取
    │   ├── results/
    │   │   ├── graph_classification_splits/
    │   │   │   ├── mutag_257bcabcd14837ed.json  # MUTAG 数据划分记录，标识 257bcabcd14837ed
    │   │   │   ├── mutag_8ec1225eed802118.json  # MUTAG 数据划分记录，标识 8ec1225eed802118
    │   │   │   ├── nci1_6ec46c5809f508df.json  # NCI1 数据划分记录，标识 6ec46c5809f508df
    │   │   │   ├── reddit-binary_3afe307c9bc24960.json  # REDDIT-BINARY 数据划分记录，标识 3afe307c9bc24960
    │   │   │   └── reddit-binary_f7a4fe531b648f3f.json  # REDDIT-BINARY 数据划分记录，标识 f7a4fe531b648f3f
    │   │   ├── mutag_appnp_k10_under_sign300max_graph_classification_dataset_runs_raw.csv  # MUTAG / APPNP K=10：每轮数据集级原始结果（CSV）
    │   │   ├── mutag_appnp_k10_under_sign300max_graph_classification_dataset_runs_raw.json  # MUTAG / APPNP K=10：每轮数据集级原始结果（JSON）
    │   │   ├── mutag_appnp_k10_under_sign300max_graph_classification_per_graph_raw.csv  # MUTAG / APPNP K=10：逐图原始结果（CSV）
    │   │   ├── mutag_appnp_k10_under_sign300max_graph_classification_per_graph_raw.json  # MUTAG / APPNP K=10：逐图原始结果（JSON）
    │   │   ├── mutag_appnp_k10_under_sign300max_graph_classification_purpose.json  # MUTAG / APPNP K=10：实验配置与目的
    │   │   ├── mutag_appnp_k10_under_sign300max_graph_classification_summary.json  # MUTAG / APPNP K=10：实验汇总
    │   │   ├── mutag_sgc_k2_under_sign300max_graph_classification_dataset_runs_raw.csv  # MUTAG / SGC K=2：每轮数据集级原始结果（CSV）
    │   │   ├── mutag_sgc_k2_under_sign300max_graph_classification_dataset_runs_raw.json  # MUTAG / SGC K=2：每轮数据集级原始结果（JSON）
    │   │   ├── mutag_sgc_k2_under_sign300max_graph_classification_per_graph_raw.csv  # MUTAG / SGC K=2：逐图原始结果（CSV）
    │   │   ├── mutag_sgc_k2_under_sign300max_graph_classification_per_graph_raw.json  # MUTAG / SGC K=2：逐图原始结果（JSON）
    │   │   ├── mutag_sgc_k2_under_sign300max_graph_classification_purpose.json  # MUTAG / SGC K=2：实验配置与目的
    │   │   ├── mutag_sgc_k2_under_sign300max_graph_classification_summary.json  # MUTAG / SGC K=2：实验汇总
    │   │   ├── mutag_sign300_under_sign300max_graph_classification_dataset_runs_raw.csv  # MUTAG / SIGN：每轮数据集级原始结果（CSV）
    │   │   ├── mutag_sign300_under_sign300max_graph_classification_dataset_runs_raw.json  # MUTAG / SIGN：每轮数据集级原始结果（JSON）
    │   │   ├── mutag_sign300_under_sign300max_graph_classification_per_graph_raw.csv  # MUTAG / SIGN：逐图原始结果（CSV）
    │   │   ├── mutag_sign300_under_sign300max_graph_classification_per_graph_raw.json  # MUTAG / SIGN：逐图原始结果（JSON）
    │   │   ├── mutag_sign300_under_sign300max_graph_classification_purpose.json  # MUTAG / SIGN：实验配置与目的
    │   │   ├── mutag_sign300_under_sign300max_graph_classification_summary.json  # MUTAG / SIGN：实验汇总
    │   │   ├── mutag_ssgc_k2_under_sign300max_graph_classification_dataset_runs_raw.csv  # MUTAG / SSGC K=2：每轮数据集级原始结果（CSV）
    │   │   ├── mutag_ssgc_k2_under_sign300max_graph_classification_dataset_runs_raw.json  # MUTAG / SSGC K=2：每轮数据集级原始结果（JSON）
    │   │   ├── mutag_ssgc_k2_under_sign300max_graph_classification_per_graph_raw.csv  # MUTAG / SSGC K=2：逐图原始结果（CSV）
    │   │   ├── mutag_ssgc_k2_under_sign300max_graph_classification_per_graph_raw.json  # MUTAG / SSGC K=2：逐图原始结果（JSON）
    │   │   ├── mutag_ssgc_k2_under_sign300max_graph_classification_purpose.json  # MUTAG / SSGC K=2：实验配置与目的
    │   │   ├── mutag_ssgc_k2_under_sign300max_graph_classification_summary.json  # MUTAG / SSGC K=2：实验汇总
    │   │   ├── nci1_appnp_k10_under_sign300max_graph_classification_dataset_runs_raw.csv  # NCI1 / APPNP K=10：每轮数据集级原始结果（CSV）
    │   │   ├── nci1_appnp_k10_under_sign300max_graph_classification_dataset_runs_raw.json  # NCI1 / APPNP K=10：每轮数据集级原始结果（JSON）
    │   │   ├── nci1_appnp_k10_under_sign300max_graph_classification_per_graph_raw.csv  # NCI1 / APPNP K=10：逐图原始结果（CSV）
    │   │   ├── nci1_appnp_k10_under_sign300max_graph_classification_per_graph_raw.json  # NCI1 / APPNP K=10：逐图原始结果（JSON）
    │   │   ├── nci1_appnp_k10_under_sign300max_graph_classification_purpose.json  # NCI1 / APPNP K=10：实验配置与目的
    │   │   ├── nci1_appnp_k10_under_sign300max_graph_classification_summary.json  # NCI1 / APPNP K=10：实验汇总
    │   │   ├── nci1_sgc_k2_under_sign300max_graph_classification_dataset_runs_raw.csv  # NCI1 / SGC K=2：每轮数据集级原始结果（CSV）
    │   │   ├── nci1_sgc_k2_under_sign300max_graph_classification_dataset_runs_raw.json  # NCI1 / SGC K=2：每轮数据集级原始结果（JSON）
    │   │   ├── nci1_sgc_k2_under_sign300max_graph_classification_per_graph_raw.csv  # NCI1 / SGC K=2：逐图原始结果（CSV）
    │   │   ├── nci1_sgc_k2_under_sign300max_graph_classification_per_graph_raw.json  # NCI1 / SGC K=2：逐图原始结果（JSON）
    │   │   ├── nci1_sgc_k2_under_sign300max_graph_classification_purpose.json  # NCI1 / SGC K=2：实验配置与目的
    │   │   ├── nci1_sgc_k2_under_sign300max_graph_classification_summary.json  # NCI1 / SGC K=2：实验汇总
    │   │   ├── nci1_sign300_under_sign300max_graph_classification_dataset_runs_raw.csv  # NCI1 / SIGN：每轮数据集级原始结果（CSV）
    │   │   ├── nci1_sign300_under_sign300max_graph_classification_dataset_runs_raw.json  # NCI1 / SIGN：每轮数据集级原始结果（JSON）
    │   │   ├── nci1_sign300_under_sign300max_graph_classification_per_graph_raw.csv  # NCI1 / SIGN：逐图原始结果（CSV）
    │   │   ├── nci1_sign300_under_sign300max_graph_classification_per_graph_raw.json  # NCI1 / SIGN：逐图原始结果（JSON）
    │   │   ├── nci1_sign300_under_sign300max_graph_classification_purpose.json  # NCI1 / SIGN：实验配置与目的
    │   │   ├── nci1_sign300_under_sign300max_graph_classification_summary.json  # NCI1 / SIGN：实验汇总
    │   │   ├── nci1_ssgc_k2_under_sign300max_graph_classification_dataset_runs_raw.csv  # NCI1 / SSGC K=2：每轮数据集级原始结果（CSV）
    │   │   ├── nci1_ssgc_k2_under_sign300max_graph_classification_dataset_runs_raw.json  # NCI1 / SSGC K=2：每轮数据集级原始结果（JSON）
    │   │   ├── nci1_ssgc_k2_under_sign300max_graph_classification_per_graph_raw.csv  # NCI1 / SSGC K=2：逐图原始结果（CSV）
    │   │   ├── nci1_ssgc_k2_under_sign300max_graph_classification_per_graph_raw.json  # NCI1 / SSGC K=2：逐图原始结果（JSON）
    │   │   ├── nci1_ssgc_k2_under_sign300max_graph_classification_purpose.json  # NCI1 / SSGC K=2：实验配置与目的
    │   │   ├── nci1_ssgc_k2_under_sign300max_graph_classification_summary.json  # NCI1 / SSGC K=2：实验汇总
    │   │   ├── reddit_binary_appnp_k10_under_sign300max_graph_classification_dataset_runs_raw.csv  # REDDIT-BINARY / APPNP K=10：每轮数据集级原始结果（CSV）
    │   │   ├── reddit_binary_appnp_k10_under_sign300max_graph_classification_dataset_runs_raw.json  # REDDIT-BINARY / APPNP K=10：每轮数据集级原始结果（JSON）
    │   │   ├── reddit_binary_appnp_k10_under_sign300max_graph_classification_per_graph_raw.csv  # REDDIT-BINARY / APPNP K=10：逐图原始结果（CSV）
    │   │   ├── reddit_binary_appnp_k10_under_sign300max_graph_classification_per_graph_raw.json  # REDDIT-BINARY / APPNP K=10：逐图原始结果（JSON）
    │   │   ├── reddit_binary_appnp_k10_under_sign300max_graph_classification_purpose.json  # REDDIT-BINARY / APPNP K=10：实验配置与目的
    │   │   ├── reddit_binary_appnp_k10_under_sign300max_graph_classification_summary.json  # REDDIT-BINARY / APPNP K=10：实验汇总
    │   │   ├── reddit_binary_sgc_k2_under_sign300max_graph_classification_dataset_runs_raw.csv  # REDDIT-BINARY / SGC K=2：每轮数据集级原始结果（CSV）
    │   │   ├── reddit_binary_sgc_k2_under_sign300max_graph_classification_dataset_runs_raw.json  # REDDIT-BINARY / SGC K=2：每轮数据集级原始结果（JSON）
    │   │   ├── reddit_binary_sgc_k2_under_sign300max_graph_classification_per_graph_raw.csv  # REDDIT-BINARY / SGC K=2：逐图原始结果（CSV）
    │   │   ├── reddit_binary_sgc_k2_under_sign300max_graph_classification_per_graph_raw.json  # REDDIT-BINARY / SGC K=2：逐图原始结果（JSON）
    │   │   ├── reddit_binary_sgc_k2_under_sign300max_graph_classification_purpose.json  # REDDIT-BINARY / SGC K=2：实验配置与目的
    │   │   ├── reddit_binary_sgc_k2_under_sign300max_graph_classification_summary.json  # REDDIT-BINARY / SGC K=2：实验汇总
    │   │   ├── reddit_binary_sign300_under_sign300max_graph_classification_dataset_runs_raw.csv  # REDDIT-BINARY / SIGN：每轮数据集级原始结果（CSV）
    │   │   ├── reddit_binary_sign300_under_sign300max_graph_classification_dataset_runs_raw.json  # REDDIT-BINARY / SIGN：每轮数据集级原始结果（JSON）
    │   │   ├── reddit_binary_sign300_under_sign300max_graph_classification_per_graph_raw.csv  # REDDIT-BINARY / SIGN：逐图原始结果（CSV）
    │   │   ├── reddit_binary_sign300_under_sign300max_graph_classification_per_graph_raw.json  # REDDIT-BINARY / SIGN：逐图原始结果（JSON）
    │   │   ├── reddit_binary_sign300_under_sign300max_graph_classification_purpose.json  # REDDIT-BINARY / SIGN：实验配置与目的
    │   │   ├── reddit_binary_sign300_under_sign300max_graph_classification_summary.json  # REDDIT-BINARY / SIGN：实验汇总
    │   │   ├── reddit_binary_ssgc_k2_under_sign300max_graph_classification_dataset_runs_raw.csv  # REDDIT-BINARY / SSGC K=2：每轮数据集级原始结果（CSV）
    │   │   ├── reddit_binary_ssgc_k2_under_sign300max_graph_classification_dataset_runs_raw.json  # REDDIT-BINARY / SSGC K=2：每轮数据集级原始结果（JSON）
    │   │   ├── reddit_binary_ssgc_k2_under_sign300max_graph_classification_per_graph_raw.csv  # REDDIT-BINARY / SSGC K=2：逐图原始结果（CSV）
    │   │   ├── reddit_binary_ssgc_k2_under_sign300max_graph_classification_per_graph_raw.json  # REDDIT-BINARY / SSGC K=2：逐图原始结果（JSON）
    │   │   ├── reddit_binary_ssgc_k2_under_sign300max_graph_classification_purpose.json  # REDDIT-BINARY / SSGC K=2：实验配置与目的
    │   │   └── reddit_binary_ssgc_k2_under_sign300max_graph_classification_summary.json  # REDDIT-BINARY / SSGC K=2：实验汇总
    │   ├── run_mutag_appnp_k10_sign300max_zk_benchmark.py  # MUTAG / SIGN 图分类实验入口
    │   ├── run_mutag_sgc_k2_sign300max_full_dataset_optimized_rust.py  # MUTAG / SGC K=2 图分类实验入口，其他实验脚本复用的基础实现
    │   ├── run_mutag_sign300max_zk_benchmark.py  # MUTAG / SIGN 图分类实验入口
    │   ├── run_mutag_ssgc_k2_sign300max_zk_benchmark.py  # MUTAG / SSGC K=2 图分类实验入口
    │   ├── run_nci1_appnp_k10_sign300max_zk_benchmark.py  # NCI1 / SIGN 图分类实验入口
    │   ├── run_nci1_sgc_k2_sign300max_zk_benchmark.py  # NCI1 / SGC K=2 图分类实验入口
    │   ├── run_nci1_sign300max_zk_benchmark.py  # NCI1 / SIGN 图分类实验入口
    │   ├── run_nci1_ssgc_k2_sign300max_zk_benchmark.py  # NCI1 / SSGC K=2 图分类实验入口
    │   ├── run_reddit_binary_appnp_k10_sign300max_zk_benchmark.py  # REDDIT-BINARY / SIGN 图分类实验入口
    │   ├── run_reddit_binary_sgc_k2_sign300max_zk_benchmark.py  # REDDIT-BINARY / SGC K=2 图分类实验入口
    │   ├── run_reddit_binary_sign300max_zk_benchmark.py  # REDDIT-BINARY / SIGN 图分类实验入口
    │   ├── run_reddit_binary_ssgc_k2_sign300max_zk_benchmark.py  # REDDIT-BINARY / SSGC K=2 图分类实验入口
    │   └── rust_streaming_backend.py  # Rust 动态库加载器及流式计算辅助接口
    ├── build.rs  # Rust 构建辅助脚本
    ├── Cargo.lock  # Rust 依赖版本锁定文件
    ├── Cargo.toml  # Rust crate 配置、依赖与编译参数
    ├── run_graph_classification.py  # 通用图分类 Python 入口
    ├── src/
    │   ├── curve_config.rs  # 曲线、域元素、生成元和群运算配置
    │   ├── graph_classification.rs  # 图分类推理、稀疏传播、承诺、证明与验证主实现
    │   ├── hyrax.rs  # 行承诺、零行处理、折叠和 IPA
    │   ├── ipa.rs  # 另一组 IPA 折叠与证明辅助实现
    │   ├── lib.rs  # Rust 模块声明及 Python 扩展接口注册
    │   ├── model_registry.rs  # 模型架构、物理容器与模型配置
    │   ├── polynomial.rs  # 多线性扩展、基函数与多项式工具
    │   ├── poseidon.rs  # Poseidon 海绵、哈希和挑战生成工具
    │   ├── quantization.rs  # 定点量化、截断和范围处理工具
    │   ├── spark.rs  # 内存操作、代数标签、乘积树和守恒检查工具
    │   ├── streaming.rs  # 暴露给 Python 的流式 MLE、承诺和折叠计算
    │   ├── verifier.rs  # 通用验证辅助代码
    │   ├── witness/
    │   │   ├── assembly.rs  # 辅助见证、多项式引擎与证明状态组装
    │   │   ├── binary_tree.rs  # 辅助二叉乘积树、轨迹 MLE 与承诺结构
    │   │   ├── container.rs  # 辅助物理容器、模型权重及掩码
    │   │   ├── diffusion.rs  # 辅助多轮传播、稀疏轨迹和全局状态 MLE
    │   │   ├── interactive.rs  # 辅助交互见证、压缩和排序检查
    │   │   ├── memory_check.rs  # 辅助内存访问记录、排序和一致性检查
    │   │   ├── mod.rs  # witness 子模块声明与公共见证类型
    │   │   ├── post_transform.rs  # 辅助传播后变换及定点截断
    │   │   ├── pre_transform.rs  # 辅助传播前变换
    │   │   └── transcript.rs  # 辅助协议转录与挑战容器
    │   └── zeno_compiler.rs  # 模型静态类型和参数分析辅助组件
    └── tests/
        └── test_graph_classification.py  # 图分类接口、输入和验证行为的测试
```


