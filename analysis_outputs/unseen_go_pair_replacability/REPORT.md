# 未见 GO Pair 边可替换性分析报告

## 数据概览
- 训练集 GO 原子数: `375`
- 训练集已知 GO pair 边数: `567`
- strict OOD unique 组合数: `76`
- strict OOD 中未见 pair 边数: `146`

## 全局统计
- 每个锚点-未知边的平均已知邻居数: `15.42`
- 至少存在一个 sibling 支持的比例: `0.120`
- 至少存在一个 parent 支持的比例: `0.048`
- 至少存在一个 child 支持的比例: `0.075`
- 平均最小语义距离 `min dist(e, N(a))`: `4.05`
- 至少存在 1 个共同邻居的比例: `0.390`
- 平均 link-prediction Jaccard: `0.041`
- sibling-set 之间至少存在一条训练边的比例: `0.089`
- 理论可泛化 pair 比例（只要任一树/图条件满足）: `0.568`

## 共现支持与模型正确率关系

- `anchor_known_neighbor_count` 与 `strict_combo_exact_rate_mean` 的相关系数: `-0.021`
- `anchor_sibling_support_count` 与 `strict_combo_exact_rate_mean` 的相关系数: `-0.001`
- `sibling_cross_edge_count` 与 `strict_combo_exact_rate_mean` 的相关系数: `0.115`
- `link_common_neighbors` 与 `strict_combo_exact_rate_mean` 的相关系数: `0.144`
- `best_replaceability_score` 与 `strict_combo_exact_rate_mean` 的相关系数: `0.107`
- `mean_full_test_hit_rate` 与 `strict_combo_exact_rate_mean` 的相关系数: `0.192`
- `expected_pair_rate_fulltest` 与 `strict_combo_exact_rate_mean` 的相关系数: `0.226`
- `residual_vs_fulltest_pair_expectation` 与 `strict_combo_exact_rate_mean` 的相关系数: `0.650`
- `anchor_freq` 与 `strict_combo_exact_rate_mean` 的相关系数: `0.123`
- `unseen_partner_freq` 与 `strict_combo_exact_rate_mean` 的相关系数: `0.123`
- `anchor_partner_degree` 与 `strict_combo_exact_rate_mean` 的相关系数: `-0.021`
- `unseen_partner_degree` 与 `strict_combo_exact_rate_mean` 的相关系数: `-0.021`

- `tree_support_supported_group`: support 组平均正确率=`0.122`，无 support 组=`0.070`，样本数=`82/210`
- `graph_support_supported_group`: support 组平均正确率=`0.112`，无 support 组=`0.055`，样本数=`152/140`
- `theoretical_generalizable_supported_group`: support 组平均正确率=`0.114`，无 support 组=`0.045`，样本数=`166/126`

## 排除单 GO 难度影响后的观察

- `mean_full_test_hit_rate` 表示组成该未见 pair 的两个 GO，在完整测试集上的平均边际命中率。
- `expected_pair_rate_fulltest` 现在改为两个 GO 的完整测试集边际命中率的 `min`，表示“如果瓶颈只由更难的那个 GO 决定”时的 pair 期望成功率。
- `residual_vs_fulltest_pair_expectation` 为 strict OOD 观察成功率减去这个期望，越负说明越像是组合泛化额外失败。

## Top 20 最可替换的未见边（按 best replaceability score）

- `(GO:0030550, GO:0015459)`: best substitute=`GO:0099106`，relation=`descendant`，score=`0.771`，common-neighbors=`5`，min-dist=`1.0`
- `(GO:0008774, GO:0050661)`: best substitute=`GO:0051287`，relation=`sibling`，score=`0.753`，common-neighbors=`1`，min-dist=`2.0`
- `(GO:0052915, GO:0008990)`: best substitute=`GO:0070043`，relation=`sibling`，score=`0.728`，common-neighbors=`2`，min-dist=`2.0`
- `(GO:0001671, GO:0051082)`: best substitute=`GO:0051087`，relation=`sibling`，score=`0.708`，common-neighbors=`1`，min-dist=`2.0`
- `(GO:0042802, GO:0004585)`: best substitute=`GO:0004070`，relation=`sibling`，score=`0.682`，common-neighbors=`1`，min-dist=`2.0`
- `(GO:0000036, GO:0008289)`: best substitute=`GO:0000035`，relation=`sibling`，score=`0.673`，common-neighbors=`1`，min-dist=`2.0`
- `(GO:0009055, GO:0008137)`: best substitute=`GO:0016655`，relation=`descendant`，score=`0.667`，common-neighbors=`2`，min-dist=`3.0`
- `(GO:0009055, GO:0003954)`: best substitute=`GO:0016652`，relation=`sibling`，score=`0.661`，common-neighbors=`1`，min-dist=`2.0`
- `(GO:0000774, GO:0042802)`: best substitute=`GO:0051082`，relation=`sibling`，score=`0.658`，common-neighbors=`1`，min-dist=`1.0`
- `(GO:0051087, GO:0042802)`: best substitute=`GO:0051082`，relation=`sibling`，score=`0.658`，common-neighbors=`1`，min-dist=`1.0`
- `(GO:0031177, GO:0008289)`: best substitute=`GO:0000035`，relation=`sibling`，score=`0.657`，common-neighbors=`0`，min-dist=`2.0`
- `(GO:0051082, GO:0042802)`: best substitute=`GO:0051087`，relation=`sibling`，score=`0.636`，common-neighbors=`3`，min-dist=`1.0`
- `(GO:0001671, GO:0042802)`: best substitute=`GO:0051087`，relation=`sibling`，score=`0.636`，common-neighbors=`0`，min-dist=`2.0`
- `(GO:0003954, GO:0009055)`: best substitute=`GO:0008137`，relation=`ancestor`，score=`0.625`，common-neighbors=`1`，min-dist=`1.0`
- `(GO:0004808, GO:0071949)`: best substitute=`GO:0050660`，relation=`descendant`，score=`0.615`，common-neighbors=`1`，min-dist=`1.0`
- `(GO:0016645, GO:0071949)`: best substitute=`GO:0050660`，relation=`descendant`，score=`0.615`，common-neighbors=`1`，min-dist=`1.0`
- `(GO:0050661, GO:0000166)`: best substitute=`GO:0051287`，relation=`ancestor`，score=`0.609`，common-neighbors=`1`，min-dist=`1.0`
- `(GO:0042802, GO:0004347)`: best substitute=`GO:0004342`，relation=`sibling`，score=`0.605`，common-neighbors=`1`，min-dist=`2.0`
- `(GO:0016787, GO:0008773)`: best substitute=`GO:0033819`，relation=`sibling`，score=`0.605`，common-neighbors=`0`，min-dist=`2.0`
- `(GO:0051287, GO:0008926)`: best substitute=`GO:0047952`，relation=`sibling`，score=`0.594`，common-neighbors=`0`，min-dist=`2.0`

## Top 20 最难替换的未见边

- `(GO:0042803, GO:0141153)`: score=`0.209`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`50.0`
- `(GO:0004477, GO:0042803)`: score=`0.241`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`10.0`
- `(GO:0036218, GO:0030145)`: score=`0.241`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`10.0`
- `(GO:0036221, GO:0030145)`: score=`0.241`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`10.0`
- `(GO:0004488, GO:0042803)`: score=`0.245`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`9.0`
- `(GO:0008757, GO:0004719)`: score=`0.245`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`9.0`
- `(GO:0036218, GO:0042803)`: score=`0.245`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`9.0`
- `(GO:0036221, GO:0042803)`: score=`0.245`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`9.0`
- `(GO:0042803, GO:0004488)`: score=`0.245`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`6.0`
- `(GO:0016597, GO:0042301)`: score=`0.245`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`5.0`
- `(GO:0016829, GO:0050660)`: score=`0.245`，parent/sibling/child support=`0/0/0`，common-neighbors=`1`，min-dist=`5.0`
- `(GO:0005212, GO:0004056)`: score=`0.250`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`8.0`
- `(GO:0004372, GO:0008168)`: score=`0.250`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`8.0`
- `(GO:0004425, GO:0005507)`: score=`0.250`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`8.0`
- `(GO:0036218, GO:0042802)`: score=`0.250`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`8.0`
- `(GO:0036221, GO:0042802)`: score=`0.250`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`8.0`
- `(GO:0097163, GO:0042803)`: score=`0.250`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`8.0`
- `(GO:1904047, GO:0042803)`: score=`0.250`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`8.0`
- `(GO:0042803, GO:0004477)`: score=`0.250`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`7.0`
- `(GO:0042803, GO:0036218)`: score=`0.250`，parent/sibling/child support=`0/0/0`，common-neighbors=`0`，min-dist=`7.0`

## Top 10 锚点 GO（未知目标最多）

- `GO:0042802`: known-neighbors=`64`，unknown-targets=`37`，mean-best-score=`0.371`
- `GO:0042803`: known-neighbors=`33`，unknown-targets=`34`，mean-best-score=`0.324`
- `GO:0005507`: known-neighbors=`5`，unknown-targets=`9`，mean-best-score=`0.331`
- `GO:0051117`: known-neighbors=`2`，unknown-targets=`6`，mean-best-score=`0.285`
- `GO:0001671`: known-neighbors=`1`，unknown-targets=`5`，mean-best-score=`0.505`
- `GO:0016829`: known-neighbors=`6`，unknown-targets=`5`，mean-best-score=`0.271`
- `GO:0047429`: known-neighbors=`0`，unknown-targets=`5`，mean-best-score=`nan`
- `GO:0030145`: known-neighbors=`14`，unknown-targets=`4`，mean-best-score=`0.388`
- `GO:0016787`: known-neighbors=`4`，unknown-targets=`4`，mean-best-score=`0.361`
- `GO:0000166`: known-neighbors=`8`，unknown-targets=`4`，mean-best-score=`0.346`

## GO 标签偏好与困难样本

- `GO:0042803`: train-freq=`615`，partner-degree=`33`，涉及未见边=`68`，hard-ratio=`0.971`，mean-exact-rate=`0.015`，fulltest-hit=`0.172`
- `GO:0042802`: train-freq=`137`，partner-degree=`64`，涉及未见边=`74`，hard-ratio=`0.892`，mean-exact-rate=`0.077`，fulltest-hit=`0.553`
- `GO:0005507`: train-freq=`419`，partner-degree=`5`，涉及未见边=`18`，hard-ratio=`0.889`，mean-exact-rate=`0.111`，fulltest-hit=`0.488`
- `GO:0051117`: train-freq=`389`，partner-degree=`2`，涉及未见边=`12`，hard-ratio=`1.000`，mean-exact-rate=`0.000`，fulltest-hit=`0.081`
- `GO:0016829`: train-freq=`862`，partner-degree=`6`，涉及未见边=`10`，hard-ratio=`1.000`，mean-exact-rate=`0.000`，fulltest-hit=`0.782`
- `GO:0047429`: train-freq=`227`，partner-degree=`0`，涉及未见边=`10`，hard-ratio=`1.000`，mean-exact-rate=`0.000`，fulltest-hit=`0.135`
- `GO:0001671`: train-freq=`169`，partner-degree=`1`，涉及未见边=`10`，hard-ratio=`1.000`，mean-exact-rate=`0.000`，fulltest-hit=`0.095`
- `GO:0030145`: train-freq=`1978`，partner-degree=`14`，涉及未见边=`8`，hard-ratio=`1.000`，mean-exact-rate=`0.000`，fulltest-hit=`0.153`
- `GO:0008289`: train-freq=`704`，partner-degree=`8`，涉及未见边=`8`，hard-ratio=`1.000`，mean-exact-rate=`0.000`，fulltest-hit=`0.162`
- `GO:0004056`: train-freq=`580`，partner-degree=`0`，涉及未见边=`8`，hard-ratio=`1.000`，mean-exact-rate=`0.000`，fulltest-hit=`0.205`

## 困难组合偏好

- `('GO:0000774', 'GO:0001671', 'GO:0042802', 'GO:0042803', 'GO:0051082', 'GO:0051087', 'GO:0051117')`: size=`7`，unseen-pairs=`13`，mean-go-freq=`525.4`，pair-support-ratio=`0.381`，expected-fulltest=`0.000`，exact-rate=`0.000`，residual=`0.000`
- `('GO:0030145', 'GO:0036218', 'GO:0036221', 'GO:0042802', 'GO:0042803', 'GO:0047429')`: size=`6`，unseen-pairs=`12`，mean-go-freq=`565.5`，pair-support-ratio=`0.200`，expected-fulltest=`0.000`，exact-rate=`0.000`，residual=`0.000`
- `('GO:0004585', 'GO:0005543', 'GO:0016597', 'GO:0042301', 'GO:0042802')`: size=`5`，unseen-pairs=`7`，mean-go-freq=`414.4`，pair-support-ratio=`0.300`，expected-fulltest=`0.000`，exact-rate=`0.000`，residual=`0.000`
- `('GO:0004347', 'GO:0042802', 'GO:0042803', 'GO:0048029', 'GO:0097367')`: size=`5`，unseen-pairs=`5`，mean-go-freq=`480.6`，pair-support-ratio=`0.500`，expected-fulltest=`0.000`，exact-rate=`0.000`，residual=`0.000`
- `('GO:0004076', 'GO:0005506', 'GO:0042803', 'GO:0051537', 'GO:0051539')`: size=`5`，unseen-pairs=`4`，mean-go-freq=`2061.6`，pair-support-ratio=`0.600`，expected-fulltest=`0.000`，exact-rate=`0.000`，residual=`0.000`
- `('GO:0004851', 'GO:0042803', 'GO:0043115', 'GO:0051266', 'GO:0051287')`: size=`5`，unseen-pairs=`4`，mean-go-freq=`690.8`，pair-support-ratio=`0.600`，expected-fulltest=`0.000`，exact-rate=`0.000`，residual=`0.000`
- `('GO:0000774', 'GO:0042802', 'GO:0042803', 'GO:0051082', 'GO:0051087')`: size=`5`，unseen-pairs=`3`，mean-go-freq=`624.0`，pair-support-ratio=`0.700`，expected-fulltest=`0.000`，exact-rate=`0.000`，residual=`0.000`
- `('GO:0004497', 'GO:0004601', 'GO:0005506', 'GO:0016705', 'GO:0020037')`: size=`5`，unseen-pairs=`2`，mean-go-freq=`1668.2`，pair-support-ratio=`0.800`，expected-fulltest=`0.065`，exact-rate=`0.000`，residual=`-0.065`
- `('GO:0016783', 'GO:0042803', 'GO:0043546', 'GO:0097163')`: size=`4`，unseen-pairs=`5`，mean-go-freq=`255.2`，pair-support-ratio=`0.167`，expected-fulltest=`0.000`，exact-rate=`0.000`，residual=`0.000`
- `('GO:0008483', 'GO:0016829', 'GO:0030170', 'GO:0042802')`: size=`4`，unseen-pairs=`4`，mean-go-freq=`1152.2`，pair-support-ratio=`0.333`，expected-fulltest=`0.406`，exact-rate=`0.000`，residual=`-0.406`

## 输出文件
- `unseen_pair_ordered_summary.csv`: 每条有向未见边 `(a, e)` 的聚合统计
- `unseen_pair_replacement_candidates.csv`: `(a, e)` 相对每个已知 `(a, b)` 的细粒度可替换性比较
- `anchor_go_summary.csv`: 以锚点 GO 为中心的统计摘要
- `performance_vs_cooccurrence.csv`: 共现/树支持特征与模型正确率关系
- `go_bias_summary.csv`: GO 标签本身的频次、hub 程度与困难度偏好
- `combo_bias_summary.csv`: strict OOD 组合层面的频次/共现偏好与困难度
- `go_difficulty_baseline.csv`: 完整测试集上的 per-GO 命中率与单标签命中率
- `full_test_protein_baseline.csv`: 完整测试集逐蛋白的 raw cover 基线摘要

## 指标说明
- `parent/sibling/child support`: 对锚点 `a` 而言，`e` 的父/兄弟/子节点里有多少个已经作为 `a` 的已知配对邻居出现过
- `link_common_neighbors`: 训练共现 pair 图里，`a` 与 `e` 这条未见边两端的共同邻居数量
- `sibling_cross_edge_count`: `a` 的 sibling 集合与 `e` 的 sibling 集合之间，训练集中已见 pair 边的数量
- `theoretically_generalizable`: 只要树支持或图支持任一条件满足，就视为这条未见边理论上可由训练信号泛化
- `replaceability score`: 综合树关系、pair 图邻域相似度和已知 `(a,b)` 边频率得到的启发式替换性分数