# pipeline — ONNX 导出 / 量化 / QNN 编译

脚本从两处临时目录整理过来（原位置仍在，未删除）：

| 子目录 | 原位置 | 内容 |
|---|---|---|
| `qaihub/` | `~/projects/rvc/rvc-android/salvage/dl_vv/` | 整条 ViiTorVoice 云端流水线：导出、静态化、切段、qai-hub 提交、link、下载、验证（81 个脚本） |
| `frontend/` | `~/tmp/fe_npu/`、`~/projects/voice_editor/sm8750_voice_editor/tools/` | w2v 前端导出 + mel 前端 + 前端量化/上传辅助 |

## 常用入口

| 想干什么 | 脚本 |
|---|---|
| 把 dualcodec 图编译成 DLC | `qaihub/compile_vv.py [decoder\|encoder\|w2v] [static_len]` |
| 编译 encode-core（按帧数定型） | `qaihub/core_pipeline.py <onnx> <samples> <frames> <tag>` |
| 静态化 + 清理 torch.export 残留 | `qaihub/core_clean.py`、`core_fold.py` |
| 切段（避开 qai-hub 80 MB 上传上限 / VTCM 限制） | `qaihub/core_split.py`、`core_slice_front.py` |
| 多 shape 共享权重编译 + link 去重 | `qaihub/a16_from_phone.py`（另有 `qaihub_shared_weight_multi_graph` 记忆） |
| qai-hub 产物直下（S3，比隧道快 ~40×） | `qaihub/qai_dl.py` |
| 云端 runner 控制（cloudflared + paramiko） | `qaihub/runner.py` |
| 编解码链路验证（对 python 参考） | `qaihub/verify_chain.py`、`verify_tail.py`、`metrics.py` |

## 注意

* 脚本里写死了 Termux 绝对路径（`/data/data/com.termux/files/home/tmp/dl_vv/...`），
  从别处跑要改路径；文件名保持原样以便和记忆里的命令对上。
* 重编译前必须 `onnx.checker.check_model(m, full_check=True)` 本地复现服务端 prepass。
* HF 下载要 `HF_HUB_DISABLE_XET=1`；HEAD 请求 qai-hub 用 `workbench.aihub.qualcomm.com`。
