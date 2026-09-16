# OpenTypeless

**A little less typing.** 面向 Apple Silicon Mac 的开源语音输入应用。原生 SwiftUI / AppKit 界面，使用 **SGLang-Omni 的原生 MLX Qwen3-ASR 服务**识别语音，用本地 Qwen3 文本模型完成整理、翻译和语音编辑。

无需账号、订阅或云端推理 API。独立项目，与 Typeless 无关联，不使用其商标素材或私有代码。

## 快速开始

需要 macOS 14+、Apple Silicon、Xcode Command Line Tools（Swift 6 工具链用于测试）、Homebrew。建议至少 16 GB 内存，并为 Python 运行环境及模型预留数 GB 磁盘空间。

在 `sglang-omni` 根目录执行：

```bash
bash openTypeless/scripts/setup.sh
open openTypeless/dist/OpenTypeless.app
```

安装脚本复用仓库根目录的 `install.sh`：创建 `openTypeless/.venv`，安装 SGLang `v0.5.19` 的 Apple Silicon 依赖、当前 SGLang-Omni，以及 `ffmpeg@7`。不会安装 CUDA 包或替换系统 Python。Homebrew 和 Command Line Tools 需预先安装。

首次打开：

1. 在首页允许 **Microphone** 和 **Accessibility**。macOS 的隐私授权必须由用户在系统界面授予，应用无法自行批准。
2. 在 **Settings → Local models → Download & prepare models** 下载并准备模型。首次启动需要联网访问 Hugging Face；权重缓存后可离线推理。
3. 在任意支持辅助功能的文本框放好光标，按 **Control + Option + Space** 开始说话，再按一次结束。结果写入原来的位置。
4. **Esc** 取消。设置中可以录制自己的快捷键，或切换为按住说话、松开完成。

从主窗口直接点击 Start speaking 时，结果显示在应用内供复制。跨应用写入请在目标应用中使用全局快捷键。

应用关闭主窗口后保留菜单栏图标；菜单中的 Quit 会退出应用并关闭其模型进程。启动登录项需要先将构建出的应用放在固定位置，建议 `~/Applications`，再开启 **Open at login**。

## 已实现的使用流程

| 功能 | 行为 |
| --- | --- |
| Dictate | 录音 → 本机 ASR → 去口头语、整理标点 → 原光标写入；支持逐字模式跳过文本模型 |
| Translate | 自动或指定语音语言，输出指定目标语言 |
| Voice edit | 在目标应用选中文字，说明如何修改，替换原选择；不保存选中文本 |
| Ask | 对选择的内容或一般问题提问；答案显示在应用中，不替换选择；不联网检索 |
| 全局快捷键 | 自定义组合键、切换或按住录音、Esc 取消、非抢焦点浮动录音条 |
| 输入设备 | 选择麦克风、实时音量、起止提示音、5 分钟录音上限 |
| Dictionary | 识别词汇提示、指定拼写替换、CSV 导入导出、从历史纠错添加词条 |
| Writing style | 全局及按应用设置 clean / verbatim / casual / formal / concise 风格和偏好 |
| History | 搜索、模式过滤、原文对照、复制、纠错、导出、删除和保留期限 |
| 音频保留 | 默认关闭；开启后可重试普通听写/翻译和导出 WAV；删除记录同步删除音频 |
| 系统设置 | 隐私权限入口、开机启动、深色/浅色主题、模型预加载和卸载 |

功能参照 [Typeless Quickstart](https://www.typeless.com/help/quickstart)、[语音编辑及问答](https://www.typeless.com/help/quickstart/ask-anything)、[历史与词典](https://www.typeless.com/help/quickstart/history-and-dictionary) 的公开交互，核对日期 2026-09-17。此版本不承诺相同的模型质量；不包含云同步、移动端键盘、跨应用被动学习、联网搜索与自动网页操作。纠错学习只发生在用户明确保存的词条上。

## 模型与运行边界

上游基线为 `27a8293c2d1e91077a48e79926868d6dc039dd3d`。该版本已经包含 `sglang_omni/models/qwen3_asr/mlx/`、MLX scheduler/runner 及 Apple Silicon 安装支持，OpenTypeless 直接复用它们，**没有重复实现 ASR 或引入 mlx-audio**。

```text
SwiftUI / AppKit
  ├─ AVAudioEngine → 16 kHz 单声道 PCM16 WAV
  ├─ 全局快捷键、Accessibility、条件恢复剪贴板
  └─ 私有 stdin/stdout JSON-lines worker
       ├─ 自主管理 SGLANG_USE_MLX=1 sgl-omni serve
       │    └─ /v1/audio/transcriptions → 原生 Qwen3-ASR MLX
       └─ MLX-LM → 本地文本整理 / 翻译 / 编辑 / 问答
```

ASR 使用 `mlx-community/Qwen3-ASR-0.6B-4bit`，文本处理使用 `mlx-community/Qwen3-1.7B-4bit`，模型版本固定在后端代码中。默认先串行处理一段音频，再整理文本；模型在进程内保持加载，减小后续请求等待时间。长音频分块由 SGLang-Omni 负责。暂不展示录音中的实时部分转写。

worker 为应用私有进程，不对外暴露控制 API。原生 SGLang-Omni 服务绑定随机 `127.0.0.1` 端口，仅用于本机 ASR，不暴露到局域网；当前上游推理接口无身份验证，同机进程可以访问该服务。退出、取消和 worker 终止会清理所属服务进程组。首次准备模型最多等待 30 分钟，普通请求最多等待 10 分钟，超时可以重试。

选中文本仅在 Voice edit / Ask 所需的请求中供本地模型使用。为防止写入错误位置，应用会在内存中比较目标字段的辅助功能内容、窗口和选择状态；某些编辑器的字段内容可能包含整篇文档，这些内容不保存、不发送给模型。不读取屏幕截图或浏览历史。辅助功能识别为密码框时拒绝录音；若无法识别目标、光标或选择发生变化，保留结果供手动复制。部分网页、自绘编辑器、远程桌面可能不暴露可靠的 Accessibility 信息，不能保证自动插入。

历史和设置保存于：

```text
~/Library/Application Support/OpenTypeless/library.json
~/Library/Application Support/OpenTypeless/Audio/
```

目录权限 `0700`，数据文件 `0600`，采用原子写入，不提供额外的磁盘加密。最多保留 1,000 条记录，支持 24 小时、7 天、30 天、1 年、永久；关闭历史会删除已有记录，关闭音频保留会删除已保留的音频。失败录音仅在当前会话暂存用于重试，退出时删除。崩溃或强制退出可能留下系统临时文件。模型下载缓存位于 Hugging Face 的标准缓存目录。应用不含分析埋点。

## 开发与构建

```bash
# 已有 Python 运行环境，只编译应用
OPENTYPELESS_PYTHON=/absolute/path/to/python bash openTypeless/scripts/build.sh

# 单元与集成测试（无需下载模型、无需麦克风权限）
bash openTypeless/scripts/test.sh

# 真实 MLX 链路：生成本地测试音频，验证识别、整理、翻译、编辑及问答
openTypeless/.venv/bin/python openTypeless/backend/smoke.py

# 使用自己的 WAV 做真实识别测试
openTypeless/.venv/bin/python openTypeless/backend/smoke.py --audio /absolute/path/to/audio.wav
```

开发模式可用 `CONFIGURATION=debug` 构建。应用 bundle 内携带 worker 源码，Info.plist 记录 Python 环境的绝对路径；当前构建是源码开发分发，不是携带全部 Python 和模型的独立安装包。迁移至另一台 Mac 时运行 setup，或者在设置里指定该机已安装的运行环境。不要移动或删除正在使用的仓库/虚拟环境。

`build.sh` 默认使用 ad-hoc 签名供本地运行。公开分发需要设置 `CODE_SIGN_IDENTITY`，使用自己的 Developer ID 签名并完成 Apple notarization。这里没有上传或发布任何版本。

测试覆盖协议分帧、退出/取消、音频重采样和时长上限、CSV 格式、词典、历史保留、损坏数据保护、后端请求校验、静音和失败恢复。辅助功能权限、麦克风硬件以及各个第三方应用的插入兼容性需要在授权后的真实桌面上验收。

### 本机验证记录

2026-09-17，在 16 GB Apple Silicon Mac、macOS 26.6.2、Swift 6.4、Python 3.12.13 上验证：

- 11 项 Swift 测试和 15 项 Python 测试通过；release 应用编译、ad-hoc 签名校验和应用自身窗口渲染通过。
- 缓存模型后设置 `HF_HUB_OFFLINE=1`，通过真实 SGLang-Omni MLX 识别与本地文本模型测试。英语测试音频由系统 Samantha 语音生成，识别结果完整包含 “The quick brown fox jumps over the lazy dog. Please send the report tomorrow.”。
- 中文语音混合技术词，在提供 SGLang 词汇提示后识别为“你好，请在明天发送报告。我们使用SGLang做性能优化。”。
- 中英法口头语整理保留原语言；法语翻译、将选中文字中的 ten 改为 eleven、问答均得到预期结果。最后一轮首次英语识别约 9.8 秒（含服务启动），首次整理约 3.5 秒（含文本模型加载），随后四个短文本请求约 0.5–1.0 秒。这是少量合成语音和文本样本的功能检查，不代表实际口音、噪声或长文本的质量与延迟保证。

可重复的真实模型检查：`HF_HUB_OFFLINE=1 openTypeless/.venv/bin/python openTypeless/backend/smoke.py`。模型需已下载；该脚本不会录制麦克风。尚未完成真实麦克风、全局快捷键及第三方应用插入的交互验收。

## 故障定位

- **没有快捷键响应**：在系统设置允许 OpenTypeless 的辅助功能访问，再重启应用；开发中重新签名可能需要移除旧授权后重新添加。
- **麦克风不可用**：允许麦克风访问，确认设置中选定设备仍连接。系统默认设备会在下一次录音时读取。
- **模型启动失败**：先运行 `scripts/setup.sh`；确认 Python 为 3.12、`ffmpeg@7` 可用，及 Hugging Face 可访问。代理环境需支持 HTTPX 的 SOCKS 依赖，setup 已包含。
- **编辑/翻译失败**：应用不会把错误语言的原始识别结果自动写入。可修复运行环境后重试，或复制已恢复的原始转写。
- **历史文件损坏**：应用会保留原文件并停止覆盖，显示具体路径。备份该文件后再手工修复或迁走它。

许可证：[Apache-2.0](../LICENSE)。模型权重和上游依赖遵循各自许可证。
