# 打包成可下载的 Mac `.app`

目标：用户从网页下载一个 `.dmg` → 拖进「应用程序」→ 双击图标 → 出现浮动光球。
app 在**本机**跑（自带后端），首启填一次 `OPENAI_API_KEY`；AI 部分仍联网调
OpenAI（不是离线）。

整条链路分三层，**必须在一台真 Mac 上构建**（带显示器/麦克风，装好 voicemem
和后端依赖）：

```
① PyInstaller  backend/main.py + 依赖 + 本地模型
                        → backend/dist/voicemem-backend/（自包含可执行）
② electron-builder  Electron 壳 + ①的产物
                        → desktop/release/VoiceMem-0.1.0.dmg（带图标）
③ 签名 + 公证    让 Gatekeeper 不拦「未知开发者」
```

---

## 前置

```bash
# 在装好 voicemem + 后端依赖的那个 Python 环境里（见 ../README.md 的 Full local setup）
cd openai_voice_demo/backend
pip install pyinstaller
# 本地 ASR / 嵌入模型（onedir 里要能找到，或首启联网下载）
bash ../../scripts/download_models.sh ../../service/models
```

## ① 冻结后端（最容易反复的一层）

```bash
cd openai_voice_demo/backend
pyinstaller voicemem_backend.spec --noconfirm
# 产物：backend/dist/voicemem-backend/voicemem-backend
```

**先单独验证它能起来**（Electron 之前）：

```bash
cd dist/voicemem-backend
OPENAI_API_KEY=sk-... VOICE_DEMO_PORT=8787 ./voicemem-backend
# 看到 "Uvicorn running on http://0.0.0.0:8787" 才算成功
```

失败几乎都是 PyInstaller 漏收模块/数据（`ModuleNotFoundError` /
`FileNotFoundError`）。按报错往 `voicemem_backend.spec` 的 `hiddenimports` /
`collect_all` 里补，重跑，直到能起 uvicorn。**这一步无法在别处代劳。**

> 模型很大（本地 ASR ~570MB、E5 ~450MB，音频原生模型可达数 GB）。可以打进
> app（`.dmg` 会很大），也可以让后端首启时下载到 `~/Library/Application Support`。

## ② 打成 .dmg

```bash
cd openai_voice_demo/desktop
# 图标：准备一张 1024×1024 PNG，转成 build/icon.icns
build/make-icns.sh path/to/icon-1024.png
npm install
npm run dist
# 产物：desktop/release/VoiceMem-0.1.0.dmg
```

`package.json` 的 `build.extraResources` 会把 ①的 `backend/dist/voicemem-backend/`
整个拷进 `VoiceMem.app/Contents/Resources/backend/`。打包版的 `main.js` 会去启动
`Resources/backend/voicemem-backend`（而不是 `python3`）。

## ③ 签名 + 公证（对外分发必须）

未签名的 app 用户下载后会被 Gatekeeper 拦。需要 Apple Developer 账号：

```bash
export CSC_LINK=path/to/DeveloperID.p12 CSC_KEY_PASSWORD=...
export APPLE_ID=you@example.com APPLE_APP_SPECIFIC_PASSWORD=xxxx APPLE_TEAM_ID=XXXXXXXXXX
npm run dist        # electron-builder 会自动签名并公证
```

没有开发者账号只想本机自测：`npm run dist` 出的未签名 app，右键 →「打开」可绕过一次。

---

## 首启体验

打包版首次双击：弹出一个小窗要 `OPENAI_API_KEY` → 存到
`~/Library/Application Support/VoiceMem/config.json` → 之后不再问。改 key 就删这个文件。

## 现状 / 边界

- 第 ②③ 层（Electron 打包配置、图标脚本、entitlements、首启弹窗、打包版启动
  逻辑）已在仓库里就绪。
- 第 ① 层给了 `voicemem_backend.spec` 作为**起点**，torch 系依赖的 PyInstaller
  收集几乎肯定需要在你的 Mac 上迭代补全——这部分没法预先验证。
