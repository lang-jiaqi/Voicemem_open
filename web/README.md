# web — voicemem 的 web demo

前台（把所有能力封装成的扁平函数）在**仓库根的 `voicemem.py`**，本目录只放 web 外壳：

| 文件 | 角色 |
|---|---|
| `../voicemem.py` | **前台**：`ingest / preprocess / search / build_context / classify / create_scene_reminder / ingest_environment / flush / self_check`。 |
| `server.py` | 极简 FastAPI 后端：把前台函数转发成 HTTP 接口，本身无业务逻辑。 |
| `index.html` | 前端（随你改）。只 POST `server.py` 的接口。 |

> 命名说明：根目录同时有 `voicemem/`（引擎包）和 `voicemem.py`（前台）。同名时
> `import voicemem` 只会拿到**包**，看不到那个文件（语言规则，目录包优先）。所以
> `server.py` 按**文件路径**加载 `voicemem.py`（见其顶部注释），既能用前台，又不影响
> `import voicemem` 正常拿到引擎包。

## 运行

```bash
pip install -e .            # voicemem 核心（从仓库根目录）
pip install fastapi uvicorn # web 后端依赖
export OPENAI_API_KEY=sk-...

cd web && python server.py  # http://localhost:8000/
```

## 接口一览

每个接口对应根 `voicemem.py` 里的一个函数：

| 接口 | 函数 | 作用 |
|---|---|---|
| `POST /api/ingest` | `ingest` | 记住一句话（带 `audio_path` 时顺带跑音频感知） |
| `POST /api/preprocess` | `preprocess` | 只跑流式预处理，拿场景/说话人/情绪信号 |
| `POST /api/search` | `search` | 检索记忆 |
| `POST /api/context` | `build_context` | 检索并渲染成可塞进 prompt 的上下文 |
| `GET /api/self-check` | `self_check` | 组件测速自检报告 |
