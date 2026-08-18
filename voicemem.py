"""voicemem.py — voicemem 的前台：把所有能力封装成扁平、直接调用的函数。

这是"简洁好理解、直接调用"的一层；真正的实现（后台/引擎）在 voicemem_core 包里。
每个函数 = 一个能力，参数直白，返回普通 dict / list / str。

用法（前台就是 `import voicemem`，所有 demo 统一走这里）::

    import voicemem
    voicemem.ingest("中午和 Alex 吃了拉面")
    voicemem.search("我中午吃了什么")
"""
from functools import lru_cache
from voicemem_core import VoiceMem, build_memory_context


@lru_cache(1)
def _vm(): return VoiceMem()


def ingest(text, speaker="Speaker 0", audio_path=None):
    """记住一句话（带 audio_path 时顺带跑场景/声纹/情绪感知）。"""
    return _vm().Ingest(text, speaker=speaker, audio_path=audio_path)

def preprocess(text, audio_path=None):
    """只跑流式预处理，拿信号但不写记忆 → {scene, speaker_id, emotion}。"""
    p = _vm().preprocess(text, audio_path=audio_path)
    return {"scene": p.scene_tag or "", "speaker_id": p.person_id or "", "emotion": p.emotion or ""}

def search(text, slots=None, entities=None):
    """检索记忆，返回 [{text, score}]。"""
    r = _vm().Search(text, slots=slots, entities=entities)
    return [{"text": h.text, "score": round(h.score, 3)} for h in r.hits]

def build_context(text):
    """检索并渲染成可塞进大模型 prompt 的记忆上下文文本。"""
    return build_memory_context(_vm().Search(text))

def classify(text):
    """把一句问题分类到记忆槽位 + 命名实体 → {slots, entities}。"""
    c = _vm().Classify(text)
    return {"slots": list(c.slots), "entities": list(c.entities)}

def create_scene_reminder(text):
    """从"到公交上提醒我…"这类话里建一条场景触发提醒 → {created, scene, message}。"""
    return _vm().CreateSceneTrigger(text)

def ingest_environment(audio_path):
    """只做背景声学场景识别（不记文字）。"""
    return _vm().IngestEnv(audio_path=audio_path)

def flush():
    """会话结束时调一次：触发右脑归因 + 认知图检查点等批量结账。"""
    _vm().Flush()

def self_check():
    """逐组件测速自检，返回报告文本。"""
    from voicemem_core.startup_check import run_startup_check
    return run_startup_check(_vm()).render()
