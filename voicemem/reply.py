"""回复层：核心交出 ``Turn`` 之后的那一步——两条路，一个口子。

``voicemem/stream.py`` 是输入侧（音频 → 记忆），这里是输出侧（记忆 → 回复）。两条路：

    # 路 A：用内置的（OpenAI 兼容 api，流式）
    vm = VoiceMem.from_config({"reply": {"provider": "openai",
                                         "config": {"model": "gpt-4o-mini"}}})

    # 路 B：用自己的模型/函数
    vm = VoiceMem(reply=my_fn)

两条路拿到的调用口完全一样::

    answer = await vm.reply(turn)                      # 收全，返回整串
    async for delta in vm.reply_stream(turn):  ...     # 流式，逐字吐

``my_fn`` 写成下面任意一种都行，``normalize()`` 会把它们统一成异步生成器::

    def       my_fn(text, memory_context) -> str          # 同步：自动丢线程，不阻塞事件循环
    async def my_fn(text, memory_context) -> str          # 协程
    async def my_fn(text, memory_context): yield delta    # 异步生成器（流式）

**TTS 不在这里。** 语音合成仍只在 web demo 里（``web/utils.py`` 的 ``tts_stream``），
核心的回复层只产出文本。
"""
from __future__ import annotations

import asyncio
import inspect
import os
from typing import AsyncIterator, Callable

# memory_context 只是「记得关于用户的哪些事」，本身不含人设/风格要求，所以内置
# provider 把它接在这句后面，而不是拿它整个当 system prompt。
DEFAULT_SYSTEM = "你是语音助手，简短自然地回答。"


def compose_system(memory_context: str, system: str | None = None) -> str:
    """人设 + 记忆 → system prompt。两边都可能为空。"""
    parts = [system or DEFAULT_SYSTEM]
    if memory_context:
        parts.append(memory_context)
    return "\n\n".join(parts)


def openai_reply(model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, system: str | None = None) -> Callable:
    """内置回复 provider：OpenAI 兼容 api，流式吐字。返回一个异步生成器函数。

    模型默认取 ``OPENAI_CHAT_MODEL``，再回落 ``gpt-4o-mini``。client 首次调用时才建，
    ``import voicemem`` 不会因此要求有 key。
    """
    client = None

    async def fn(text: str, memory_context: str = "") -> AsyncIterator[str]:
        nonlocal client
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
            )
        stream = await client.chat.completions.create(
            model=model or os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini",
            stream=True,
            messages=[{"role": "system", "content": compose_system(memory_context, system)},
                      {"role": "user", "content": text}],
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return fn


def normalize(fn: Callable) -> Callable:
    """把任意形状的回复函数规格化成「异步生成器函数」这一种。

    同步函数走 ``asyncio.to_thread``——回复生成是秒级的，直接在事件循环里跑会卡住
    读麦克风那条线。返回值若本身是异步可迭代对象（例如一个包装别人生成器的
    lambda），照样按流式展开。
    """
    if inspect.isasyncgenfunction(fn):
        return fn

    if inspect.iscoroutinefunction(fn):
        async def gen(text: str, memory_context: str = "") -> AsyncIterator[str]:
            out = await fn(text, memory_context)
            if hasattr(out, "__aiter__"):
                async for delta in out:
                    yield delta
            else:
                yield out
        return gen

    async def gen(text: str, memory_context: str = "") -> AsyncIterator[str]:
        out = await asyncio.to_thread(fn, text, memory_context)
        if hasattr(out, "__aiter__"):
            async for delta in out:
                yield delta
        else:
            yield out
    return gen


def unpack(turn_or_text, memory_context: str = "") -> tuple[str, str]:
    """``vm.reply(turn)`` 的便利：Turn / StreamState 直接拆成 (text, memory_context)。"""
    text = getattr(turn_or_text, "text", None)
    if text is not None and hasattr(turn_or_text, "memory_context"):
        return text, (memory_context or turn_or_text.memory_context)
    return turn_or_text, memory_context
