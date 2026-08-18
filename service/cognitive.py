"""认知服务：原来是独立的 Flask 进程（cognitive_server.py），通过 HTTP 被
assistant.py 调用。现在直接进程内调用 backend/voicemem，不走 HTTP/JSON 序列化
——省掉一次网络往返，也不用再把音频编码成 base64 wav 发过去、对方再解码存盘。

跟原来 Flask 版本的行为差异：
  - 不再"echo"请求字段（global_id/speaker_id/...）——调用方本来就有这些值，
    没必要在返回值里绕一圈。
  - _write_wav 直接吃 numpy 数组写文件，不经过 base64。
其余逻辑（VoiceMem 按 person_id 分池、prepare/ready 两阶段、search 用
Future 缓存实现"prepare 先起、ready 再取"的预取模式、history 落盘、
info/info_detail 组装）跟原来一致。
"""
from __future__ import annotations

import concurrent.futures
import json
import threading
import time
import wave
from pathlib import Path

import numpy as np

from voicemem_core import VoiceMem

DEFAULT_USER_ID = "voice_user"

_SLOT_ZH_TO_EN: dict[str, str] = {
    "工作": "work",       "职业": "work",
    "财务": "finance",    "金钱": "finance",
    "人际": "relationships", "关系": "relationships",
    "健康": "health",
    "目标": "goals",      "计划": "goals",
    "日常": "daily_life", "生活": "daily_life",
    "知识": "knowledge",  "学习": "knowledge",
}


def _normalize_slots(slots: list[str]) -> list[str]:
    return [_SLOT_ZH_TO_EN.get(s, s) for s in slots]


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """float32 [-1,1] 音频直接写 wav 文件，不经过 base64。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def _assemble(search_result, ingest_result: dict,
              trigger_created: dict | None = None,
              playback: dict | None = None) -> tuple[str, dict]:
    """(SearchResult, Ingest() 返回 dict, ...) -> (info, info_detail)"""
    left_lines = [f"- {h.text}" for h in search_result.hits] if search_result.hits else []
    left_info = "\n".join(left_lines) if left_lines else "(暂无相关记忆)"
    rb_info = search_result.rb_directive or ""

    parts = []
    if search_result.prestimulus_text:
        parts.append(search_result.prestimulus_text)
    if left_lines:
        parts.append(left_info)
    if rb_info:
        parts.append(rb_info)
    info = "\n\n".join(parts) if parts else "(暂无信息)"

    audiomem_info = {
        "current_scene":         ingest_result.get("current_scene") or "",
        "speaker_id":            ingest_result.get("speaker_id") or "",
        "recognized_tune":       ingest_result.get("recognized_tune"),
        "abnormal_sounds":       ingest_result.get("abnormal_sounds") or [],
        "recognized_place":      ingest_result.get("recognized_place"),
        "familiar_place_prompt": ingest_result.get("familiar_place_prompt"),
        "new_routine":           ingest_result.get("new_routine"),
        "triggered_reminders":   ingest_result.get("triggered_reminders") or [],
        "scene_trigger_created": (
            {"scene": trigger_created["scene"], "message": trigger_created["message"]}
            if trigger_created and trigger_created.get("created") else None
        ),
        "playback": playback,
    }

    return info, {
        "left_brain_info":  left_info,
        "right_brain_info": rb_info,
        "audiomem_info":    audiomem_info,
    }


class CognitiveService:
    """替代原来的 cognitive_server.py（Flask + HTTP）。"""

    def __init__(self, memory_root: Path, history_root: Path, *, base_url: str | None = None):
        self.memory_root = memory_root
        self.history_root = history_root
        self.history_root.mkdir(parents=True, exist_ok=True)
        self._base_url = base_url

        self._vm_pool: dict[str, VoiceMem] = {}
        self._vm_lock = threading.Lock()
        self._search_cache: dict[str, concurrent.futures.Future] = {}
        self._cache_lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._history_lock = threading.Lock()

    def _get_vm(self, person_id: str = DEFAULT_USER_ID) -> VoiceMem:
        with self._vm_lock:
            if person_id not in self._vm_pool:
                self._vm_pool[person_id] = VoiceMem(
                    memory_root=self.memory_root / person_id,
                    user_id=person_id,
                    base_url=self._base_url,
                )
        return self._vm_pool[person_id]

    def flush_all(self) -> None:
        """会话/连接正式结束时调用一次——补跑每个已实例化 VoiceMem 的
        session-boundary 批处理（Algorithm 1 子图判定 + 右脑长短期归因，
        见 core.py::VoiceMem.Flush）。person_id 目前实际使用中始终是
        DEFAULT_USER_ID（见上面注释：多人分流的分支从没真正触发过），但
        这里遍历整个 _vm_pool 而不是只 flush 默认用户，万一以后真的有
        多人场景也不会漏。之前这里完全没人调用过 Flush()，最后一个
        session 的批处理永远等不到"下一个 session 开始"这个触发信号，
        实际上永远不会跑。"""
        with self._vm_lock:
            pool = dict(self._vm_pool)
        for person_id, vm in pool.items():
            try:
                vm.Flush()
            except Exception as e:
                print(f"[flush_all] Flush failed for {person_id}: {e}")

    def _save_history(self, global_id: str, record_update: dict) -> None:
        path = self.history_root / f"{global_id}.json"
        with self._history_lock:
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
            else:
                record = {"global_id": global_id, "states": []}
            record["states"].append(record_update)
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 200ms: prepare ----------
    def prepare(self, *, global_id: str, transcript: str, entities: list[str],
                slots: list[str], emotion: str, person_id: str = DEFAULT_USER_ID) -> dict:
        if not transcript:
            return {"status": "prepare_fail:empty_transcript"}

        def _do_search():
            return self._get_vm(person_id).Search(
                query=transcript,
                slots=_normalize_slots(slots),
                entities=entities,
                emotion=emotion,
            )

        future = self._executor.submit(_do_search)
        with self._cache_lock:
            self._search_cache[global_id] = future

        self._save_history(global_id, {"state": "prepare", "status": "prepare_successful"})
        print(f"[prepare] global_id={global_id}  slots={slots}  entities={entities}")
        return {"status": "prepare_successful"}

    def _finish_ingest(self, *, global_id: str, transcript: str, speaker_id: str,
                       entities: list[str], emotion: str, audio: np.ndarray | None,
                       sample_rate: int, person_id: str, search_result) -> None:
        """慢路径：音频理解与记忆写入，不阻塞当前轮 Gemini 首音。"""
        # 场景触发提醒的注册（比如"到公交上提醒我打电话给妈妈"）
        trigger_created: dict = {}
        if transcript:
            try:
                trigger_created = self._get_vm(person_id).CreateSceneTrigger(transcript)
            except Exception as e:
                print(f"[scene_trigger] create failed: {e}")

        # 场景/声纹/音乐/异常声/地点/规律依赖真实音频，全部放到后台写入。
        ingest_result: dict = {}
        if transcript:
            wav_path = None
            if audio is not None and len(audio):
                wav_path = self.memory_root / person_id / "raw_audio" / f"{global_id}.wav"
                _write_wav(wav_path, audio, sample_rate)
            try:
                ingest_result = self._get_vm(person_id).Ingest(
                    transcript, speaker_id, emotion, entities,
                    audio_path=str(wav_path) if wav_path else None,
                )
            except Exception as e:
                print(f"[ingest] error: {e}")

        # 原声回放——跟这轮 Ingest() 没关系，是在找"之前"某条记忆的原始录音。
        playback: dict | None = None
        if transcript:
            try:
                playback = self._get_vm(person_id).TryPlayback(transcript)
            except Exception as e:
                print(f"[playback] error: {e}")

        info, info_detail = _assemble(search_result, ingest_result, trigger_created, playback)
        result = {
            "status":      "search_end_success",
            "info":        info,
            "info_detail": info_detail,
            "timing":      search_result.timing,
        }
        self._save_history(global_id, {
            "state": "ready", **result,
            "ingest": {
                "memory_ids":  ingest_result.get("memory_ids", []),
                "facts_count": ingest_result.get("facts_count", 0),
            },
        })
        print(
            f"[ingest-bg] global_id={global_id}  "
            f"hits={len(search_result.hits)}  rb={'yes' if search_result.rb_directive else 'no'}  "
            f"scene={ingest_result.get('current_scene')}"
        )

    # ---------- ready：检索结果走关键路径，音频写入走后台 ----------
    def ready(self, *, global_id: str, transcript: str, speaker_id: str,
              entities: list[str], slots: list[str], emotion: str,
              audio: np.ndarray | None = None, sample_rate: int = 16000,
              person_id: str = DEFAULT_USER_ID) -> dict:
        with self._cache_lock:
            future = self._search_cache.pop(global_id, None)

        if future is None:
            return {"status": "search_fail:no_prepare"}

        try:
            search_result = future.result(timeout=10.0)
        except concurrent.futures.TimeoutError:
            return {"status": "search_fail:timeout"}
        except Exception as e:
            return {"status": f"search_fail:{e}"}

        # 立刻返回检索到的左/右脑上下文，使 Gemini 可马上开始生成。此时没有
        # audiomem 信号是正常的；它们会在后台落盘，供后续轮次检索使用。
        info, info_detail = _assemble(search_result, {})
        result = {
            "status": "search_end_success",
            "info": info,
            "info_detail": info_detail,
            "timing": search_result.timing,
            "ingest_pending": True,
        }
        self._executor.submit(
            self._finish_ingest,
            global_id=global_id, transcript=transcript, speaker_id=speaker_id,
            entities=entities, emotion=emotion, audio=audio, sample_rate=sample_rate,
            person_id=person_id, search_result=search_result,
        )
        print(f"[ready-fast] global_id={global_id}  hits={len(search_result.hits)}")
        return result
