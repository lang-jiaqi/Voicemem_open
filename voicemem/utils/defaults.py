"""voicemem 各能力的内置默认实现工厂（util 名 -> 无参工厂）。

core.py 的 Utils 用它建默认；传函数给 VoiceMem(embedding=..., schema=...) 即覆盖对应项。
放这里而不是 core.py，是让顶层门面只讲「系统骨架」，不被这些具体默认实现的 import 撑大。
"""
from __future__ import annotations

import os


def default_utils(base_url, memory_root):
    def embedding():
        from voicemem.leftbrain.local_memory_store import OpenAILocalEmbedder, OpenAILocalEmbedderConfig
        return OpenAILocalEmbedder(OpenAILocalEmbedderConfig(base_url=base_url))
    def schema():
        from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QuerySlotClassifier
        return QuerySlotClassifier()
    def entity():
        from voicemem.leftbrain.cognitive_graph.annotator import CognitiveAnnotator, CognitiveAnnotatorConfig
        return CognitiveAnnotator(CognitiveAnnotatorConfig(base_url=base_url))
    def emotion():
        from voicemem.utils.audio.emotion.paper_emotion_detector import PaperAlignedEmotionDetector
        return PaperAlignedEmotionDetector()
    def voiceprint():
        from voicemem.utils.audio.voiceprint.speaker_encoder import SpeakerEncoder
        return SpeakerEncoder(device="cpu")
    def asr():
        from voicemem.utils.audio.asr import StreamingASR
        d = os.environ.get("VOICEMEM_MODELS_DIR", "models")
        return StreamingASR(f"{d}/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20")
    def memory_engine():
        from pathlib import Path
        from voicemem.leftbrain.mem0_backend_store import Mem0BackendStore
        return Mem0BackendStore(embedding(), memory_root=Path(memory_root or "results/voice_memory"))
    return {"embedding": embedding, "schema": schema, "entity": entity, "emotion": emotion,
            "voiceprint": voiceprint, "asr": asr, "memory_engine": memory_engine}
