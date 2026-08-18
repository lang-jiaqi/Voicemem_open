"""3D-Speaker ERes2Net 声纹向量常驻 worker。

voicemem 主程序（另一个 conda 环境，没装 funasr/modelscope）通过 subprocess
启动这个脚本，用「一行路径 -> 一行 JSON」的协议跨环境拿声纹向量：

    请求（stdin，一行一个）：  /path/to/audio.wav
    响应（stdout，一行一个）：  {"embedding": [0.01, -0.02, ...]}   # 192 维
                              或 {"error": "..."}

启动完成后先打印一行 "READY"，调用方据此确认模型已加载完毕。
所有非协议输出（下载进度条、版本检查提示等）必须走 stderr，不能污染 stdout。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def main() -> None:
    import soundfile as sf
    import sherpa_onnx
    from scipy.signal import resample_poly

    # 保留旧的 device 参数兼容性；sherpa-onnx 的 CPU 推理已足够快。
    _device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    model_path = os.environ.get(
        "VOICEMEM_SPEAKER_MODEL",
        str(
            __import__("pathlib").Path(__file__).resolve().parents[2]
            / "service" / "models"
            / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
        ),
    )
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model_path, num_threads=2)
    )
    window = int(float(os.environ.get("VOICEMEM_SPEAKER_WINDOW", "3.0")) * 16000)
    hop = int(float(os.environ.get("VOICEMEM_SPEAKER_HOP", "1.5")) * 16000)

    print("READY", flush=True)

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        if path == "__exit__":
            break
        try:
            audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if sample_rate != 16000:
                audio = resample_poly(audio, 16000, int(sample_rate))
            audio = np.asarray(audio, dtype=np.float32)
            # 一整段只取一个向量很容易被开头噪声/尾部静音带偏；切成短窗，
            # 用 RMS 门控过滤静音，再平均多个有效窗口。
            if len(audio) <= window:
                chunks = [audio]
            else:
                chunks = [audio[i:i + window] for i in range(0, len(audio) - window + 1, hop)]
            embeddings = []
            for chunk in chunks:
                if len(chunk) < 16000 or float(np.sqrt(np.mean(chunk * chunk))) < 0.005:
                    continue
                stream = extractor.create_stream()
                stream.accept_waveform(16000, chunk)
                stream.input_finished()
                if extractor.is_ready(stream):
                    embeddings.append(np.asarray(extractor.compute(stream), dtype=np.float32))
            if not embeddings:
                raise RuntimeError("speaker embedding 未达到最小语音长度")
            vec = np.mean(embeddings, axis=0)
            vec /= np.linalg.norm(vec) + 1e-8
            vec = vec.tolist()
            print(json.dumps({"embedding": vec}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
