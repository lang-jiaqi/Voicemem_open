"""VoiceMem._reconcile_speaker_candidates —— 候选声纹自动回收的姓名冲突守卫。

VoiceprintStore 只看声学分数，不认姓名；真正的安全网在这里：两个 person_id
如果已经被明确报过不同的姓名，哪怕声学分数再高也绝不能强行合并（那正是
之前 Jennifer 冒领 Nancy 孤儿声纹那次事故的教训）。只有在没有姓名冲突时，
才允许把声学上收敛到同一个人的候选并回去，并把已知的姓名传播到合并后的
身份上。
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from voicemem.core import VoiceMem
from voicemem.voiceprint import l2norm
from voicemem.voiceprint_store import VoiceprintStore


def _noisy(rng: np.random.Generator, base: np.ndarray, std: float) -> np.ndarray:
    return l2norm(base + rng.standard_normal(base.shape[0]) * std)


def _build_split_pair(rng, true_voice, match_thr=0.50, cand_thr=0.40, store=None):
    """构造"同一个人被拆成两个 id"的声学状态，返回 (store, person_main, person_fork)。"""
    store = store or VoiceprintStore(Path(tempfile.mkdtemp()), match_threshold=match_thr, candidate_threshold=cand_thr)
    r1 = store.identify(_noisy(rng, true_voice, 0.08), context="intro")
    person_main = r1.person_id
    for i in range(2):
        store.identify(_noisy(rng, true_voice, 0.08), context=f"more{i}")

    cand_vec = None
    for _ in range(300):
        v = _noisy(rng, true_voice, 0.28)
        if 0.40 <= store._profiles[person_main].score(l2norm(v)) < 0.50:
            cand_vec = v
            break
    assert cand_vec is not None

    r2 = store.identify(cand_vec, context="noisy outlier, same person")
    person_fork = r2.person_id
    for i in range(4):
        v = _noisy(rng, true_voice, 0.08)
        store._profiles[person_fork].update(v, quality=1.0)
        store._meta[person_fork]["obs_count"] += 1
    store._save()
    return store, person_main, person_fork


class ReconcileSpeakerCandidatesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.vm = VoiceMem(
            memory_root=self.tmp, user_id="test",
            enable_scene=False, enable_music=False,
            enable_abnormal_sound=False, enable_voiceprint=True, enable_emotion=False,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_merges_when_no_name_conflict_and_propagates_name(self):
        rng = np.random.default_rng(7)
        true_voice = l2norm(rng.standard_normal(64))
        store, person_main, person_fork = _build_split_pair(
            rng, true_voice, store=self.vm._get_vp_store()
        )
        # 只有 fork 侧被自我介绍绑过名字（真实场景：拆分之后自报姓名先落在了
        # 新 id 上），person_main 还没有名字
        self.vm._get_registry().bind(person_fork, name="Nancy")

        resolved_person, resolved_speaker = self.vm._reconcile_speaker_candidates(
            person_fork, person_fork
        )

        self.assertEqual(resolved_person, person_main)
        self.assertEqual(resolved_speaker, person_main)
        self.assertNotIn(person_fork, store._profiles)
        self.assertEqual(self.vm._get_registry().display_name(person_main), "Nancy")

    def test_never_merges_across_different_confirmed_names(self):
        rng = np.random.default_rng(7)
        true_voice = l2norm(rng.standard_normal(64))
        store, person_main, person_fork = _build_split_pair(
            rng, true_voice, store=self.vm._get_vp_store()
        )
        # 两边都已经被明确报过不同的姓名——声学上很像，但这必须被当成两个人
        registry = self.vm._get_registry()
        registry.bind(person_main, name="Nancy")
        registry.bind(person_fork, name="Jennifer")

        resolved_person, resolved_speaker = self.vm._reconcile_speaker_candidates(
            person_fork, person_fork
        )

        self.assertEqual(resolved_person, person_fork, "姓名冲突时不该被重定向到另一个 id")
        self.assertIn(person_main, store._profiles)
        self.assertIn(person_fork, store._profiles)
        self.assertEqual(registry.display_name(person_main), "Nancy")
        self.assertEqual(registry.display_name(person_fork), "Jennifer")

    def test_merge_propagates_memory_tags(self):
        """merge_persons() 只合并声纹画像本身，不知道 memory_tags 表——真实
        bug：分叉之前用 fork_pid 抽取、打过 speaker:{fork_pid} 标签的记忆，
        合并之后如果不回填成 speaker:{into_pid}，speaker_filter 查 into_pid
        永远找不到这些记忆，声纹层面"认出是同一个人"跟检索层面就对不上。"""
        rng = np.random.default_rng(7)
        true_voice = l2norm(rng.standard_normal(64))
        store, person_main, person_fork = _build_split_pair(
            rng, true_voice, store=self.vm._get_vp_store()
        )
        self.vm._get_registry().bind(person_main, name="Nancy")

        # 直接建 CognitiveGraphStoreV2 写测试标签。memory_tags 对 memories.id
        # 有外键约束，先插两条最小的占位记忆。
        from types import SimpleNamespace
        from voicemem.leftbrain.cognitive_graph.store_v2 import CognitiveGraphStoreV2
        from voicemem.leftbrain.cognitive_graph.slot_v2 import SlotV2
        cog_store = CognitiveGraphStoreV2(self.vm._cognitive_db)
        cog_store.upsert_memory_record("test", "mem_1", SlotV2.KNOWLEDGE, "Nancy said something")
        cog_store.upsert_memory_record("test", "mem_2", SlotV2.KNOWLEDGE, "Nancy said something else")
        cog_store.upsert_memory_tags("mem_1", "test", [(f"speaker:{person_fork}", 1.0)])
        cog_store.upsert_memory_tags("mem_2", "test", [(f"speaker:{person_fork}", 1.0)])

        # _reconcile_speaker_candidates 内部走 self._get_repo()._cognitive_store
        # 做标签回填——真正的 _get_repo() 会连带建一个需要联网校验 OPENAI_API_KEY
        # 的 embedder，跟这条测试想验证的东西无关，预置缓存跳过它，直接把同一个
        # cog_store 接到 _get_repo() 返回值上。
        self.vm._cache["repo"] = SimpleNamespace(_cognitive_store=cog_store)
        self.vm._reconcile_speaker_candidates(person_fork, person_fork)

        tagged_into = set(cog_store.memory_ids_for_slots_v2("test", [f"speaker:{person_main}"]))
        tagged_fork = set(cog_store.memory_ids_for_slots_v2("test", [f"speaker:{person_fork}"]))
        self.assertEqual(tagged_into, {"mem_1", "mem_2"})
        self.assertEqual(tagged_fork, set())

    def test_unnamed_orphan_with_real_history_needs_more_than_bare_match_thr(self):
        """真实 bug（85轮多人 benchmark 复测时发现）：Nancy 自我介绍 session 里
        有一句话噪声偏大，被拆成了一个新 id（``person_cffbe7e7`` 那种），这个
        新 id 当时还没绑名字，但已经是 Nancy 真实说的一句话。后来 Jennifer 自
        我介绍时声音恰好也落在这个 id 的候选带里，cross_score=0.542——只比
        旧的 match_thr(0.50) 高一点点——就被自动合并、还把"Jennifer"焊死在了
        这个其实是 Nancy 声音的 id 上，导致 Nancy 后续所有真话都被记到
        Jennifer 名下。这里验证：即使 into 侧完全没绑名字（无法套用姓名冲突
        守卫），cross 分数没到新的、更高的 merge_thr(0.65) 时也不能合并——
        "还没绑名字"不等于"可以随便认领"，尤其它已经是某个真人说的一句话
        （不是空画像）。
        """
        rng = np.random.default_rng(9)
        D = 64
        voice_nancy = l2norm(rng.standard_normal(D))
        raw = rng.standard_normal(D)
        orth = l2norm(raw - np.dot(raw, voice_nancy) * voice_nancy)
        # 两人声音本身有点像（余弦 0.55），但绝不是同一个人——cross 会落在
        # match_thr(0.50) 和 merge_thr(0.65) 之间，正是旧代码会误合并的区间。
        voice_jennifer = l2norm(0.55 * voice_nancy + np.sqrt(1 - 0.55 ** 2) * orth)

        store = self.vm._get_vp_store()
        registry = self.vm._get_registry()

        # Nancy 那句噪声偏大的话，拆出了一个还没绑名字的孤儿 id（into 侧）。
        r_orphan = store.identify(_noisy(rng, voice_nancy, 0.05))
        person_orphan = r_orphan.person_id

        # Jennifer 自我介绍：声纹落在 person_orphan 的 candidate 带，建了候选
        # 记录，然后自己又攒了几条真实观测（分数比对 person_orphan 更稳）。
        cand_vec = None
        for _ in range(2000):
            v = _noisy(rng, voice_jennifer, 0.05)
            score = store._profiles[person_orphan].score(l2norm(v))
            if 0.40 <= score < 0.50:
                cand_vec = v
                break
        self.assertIsNotNone(cand_vec, "没能采样出目标相似度的候选向量，调整噪声参数")
        r_fork = store.identify(cand_vec)
        person_jennifer = r_fork.person_id
        self.assertNotEqual(person_jennifer, person_orphan)
        registry.bind(person_jennifer, name="Jennifer")

        # 只加 1 条（obs_count: 1 -> 2），刻意留在 <3——真实事故里 fork 侧
        # 触发合并检查那一刻 obs_count 正是 2（自我介绍 1 条 + 紧接着 1 条真实
        # match）。obs_count>=3 会命中 thin_and_stuck 那条专门放宽的路径，
        # 这里要测的是"两边都还没攒到 thin_and_stuck 门槛"的一般情况，得避开它。
        v = _noisy(rng, voice_jennifer, 0.05)
        store._profiles[person_jennifer].update(v, quality=1.0)
        store._meta[person_jennifer]["obs_count"] += 1
        store._save()
        self.assertLess(store.get_meta(person_jennifer)["obs_count"], 3)

        resolved_person, resolved_speaker = self.vm._reconcile_speaker_candidates(
            person_jennifer, person_jennifer
        )

        self.assertEqual(
            resolved_person, person_jennifer,
            "cross 分数不到 merge_thr 时不该被合并，哪怕 into 侧还没绑名字",
        )
        self.assertIn(person_orphan, store._profiles)
        self.assertIn(person_jennifer, store._profiles)
        self.assertNotEqual(registry.display_name(person_orphan), "Jennifer")

    def test_thin_and_stuck_relaxation_blocked_by_conflicting_session_pin(self):
        """Fix B（merge_threshold）只收紧了"两边都不是 thin_and_stuck"的一般
        合并路径；thin_and_stuck 那条为了救"同一个人自己卡住的旧画像"而特意
        放宽到 cand_thr(0.40) 的路径完全没碰，且它没有任何信号能分辨"这是我
        自己的旧碎片"和"这是另一个人的碎片"——merge_thr 收紧之后复测
        benchmark，就是从这条路径漏过去的：Nancy 自我介绍 session（key_003）
        里一句噪声话拆出的孤儿声纹 obs_count 一直卡在 1（thin，因为后续每次
        都被判给了 Nancy 自己那个更稳的主画像），Jennifer 在另一个 session
        （key_011）自我介绍后攒到 3 条真实观测（stuck 判定里的 fork），
        cross=0.542 只到 cand_thr 就命中了 thin_and_stuck，把"Jennifer"焊死
        在了 Nancy 的孤儿碎片上。这里验证 core.py 新加的第二独立信号——孤儿
        碎片的出生 session (key_003) 早就被自我介绍钉死成了另一个人（Nancy
        自己的主 id，既不是孤儿也不是 Jennifer）——能在 thin_and_stuck 放宽
        阈值的情况下依然否决这次合并。
        """
        rng = np.random.default_rng(11)
        D = 64
        voice_nancy = l2norm(rng.standard_normal(D))
        raw1 = rng.standard_normal(D)
        orth = l2norm(raw1 - np.dot(raw1, voice_nancy) * voice_nancy)
        raw2 = rng.standard_normal(D)
        orth2 = l2norm(
            raw2 - np.dot(raw2, voice_nancy) * voice_nancy - np.dot(raw2, orth) * orth
        )
        # 两人声音底子上有点像（余弦 0.35，低于 cand_thr，本身不会互相误判）。
        voice_jennifer = l2norm(0.35 * voice_nancy + np.sqrt(1 - 0.35 ** 2) * orth)

        store = self.vm._get_vp_store()
        registry = self.vm._get_registry()

        # Nancy 自我介绍 session（key_003）：先建好她自己的主 id 并绑名、钉 pin。
        r_main = store.identify(_noisy(rng, voice_nancy, 0.05), context="Hi I'm Nancy")
        person_nancy_main = r_main.person_id
        self.vm._person_origin_session[person_nancy_main] = "key_003"
        registry.bind(person_nancy_main, name="Nancy")
        self.vm._session_person_pin["key_003"] = person_nancy_main

        # 同一个 session 里紧接着一句噪声偏大的话，跟刚才那个只有 1 条观测、
        # 还很薄的主画像对不上（cos=0.30 < cand_thr），被拆成了一个还没绑
        # 名字的孤儿声纹——这个孤儿其实还是 Nancy 说的，只是这一句噪声大、
        # 偏巧又在 orth/orth2 方向上跟 Jennifer 的声音更接近一点（cos=0.45，
        # 现实里短句声纹分数本来就噪声很大，完全可能出现这种巧合）。
        p, q = 0.30, 0.3684
        r_coef = float(np.sqrt(1 - p ** 2 - q ** 2))
        orphan_vec = l2norm(p * voice_nancy + q * orth + r_coef * orth2)
        self.assertLess(
            store._profiles[person_nancy_main].score(orphan_vec), 0.40,
            "构造用例失效：孤儿向量不该匹配上 Nancy 主画像",
        )
        r_orphan = store.identify(orphan_vec, context="noisy Nancy utterance")
        person_orphan = r_orphan.person_id
        self.assertNotEqual(person_orphan, person_nancy_main)
        self.vm._person_origin_session[person_orphan] = "key_003"

        # Jennifer 在另一个 session（key_011）自我介绍：声纹恰好落在孤儿声纹
        # 的候选带（cos≈0.44，介于 cand_thr 和 match_thr 之间，且明显高于
        # 她自己声音跟 Nancy 主画像的分数），建了候选记录，然后自己又攒了
        # >=3 条真实观测——正是 thin_and_stuck（top_obs<=2, fork_obs>=3）的
        # 判定条件。
        jennifer_intro_vec = _noisy(rng, voice_jennifer, 0.02)
        self.assertGreater(
            store._profiles[person_orphan].score(jennifer_intro_vec),
            store._profiles[person_nancy_main].score(jennifer_intro_vec),
            "构造用例失效：Jennifer 这句应该更像孤儿声纹而不是 Nancy 主画像",
        )
        r_jennifer = store.identify(jennifer_intro_vec, context="Hi I'm Jennifer")
        person_jennifer = r_jennifer.person_id
        self.assertNotEqual(person_jennifer, person_orphan)
        self.vm._person_origin_session[person_jennifer] = "key_011"
        registry.bind(person_jennifer, name="Jennifer")
        self.vm._session_person_pin["key_011"] = person_jennifer

        for _ in range(2):
            v = _noisy(rng, voice_jennifer, 0.05)
            store._profiles[person_jennifer].update(v, quality=1.0)
            store._meta[person_jennifer]["obs_count"] += 1
        store._save()

        top_obs = store.get_meta(person_orphan)["obs_count"]
        fork_obs = store.get_meta(person_jennifer)["obs_count"]
        self.assertLessEqual(top_obs, 2)
        self.assertGreaterEqual(fork_obs, 3)  # 确认命中的确实是 thin_and_stuck 放宽路径

        resolved_person, resolved_speaker = self.vm._reconcile_speaker_candidates(
            person_jennifer, person_jennifer
        )

        self.assertEqual(
            resolved_person, person_jennifer,
            "孤儿碎片的出生 session 已被钉死成 Nancy，thin_and_stuck 放宽阈值也不该合并",
        )
        self.assertIn(person_orphan, store._profiles)
        self.assertIn(person_jennifer, store._profiles)
        self.assertNotEqual(registry.display_name(person_orphan), "Jennifer")

    def test_noop_when_no_candidates_pending(self):
        rng = np.random.default_rng(3)
        vp = self.vm._get_vp_store()
        r = vp.identify(l2norm(rng.standard_normal(64)))
        resolved_person, resolved_speaker = self.vm._reconcile_speaker_candidates(
            r.person_id, r.person_id
        )
        self.assertEqual(resolved_person, r.person_id)
        self.assertEqual(resolved_speaker, r.person_id)


if __name__ == "__main__":
    unittest.main()
