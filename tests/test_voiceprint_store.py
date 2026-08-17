"""VoiceprintStore 候选自动回收（find_resolvable_candidates / merge_persons）。

真实 bug 场景：短句声纹分数噪声很大，同一个人紧接着的两句话可能因为分数
差一点没到 match_thr 而被拆成一个新的 person_id（"candidate" 分支）。之前
这个拆分是永久的——候选记录只是写进 voiceprint_meta.json 里的死数据，
verify_candidate() 是个永远返回 pending 的桩，没人调用。这里测的是新加的
自动回收机制：等被拆出去的声纹自己也攒够几条真实观测后，拿两边攒起来的
画像互相打分（比单条向量的噪声小得多），真的收敛到同一个人才自动并回去；
如果两边其实是声音相似但真不是同一个人，攒得越多分数应该越低，绝不会被
误合并。
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from voicemem.voiceprint import l2norm
from voicemem.voiceprint_store import VoiceprintStore


def _noisy(rng: np.random.Generator, base: np.ndarray, std: float) -> np.ndarray:
    return l2norm(base + rng.standard_normal(base.shape[0]) * std)


class VoiceprintStoreCandidateMergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = VoiceprintStore(self.tmp, match_threshold=0.50, candidate_threshold=0.40)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_same_person_outlier_gets_merged_back(self):
        """同一个人的一句噪声偏大的话被拆成新 id 后，后续真实观测应该把它并回去。"""
        rng = np.random.default_rng(7)
        D = 64
        true_voice = l2norm(rng.standard_normal(D))

        r1 = self.store.identify(_noisy(rng, true_voice, 0.08), context="intro")
        self.assertEqual(r1.action, "new")
        person_a = r1.person_id
        for i in range(2):
            r = self.store.identify(_noisy(rng, true_voice, 0.08), context=f"more{i}")
            self.assertEqual(r.action, "match")
            self.assertEqual(r.person_id, person_a)

        # 找一条落在 candidate 带（同一个人，但这一句噪声偏大）的观测
        cand_vec = None
        for _ in range(300):
            v = _noisy(rng, true_voice, 0.28)
            score = self.store._profiles[person_a].score(l2norm(v))
            if 0.40 <= score < 0.50:
                cand_vec = v
                break
        self.assertIsNotNone(cand_vec, "没能采样出一条落在 candidate 带的向量，调整噪声参数")

        r2 = self.store.identify(cand_vec, context="noisy outlier, still same person")
        self.assertEqual(r2.action, "new")
        person_fork = r2.person_id
        self.assertNotEqual(person_fork, person_a)
        self.assertEqual(len(self.store._candidates), 1)

        # 在合并之前，从 fork 自己的角度看，还没有可回收的候选（只有一条孤立观测）
        self.assertEqual(self.store.find_resolvable_candidates(person_fork), [])

        # 同一个人后续几句真实话，恰好都落在了这个新 id 下（真实 bug 里就是这样：
        # 拆出来的孤儿声纹后面反而更常被匹配上）
        for i in range(4):
            v = _noisy(rng, true_voice, 0.08)
            self.store._profiles[person_fork].update(v, quality=1.0)
            self.store._meta[person_fork]["obs_count"] += 1
        self.store._save()

        resolvable = self.store.find_resolvable_candidates(person_fork)
        self.assertEqual(len(resolvable), 1)
        cid, absorb_pid, into_pid, cross = resolvable[0]
        self.assertEqual(absorb_pid, person_fork)
        self.assertEqual(into_pid, person_a)
        self.assertGreaterEqual(cross, 0.50)

        prior_obs_a = self.store.get_meta(person_a)["obs_count"]
        prior_obs_fork = self.store.get_meta(person_fork)["obs_count"]

        self.store.merge_persons(absorb_pid, into_pid)

        self.assertNotIn(person_fork, self.store._profiles)
        self.assertIn(person_a, self.store._profiles)
        self.assertEqual(self.store._candidates, {})
        self.assertEqual(
            self.store.get_meta(person_a)["obs_count"], prior_obs_a + prior_obs_fork
        )
        self.assertFalse(self.store._profile_path(person_fork).exists())

    def test_different_people_with_similar_voices_never_merge(self):
        """声音相似但真的是两个人：分数够到 candidate 带纯属巧合，攒得越多分数应该越低，不能被合并。"""
        rng = np.random.default_rng(11)
        D = 64
        voice_a = l2norm(rng.standard_normal(D))
        raw = rng.standard_normal(D)
        orth = l2norm(raw - np.dot(raw, voice_a) * voice_a)
        # 两个人的真实声音本身就有点像（余弦 0.40），但绝不是同一个人
        voice_b = l2norm(0.40 * voice_a + np.sqrt(1 - 0.40 ** 2) * orth)
        self.assertAlmostEqual(float(np.dot(voice_a, voice_b)), 0.40, places=6)

        r1 = self.store.identify(_noisy(rng, voice_a, 0.08), context="person A intro")
        person_a = r1.person_id
        for i in range(2):
            self.store.identify(_noisy(rng, voice_a, 0.08), context=f"A more{i}")

        cand_vec = None
        for _ in range(500):
            v = _noisy(rng, voice_b, 0.05)
            score = self.store._profiles[person_a].score(l2norm(v))
            if 0.40 <= score < 0.50:
                cand_vec = v
                break
        self.assertIsNotNone(cand_vec, "没能采样出一条落在 candidate 带的向量，调整噪声参数")

        r2 = self.store.identify(cand_vec, context="person B, coincidentally borderline vs A")
        person_b_fork = r2.person_id
        self.assertNotEqual(person_b_fork, person_a)

        # person B 自己后续的真实观测（真的是 B 的声音，不是 A 的）
        for i in range(4):
            v = _noisy(rng, voice_b, 0.05)
            self.store._profiles[person_b_fork].update(v, quality=1.0)
            self.store._meta[person_b_fork]["obs_count"] += 1
        self.store._save()

        # 越攒越像 B 自己、越不像 A —— 不应该被判定为可回收
        resolvable = self.store.find_resolvable_candidates(person_b_fork)
        self.assertEqual(resolvable, [], "声音相似的两个不同人被错误地判定为可合并")
        self.assertIn(person_a, self.store._profiles)
        self.assertIn(person_b_fork, self.store._profiles)

    def test_thin_stuck_anchor_merged_below_match_thr(self):
        """真实场景（85轮多人 benchmark 复测时发现）：Nancy 自我介绍那一句语气
        跟平时聊天不一样，声纹分数天生偏低，绑定的画像从此只有这 1 条样本；
        之后她所有真实发言全被 identify() 判给了分数更稳的分叉画像（分叉画像
        每次都赢，原画像永远没机会再被真实匹配更新）。两边互相打分只能到
        0.40~0.50 之间——不到 match_thr，但如果一直卡住不回收，原画像绑定的
        名字就再也关联不上她后续的任何一句话。这里验证：top 样本数<=2 且
        fork 已攒到>=3条真实样本时，即使 cross 达不到 match_thr，只要过了
        cand_thr 也应该判定为可回收。"""
        rng = np.random.default_rng(2)
        D = 64
        true_voice = l2norm(rng.standard_normal(D))

        # top：只有一条跟平时语气偏差较大的自我介绍样本
        intro_vec = None
        for _ in range(2000):
            v = _noisy(rng, true_voice, 0.6)
            if 0.44 <= float(np.dot(l2norm(v), true_voice)) < 0.50:
                intro_vec = v
                break
        self.assertIsNotNone(intro_vec, "没能采样出目标相似度的自我介绍向量，调整噪声参数")
        r1 = self.store.identify(intro_vec)
        person_top = r1.person_id
        self.assertEqual(self.store.get_meta(person_top)["obs_count"], 1)

        # 分叉：同一个人后续一句正常语气的话，纯声学分数没到 match_thr
        cand_vec = None
        for _ in range(500):
            v = _noisy(rng, true_voice, 0.05)
            score = self.store._profiles[person_top].score(l2norm(v))
            if 0.40 <= score < 0.50:
                cand_vec = v
                break
        self.assertIsNotNone(cand_vec)
        r2 = self.store.identify(cand_vec)
        person_fork = r2.person_id
        self.assertNotEqual(person_fork, person_top)

        # fork 之后持续攒到真实样本，top 从头到尾都没再被更新过
        for i in range(4):
            v = _noisy(rng, true_voice, 0.05)
            self.store._profiles[person_fork].update(v, quality=1.0)
            self.store._meta[person_fork]["obs_count"] += 1
        self.store._save()
        self.assertEqual(self.store.get_meta(person_top)["obs_count"], 1)
        self.assertGreaterEqual(self.store.get_meta(person_fork)["obs_count"], 3)

        resolvable = self.store.find_resolvable_candidates(person_fork)
        self.assertEqual(len(resolvable), 1)
        _, absorb_pid, into_pid, cross = resolvable[0]
        self.assertEqual(absorb_pid, person_fork)
        self.assertEqual(into_pid, person_top)
        self.assertGreaterEqual(cross, 0.40)
        self.assertLess(cross, 0.50, "这个用例本来就是为了验证卡在 cand_thr~match_thr 之间也能被放宽回收")

    def test_thin_stuck_anchor_different_person_stays_unresolved(self):
        """跟上一条相同的"top 只有1条样本、fork 已攒了>=3条"结构，但这次 fork
        真的是另一个人（声音基础相似度只有0.30，比 candidate 阈值高不了多少）
        ——即使套用放宽后的 cand_thr 判定，也绝不能被合并。"""
        rng = np.random.default_rng(5)
        D = 64
        voice_a = l2norm(rng.standard_normal(D))
        raw = rng.standard_normal(D)
        orth = l2norm(raw - np.dot(raw, voice_a) * voice_a)
        voice_b = l2norm(0.30 * voice_a + np.sqrt(1 - 0.30 ** 2) * orth)

        intro_vec = _noisy(rng, voice_a, 0.15)
        r1 = self.store.identify(intro_vec)
        person_top = r1.person_id
        self.assertEqual(self.store.get_meta(person_top)["obs_count"], 1)

        cand_vec = None
        for _ in range(2000):
            v = _noisy(rng, voice_b, 0.05)
            score = self.store._profiles[person_top].score(l2norm(v))
            if 0.40 <= score < 0.50:
                cand_vec = v
                break
        self.assertIsNotNone(cand_vec, "没能采样出目标相似度的候选向量，调整噪声参数")
        r2 = self.store.identify(cand_vec)
        person_fork = r2.person_id
        self.assertNotEqual(person_fork, person_top)

        for i in range(4):
            v = _noisy(rng, voice_b, 0.05)
            self.store._profiles[person_fork].update(v, quality=1.0)
            self.store._meta[person_fork]["obs_count"] += 1
        self.store._save()
        self.assertEqual(self.store.get_meta(person_top)["obs_count"], 1)
        self.assertGreaterEqual(self.store.get_meta(person_fork)["obs_count"], 3)

        resolvable = self.store.find_resolvable_candidates(person_fork)
        self.assertEqual(resolvable, [], "结构薄弱不该成为放宽合并的充分条件，真不同的人不能被合并")

    def test_merge_persons_is_noop_for_unknown_ids(self):
        rng = np.random.default_rng(3)
        r = self.store.identify(l2norm(rng.standard_normal(64)))
        before = dict(self.store._profiles)
        self.store.merge_persons("person_does_not_exist", r.person_id)
        self.assertEqual(set(self.store._profiles.keys()), set(before.keys()))


if __name__ == "__main__":
    unittest.main()
