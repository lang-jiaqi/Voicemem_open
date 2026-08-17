"""文本路由:句子 -> 实体 + 簇(分类)。来自既有实现,原样保留。"""
import jieba
import jieba.posseg as pseg
import numpy as np
from model2vec import StaticModel


class Encoder:
    """文本 -> 归一化向量。中英文同空间。~30MB 静态模型, CPU 亚毫秒。"""

    def __init__(self, model_name="minishlab/M2V_multilingual_output"):
        self.model = StaticModel.from_pretrained(model_name)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        v = np.asarray(self.model.encode(texts), dtype=np.float32)
        if v.ndim == 1:
            v = v[None, :]
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1e-9
        return v / n


class EntityExtractor:
    """jieba 抽实体: 中文名词/双字动词 + 英文词。零模型。"""

    ENTITY_POS = {"n", "nr", "ns", "nt", "nz", "nl", "ng", "vn", "v"}
    STOPWORDS = {"个", "些", "种", "样", "点", "次", "时", "事", "了下", "有点"}
    VERB_STOP = {"是", "有", "要", "想", "喜欢", "讨厌", "吃", "喝", "去", "来",
                 "做", "看", "说", "用", "会", "能", "让", "觉得", "感觉",
                 "知道", "希望", "需要", "可以", "应该"}
    # 英文在 jieba 中通常全都标成 eng；不能把每个英文 token 都当实体。
    # 这里优先排除口语连接词、对话控制语和语言本身的名称。
    EN_STOP = {"the", "a", "an", "i", "you", "he", "she", "it", "we", "they",
               "is", "am", "are", "was", "were", "to", "of", "in", "on", "at",
               "and", "or", "but", "my", "your", "this", "that", "very", "like",
               "want", "have", "had", "do", "did", "will", "would", "can", "me",
               "more", "too", "some", "for", "with", "every", "get", "got",
               "let", "next", "after", "so", "when", "what", "where", "why", "how",
               "about", "right", "okay", "ok", "yeah", "yes", "no", "well", "just",
               "really", "actually", "basically", "maybe", "please", "thanks", "thank",
               "hello", "hi", "start", "starting", "talk", "talking", "speak", "speaking",
               "say", "saying", "chat", "conversation", "language", "english", "chinese",
               "mandarin", "cantonese", "word", "words", "sentence", "sentences"}

    def __init__(self, force_nouns=None, verb_min_len=2):
        if force_nouns:
            for w in force_nouns:
                jieba.add_word(w, tag="n")
        self.verb_min_len = verb_min_len

    def extract(self, text):
        out, seen = [], set()
        for w in pseg.cut(text):
            word, flag = w.word, w.flag
            if flag == "eng":
                lw = word.lower()
                if len(lw) > 1 and lw not in self.EN_STOP and word not in seen:
                    seen.add(word); out.append(word)
                continue
            if flag not in self.ENTITY_POS or word in self.STOPWORDS:
                continue
            if flag == "v" and (word in self.VERB_STOP or len(word) < self.verb_min_len):
                continue
            if word not in seen:
                seen.add(word); out.append(word)
        return out

    def is_discourse_only(self, text):
        """英语口语组织语不进入记忆路由，也不制造无意义的检索槽位。"""
        english_words = [w.word.lower() for w in pseg.cut(text) if w.flag == "eng" and len(w.word) > 1]
        return bool(english_words) and all(word in self.EN_STOP for word in english_words)


class MemoryRouter:
    """句子 -> 实体 + 该去哪几个簇(分类)做 RAG。"""

    def __init__(self, encoder, entity_extractor):
        self.enc = encoder
        self.ee = entity_extractor
        self.names = []
        self.cat_vecs = None

    def set_categories(self, categories):
        """categories: {分类名: 描述文本}"""
        self.names = list(categories.keys())
        descs = [categories[n] for n in self.names]
        self.cat_vecs = self.enc.encode(descs) if descs else None

    def route(self, text, top_k=3, threshold=0.0):
        """返回 (实体列表, [(分类名, 分数), ...] 降序)。"""
        entities = self.ee.extract(text)
        if self.ee.is_discourse_only(text):
            return [], []
        if self.cat_vecs is None or not text.strip():
            return entities, []
        qv = self.enc.encode(text)[0]
        sims = self.cat_vecs @ qv
        order = np.argsort(-sims)
        cats = [(self.names[i], float(sims[i])) for i in order
                if sims[i] >= threshold][:top_k]
        return entities, cats
