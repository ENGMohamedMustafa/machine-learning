# app.py
import os
import tempfile
import torch
import faiss
import numpy as np
import streamlit as st
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
import fitz  # PyMuPDF
import soundfile as sf

# ---------------------------
# RAGCore class encapsulates all logic
# ---------------------------
class RAGCore:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        reranker_model_name: str = "BAAI/bge-reranker-base",
        llm_model_name: str = "TheBloke/Mistral-7B-Instruct-v0.1",
        translation_model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = None,
        tts_voices_dir: str = "models/voices",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Embedding model for FAISS indexing and query embedding
        self.embedder = SentenceTransformer(embedding_model_name, device=self.device)

        # Reranker model (bi-encoder)
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
        self.reranker_model = AutoModel.from_pretrained(reranker_model_name).to(self.device)
        self.reranker_model.eval()

        # LLM model and tokenizer (seq2seq)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name).to(self.device)
        self.llm_model.eval()

        # Translation model and tokenizer (NLLB-200)
        self.translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name).to(self.device)
        self.translation_model.eval()

        # FAISS index and corpus storage
        self.faiss_index = None
        self.corpus_texts = []
        self.corpus_embeddings = None

        # TTS voices directory
        self.tts_voices_dir = tts_voices_dir

        # Prompt template
        self.prompt_template = (
            "أنت مساعد ذكي متخصص في الإجابة فقط من محتوى الكتاب المرفق. "
            "إذا لم توجد المعلومة في الكتاب، أجب: 'المعلومة غير متوفرة في الكتاب'.\n"
            "يمكنك تقديم ملخصات، شروحات، أمثلة، ومقارنات من الكتاب فقط.\n\n"
            "السؤال: {query}\n"
            "المحتوى المسترجع:\n{context}\n"
            "الإجابة:"
        )

    def add_texts_to_index(self, texts: List[str]):
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype("float32")
        if self.faiss_index is None:
            dim = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.corpus_texts = []
            self.corpus_embeddings = np.empty((0, dim), dtype="float32")
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings)
        self.corpus_texts.extend(texts)
        self.corpus_embeddings = np.vstack([self.corpus_embeddings, embeddings])

    def retrieve(self, query: str, top_k: int = 10, rerank_top_k: int = 5) -> List[Tuple[str, float]]:
        if self.faiss_index is None or len(self.corpus_texts) == 0:
            return []
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        query_emb = query_emb.astype("float32")
        faiss.normalize_L2(query_emb)
        D, I = self.faiss_index.search(query_emb, top_k)
        retrieved_texts = [self.corpus_texts[i] for i in I[0]]
        rerank_texts = retrieved_texts[:rerank_top_k]
        rerank_scores = self._rerank(query, rerank_texts)
        reranked_results = list(zip(rerank_texts, rerank_scores))
        for i, text in enumerate(retrieved_texts[rerank_top_k:]):
            reranked_results.append((text, float(D[0][rerank_top_k + i])))
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return reranked_results[:top_k]

    def _rerank(self, query: str, texts: List[str]) -> List[float]:
        scores = []
        self.reranker_model.eval()
        with torch.no_grad():
            for doc in texts:
                inputs = self.reranker_tokenizer(query, doc, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                outputs = self.reranker_model(**inputs)
                pooled = outputs.last_hidden_state[:, 0, :]
                score = pooled.norm().item()
                scores.append(score)
        return scores

    def generate_answer(self, query: str, context: str, max_length: int = 512) -> str:
        prompt = self.prompt_template.format(query=query, context=context)
        inputs = self.llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                eos_token_id=self.llm_tokenizer.eos_token_id,
            )
        answer = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        if prompt in answer:
            answer = answer.replace(prompt, "").strip()
        return answer

    def self_rag(self, query: str, prev_answer: str, max_retries: int = 2) -> str:
        for _ in range(max_retries):
            if "المعلومة غير متوفرة في الكتاب" not in prev_answer and len(prev_answer.strip()) > 20:
                return prev_answer
            retrieved = self.retrieve(query, top_k=15, rerank_top_k=10)
            context = "\n\n".join([t for t, _ in retrieved])
            new_answer = self.generate_answer(query, context)
            if new_answer != prev_answer:
                prev_answer = new_answer
            else:
                break
        return prev_answer

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        inputs = self.translation_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        forced_bos_token_id = self.translation_tokenizer.lang_code_to_id[target_lang]
        with torch.no_grad():
            generated_tokens = self.translation_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=1024,
                num_beams=5,
                early_stopping=True,
            )
        translated = self.translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated

    def text_to_speech(self, text: str, lang: str = "ar") -> str:
        voice_dir = os.path.join(self.tts_voices_dir, "ar" if lang == "ar" else "en")
        if not os.path.exists(voice_dir):
            raise FileNotFoundError(f"TTS voice directory not found: {voice_dir}")
        try:
            from TTS.api import TTS
        except ImportError:
            raise ImportError("Please install Coqui TTS: pip install TTS")
        voice_models = [f for f in os.listdir(voice_dir) if f.endswith(".pth") or f.endswith(".onnx")]
        if not voice_models:
            raise FileNotFoundError(f"No TTS model found in {voice_dir}")
        model_path = os.path.join(voice_dir, voice_models[0])
        tts = TTS(model_path, progress_bar=False, gpu=torch.cuda.is_available())
        wav = tts.tts(text)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp_file.name, wav, samplerate=tts.synthesizer.output_sample_rate)
        tmp_file.close()
        return tmp_file.name

    @staticmethod
    def extract_text_from_pdf(pdf_file_path: str) -> List[str]:
        doc = fitz.open(pdf_file_path)
        texts = []
        for page in doc:
            text = page.get_text("text")
            paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
            texts.extend(paragraphs)
        return texts

# ---------------------------
# Streamlit app
# ---------------------------

@st.cache_resource(show_spinner=False)
def load_rag_core():
    return RAGCore(
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
        reranker_model_name="BAAI/bge-reranker-base",
        llm_model_name="TheBloke/Mistral-7B-Instruct-v0.1",
        translation_model_name="facebook/nllb-200-distilled-600M",
        device="cuda" if torch.cuda.is_available() else "cpu",
        tts_voices_dir="models/voices",
    )

rag_core = load_rag_core()

st.set_page_config(page_title="RAG Arabic-English Book QA", layout="wide")
st.title("📚 RAG Chat: سؤال وجواب من الكتب (عربي / إنجليزي)")

with st.sidebar:
    st.header("تعليمات")
    st.markdown(
        """
        - ارفع ملفات PDF للكتب (يمكن رفع أكثر من ملف).
        - انتظر حتى يتم معالجة الملفات وفهرستها.
        - اكتب سؤالك في صندوق الأسئلة.
        - اختر لغة الإجابة: عربي أو إنجليزي.
        - اضغط زر 'اسأل' للحصول على الإجابة.
        - يمكنك تشغيل الصوت للاستماع للإجابة.
        """
    )
    st.markdown("---")
    st.markdown("### تحميل النماذج والأصوات")
    st.markdown(
        """
        - **نماذج LLM**: يمكنك تحميل Mistral 7B أو Llama 3 من [HuggingFace](https://huggingface.co/models).
        - **نماذج الترجمة NLLB-200**: [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M).
        - **نماذج الأصوات TTS**:
          - Coqui TTS: [https://github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS)
          - ضع ملفات النماذج الصوتية في مجلد `models/voices/ar` للعربية و`models/voices/en` للإنجليزية.
        """
    )

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader(
    "📄 ارفع ملفات PDF للكتب (يمكن رفع أكثر من ملف)", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in [uf.name for uf in st.session_state.uploaded_files]]
    if new_files:
        st.info(f"معالجة {len(new_files)} ملف جديد...")
        for file in new_files:
            with open(f"temp_{file.name}", "wb") as f:
                f.write(file.getbuffer())
            texts = rag_core.extract_text_from_pdf(f"temp_{file.name}")
            rag_core.add_texts_to_index(texts)
            st.session_state.uploaded_files.append(file)
            os.remove(f"temp_{file.name}")
        st.success("تم فهرسة الملفات بنجاح!")

lang = st.radio("اختر لغة الإجابة:", options=["عربي", "إنجليزي"], horizontal=True)
query = st.text_input("اكتب سؤالك هنا:")

if st.button("اسأل") and query.strip() != "":
    if rag_core.faiss_index is None or len(rag_core.corpus_texts) == 0:
        st.warning("يرجى رفع ملفات PDF أولاً لفهرسة المحتوى.")
    else:
        with st.spinner("جاري البحث والإجابة..."):
            retrieved = rag_core.retrieve(query, top_k=10, rerank_top_k=5)
            context = "\n\n".join([t for t, _ in retrieved])
            answer = rag_core.generate_answer(query, context)
            answer = rag_core.self_rag(query, answer)
            if lang == "عربي":
                final_answer = answer
            else:
                final_answer = rag_core.translate(answer, source_lang="arb_Arab", target_lang="eng_Latn")
            st.session_state.chat_history.append({"query": query, "answer": final_answer, "lang": lang})

if st.session_state.chat_history:
    st.markdown("---")
    st.header("المحادثة")
    for i, chat in enumerate(st.session_state.chat_history):
        st.markdown(f"**سؤال:** {chat['query']}")
        st.markdown(f"**إجابة ({chat['lang']}):** {chat['answer']}")
        if st.button(f"🔊 تشغيل الصوت للإجابة #{i+1}"):
            try:
                audio_path = rag_core.text_to_speech(chat["answer"], lang="ar" if chat["lang"] == "عربي" else "en")
                audio_file = open(audio_path, "rb").read()
                st.audio(audio_file, format="audio/wav")
                os.remove(audio_path)
            except Exception as e:
                st.error(f"حدث خطأ في تحويل النص إلى كلام: {e}")