# meeting_transcript_summarizer.py

import os
import re
import sys
import json
import uuid
import smtplib
import datetime
from typing import List, Optional, Dict, Any, Tuple, Union
from email.mime.text import MIMEText

import requests
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

# ----------------------------
# Configuration
# ----------------------------
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# ----------------------------
# Utility
# ----------------------------
def approx_token_count(text: str) -> int:
    """Approximate token count fallback using word count."""
    return max(1, len(text.split()))

# ----------------------------
# Transcript Loading
# ----------------------------
TRANSCRIPT_LINE_RE = re.compile(
    r"^\s*(?:\[(?P<ts>\d{1,2}:\d{2}:\d{2})\]\s*)?(?P<speaker>[^:]+):\s*(?P<text>.+)$"
)

def parse_hms(ts: str) -> float:
    parts = [int(x) for x in ts.split(":")]
    if len(parts) == 3:
        h, m, s = parts
        return float(h * 3600 + m * 60 + s)
    return 0.0

class Turn:
    """Data structure for a single speaker turn."""
    def __init__(self, speaker: str, text: str, start: Optional[float] = None,
                 end: Optional[float] = None, turn_id: Optional[str] = None):
        self.speaker = speaker
        self.text = text
        self.start = start
        self.end = end
        self.turn_id = turn_id or str(uuid.uuid4())

def load_transcript(path: str) -> List[Turn]:
    """Load transcript from .txt, .json, or .jsonl formats."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    turns: List[Turn] = []
    suffix = os.path.splitext(path)[1].lower()

    if suffix in {".json", ".jsonl"}:
        with open(path, "r", encoding="utf-8") as f:
            if suffix == ".jsonl":
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    turns.append(Turn(
                        speaker=obj.get("speaker", "Unknown"),
                        text=obj.get("text", ""),
                        start=obj.get("start"),
                        end=obj.get("end"),
                        turn_id=str(obj.get("id", str(uuid.uuid4()))),
                    ))
            else:
                data = json.load(f)
                for obj in data:
                    turns.append(Turn(
                        speaker=obj.get("speaker", "Unknown"),
                        text=obj.get("text", ""),
                        start=obj.get("start"),
                        end=obj.get("end"),
                        turn_id=str(obj.get("id", str(uuid.uuid4()))),
                    ))
        return turns

    # Plain text parsing (.txt)
    with open(path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue
            m = TRANSCRIPT_LINE_RE.match(line)
            if m:
                ts = m.group("ts")
                speaker = m.group("speaker").strip()
                text = m.group("text").strip()
                start = parse_hms(ts) if ts else None
                turns.append(Turn(speaker=speaker, text=text, start=start, turn_id=f"t{i}"))
            else:
                if turns:
                    turns[-1].text += " " + line
                else:
                    turns.append(Turn(speaker="Unknown", text=line, turn_id=f"t{i}"))

    for idx in range(len(turns) - 1):
        if turns[idx].start is not None and turns[idx + 1].start is not None:
            turns[idx].end = turns[idx + 1].start
    return turns

def load_transcript_from_text(transcript_text: str) -> List[Turn]:
    """Parse transcript provided directly as raw text."""
    turns: List[Turn] = []
    for i, raw in enumerate(transcript_text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        m = TRANSCRIPT_LINE_RE.match(line)
        if m:
            ts = m.group("ts")
            speaker = m.group("speaker").strip()
            text = m.group("text").strip()
            start = parse_hms(ts) if ts else None
            turns.append(Turn(speaker=speaker, text=text, start=start, turn_id=f"t{i}"))
        else:
            if turns:
                turns[-1].text += " " + line
            else:
                turns.append(Turn(speaker="Unknown", text=line, turn_id=f"t{i}"))
    return turns

# ----------------------------
# Transcript Chunking
# ----------------------------
def sentence_split(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]

def chunk_transcript(turns: List[Turn], max_tokens: int = 700, overlap_tokens: int = 120) -> List[Dict[str, Any]]:
    """Chunk transcript turns into coherent pieces with speakers and timestamps."""
    chunks, cur_text, cur_speakers, cur_turn_ids = [], [], [], []
    cur_start, cur_end, cur_token_est = None, None, 0

    def flush():
        nonlocal cur_text, cur_speakers, cur_turn_ids, cur_start, cur_end, cur_token_est
        if not cur_text:
            return
        chunks.append({
            "id": str(uuid.uuid4()),
            "text": "\n".join(cur_text).strip(),
            "speakers": list(dict.fromkeys(cur_speakers)),
            "start": cur_start,
            "end": cur_end,
            "turn_ids": list(cur_turn_ids),
        })
        cur_text, cur_speakers, cur_turn_ids = [], [], []
        cur_start, cur_end, cur_token_est = None, None, 0

    for t in turns:
        sents = sentence_split(t.text) or [t.text]
        for sent in sents:
            addition = f"{t.speaker}: {sent}"
            add_tokens = approx_token_count(addition)
            if cur_token_est + add_tokens > max_tokens:
                if not cur_text:
                    cur_text.append(addition)
                    cur_speakers.append(t.speaker)
                    cur_turn_ids.append(t.turn_id)
                    if cur_start is None:
                        cur_start = t.start
                    cur_end = t.end
                    flush()
                else:
                    flush()
                    cur_text.append(addition)
                    cur_speakers.append(t.speaker)
                    cur_turn_ids.append(t.turn_id)
                    if cur_start is None:
                        cur_start = t.start
                    cur_end = t.end
                    cur_token_est = approx_token_count("\n".join(cur_text))
                continue
            else:
                cur_text.append(addition)
                cur_speakers.append(t.speaker)
                cur_turn_ids.append(t.turn_id)
                if cur_start is None:
                    cur_start = t.start
                cur_end = t.end
                cur_token_est += add_tokens
    flush()
    return chunks

# ----------------------------
# Embeddings & FAISS
# ----------------------------
def embed_text(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    return emb[0]

def create_faiss_index(docs: List[Dict[str, Any]]) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
    dimension = 384
    index = faiss.IndexFlatL2(dimension)
    embeddings = [embed_text(d["text"]) for d in docs]
    embeddings = np.vstack(embeddings).astype(np.float32)
    index.add(embeddings)
    return index, embeddings

def search_faiss_index(index, query_vec, embeddings, docs, k=5):
    q = query_vec.reshape(1, -1).astype(np.float32) if query_vec.ndim == 1 else query_vec.astype(np.float32)
    D, I = index.search(q, k)
    return [docs[i] for i in I[0] if 0 <= i < len(docs)]

# ----------------------------
# Topic Detection
# ----------------------------
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except Exception:
    KMeans, silhouette_score = None, None

def detect_topics(docs: List[Dict[str, Any]], embeddings: np.ndarray, k_min: int = 2, k_max: int = 6):
    n = len(docs)
    if n == 0 or KMeans is None or n < 2:
        return [{"label": "General Discussion", "indices": list(range(n))}]
    best_labels, best_score = None, -2
    max_k = min(k_max, n)
    for k in range(max(2, k_min), max_k + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
            labels = km.labels_
            if len(set(labels)) == 1:
                continue
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score, best_labels = score, labels
        except Exception:
            continue
    if best_labels is None:
        return [{"label": "General Discussion", "indices": list(range(n))}]
    topics = []
    for k in sorted(set(best_labels)):
        idxs = [i for i, lab in enumerate(best_labels) if lab == k]
        samples = [docs[i]["text"][:600] for i in idxs[:4]]
        label = label_topic_with_llm(samples)
        topics.append({"label": label, "indices": idxs})
    return topics

def label_topic_with_llm(samples: List[str]) -> str:
    system = "You are a utility that names meeting discussion topics. Return a short (2-4 words) topic label."
    user = "\n\n".join(f"Snippet {i+1}:\n{txt}" for i, txt in enumerate(samples))
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=30,
        )
        name = resp.choices[0].message.content.strip()
        name = re.sub(r"[^A-Za-z0-9\s\-&]", "", name).strip()
        return name if name else "General Discussion"
    except Exception:
        return "General Discussion"

# ----------------------------
# Summarization
# ----------------------------
SUMMARY_SYSTEM = "You are an expert meeting summarizer. Produce concise, faithful summaries."

SUMMARY_USER_TEMPLATE = """
Using the provided meeting excerpts, produce a summary:

High-Level Meeting Summary

Topics Discussed

<topic name>:
- <point>
"""

def generate_meeting_summary(topics: List[Dict[str, Any]], docs: List[Dict[str, Any]], custom_prompt: Optional[str] = None) -> str:
    topic_blocks = []
    for t in topics:
        label, indices = t.get("label", "Topic"), t.get("indices", [])
        reps = []
        for idx in indices[:4]:
            d = docs[idx]
            header = f"Speakers: {', '.join(d.get('speakers', []))}"
            reps.append(header + "\n" + d["text"])
        topic_blocks.append(f"Topic: {label}\n\n" + "\n\n".join(reps))
    big_context = "\n\n".join(topic_blocks)

    if not custom_prompt or not custom_prompt.strip():
        user_prompt = SUMMARY_USER_TEMPLATE.format(context=big_context)
        system_prompt = SUMMARY_SYSTEM
    else:
        system_prompt = "You are an expert assistant for summarizing meeting transcripts."
        user_prompt = f"Here are excerpts:\n\n{big_context}\n\nFollow these instructions:\n\n{custom_prompt.strip()}"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.1,
            max_tokens=1500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating meeting summary: {str(e)}"

# ----------------------------
# Orchestration
# ----------------------------
def process_transcript_file(transcript_path: Optional[str] = None, transcript_text: Optional[str] = None,
                            email_recipient: Optional[Union[List[str], str]] = None, custom_prompt: Optional[str] = None):
    if transcript_text:
        turns = load_transcript_from_text(transcript_text)
    elif transcript_path:
        turns = load_transcript(transcript_path)
    else:
        raise ValueError("No transcript input provided.")

    if not turns:
        raise ValueError("No turns found in transcript.")

    chunks = chunk_transcript(turns, max_tokens=700, overlap_tokens=120)
    index, embeddings = create_faiss_index(chunks)

    try:
        topics = detect_topics(chunks, embeddings)
    except Exception:
        topics = [{"label": "General Discussion", "indices": list(range(len(chunks)))}]

    final_summary = generate_meeting_summary(topics, chunks, custom_prompt=custom_prompt)

    if email_recipient:
        recipients = email_recipient if isinstance(email_recipient, list) else [email_recipient]
        for r in recipients:
            try:
                send_email_notification(r, final_summary, None)
            except Exception:
                pass

    if transcript_path:
        try:
            os.remove(transcript_path)
        except Exception:
            pass

    return final_summary

# ----------------------------
# Email
# ----------------------------
def send_email_notification(recipient_email: Union[str, List[str]], summary: Optional[str], error_message: Optional[str]):
    from_email = "noreply@example.com"
    subject = "Transcript Processing Outcome"
    body = f"Processing failed:\n\n{error_message}" if error_message else f"Transcript processed successfully:\n\n{summary}"
    recipients = recipient_email if isinstance(recipient_email, list) else [recipient_email]

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = ", ".join(recipients)

    try:
        with smtplib.SMTP("localhost") as server:
            server.sendmail(from_email, recipients, msg.as_string())
        print(f"Email sent to: {', '.join(recipients)}")
    except Exception as e:
        print(f"Failed to send email(s): {str(e)}")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Meeting Transcript Summarizer")
    parser.add_argument("transcript", help="Path to transcript file (or '-' for stdin)")
    parser.add_argument("--email", help="Notification email (single)")
    args = parser.parse_args()

    if args.transcript == "-":
        txt = sys.stdin.read()
        summary = process_transcript_file(transcript_text=txt, email_recipient=args.email)
    else:
        summary = process_transcript_file(transcript_path=args.transcript, email_recipient=args.email)

    print("\n\n==== Final Meeting Summary ====\n")
    print(summary)
