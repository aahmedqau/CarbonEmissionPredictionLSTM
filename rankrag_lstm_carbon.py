
"""
RankRAG + LSTM (Carbon Emissions) — Reference Implementation
============================================================

This single-file script gives you a minimal, dependency-light prototype of the
diagram you shared:

- RankRAG pipeline
  1) Instruction Tuning (placeholder hook)
  2) Context-Rich Fine Tuning (placeholder hook)
  3) Context Retrieval (TF–IDF cosine similarity)
  4) Reranking (a tiny learned linear model, trained with pairwise preference)
  5) Classify Top‑k Carbon Emissions (logistic regression over retrieved context)

- LSTM forecaster
  A PyTorch LSTM that predicts the next-step carbon emissions time series value,
  optionally conditioned on an embedding of the retrieved context.

This is a compact educational implementation. You can replace the placeholder
hooks with your own LLM / instruction-tuning code and swap the simple
reranker/classifier with stronger models (e.g., cross-encoder, gradient-boosted trees).

No internet needed. Works with:
  - numpy
  - scikit-learn
  - torch

Usage (demo):
  python rankrag_lstm_carbon.py

Author: ChatGPT (GPT-5 Thinking)
License: MIT
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import random
import math
import numpy as np

# Sklearn for TF-IDF, logistic regression, and utilities
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Torch for LSTM forecaster
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------
# Synthetic Data Generators
# ---------------------------

EMISSION_LABELS = ["power", "industry", "transport", "residential", "agriculture"]

def seed_everything(seed: int = 1337):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

seed_everything(42)


def make_synthetic_context_corpus(n_docs: int = 200) -> Tuple[List[str], List[str]]:
  """Generate a small synthetic corpus of carbon-emission related contexts.

  Returns:
      docs: list of text documents
      labels: sector label for each document (one of EMISSION_LABELS)
  """
  sector_keywords = {
      "power": ["coal", "gas", "renewable", "turbine", "grid", "baseload", "dispatch", "capacity", "MW", "generator"],
      "industry": ["cement", "steel", "smelter", "kiln", "furnace", "process", "feedstock", "ammonia", "ethylene", "refinery"],
      "transport": ["diesel", "fleet", "traffic", "aviation", "shipping", "logistics", "BEV", "hybrid", "rail", "port"],
      "residential": ["heating", "cooking", "appliance", "gas stove", "insulation", "retrofit", "boiler", "lighting", "HVAC", "meter"],
      "agriculture": ["fertilizer", "livestock", "methane", "soil", "tractor", "irrigation", "harvest", "manure", "pasture", "enteric"],
  }

  docs, labels = [], []
  for _ in range(n_docs):
      label = random.choice(EMISSION_LABELS)
      kws = random.sample(sector_keywords[label], k=5)
      fluff = random.sample(sum(sector_keywords.values(), []), k=4)
      txt = f"{label} sector report: " + ", ".join(kws + fluff) + ". Emissions audit and mitigation plan."
      docs.append(txt)
      labels.append(label)
  return docs, labels


def make_synthetic_emission_series(T: int = 400) -> np.ndarray:
  """Create a synthetic monthly emissions series with trend + seasonality + noise."""
  t = np.arange(T)
  trend = 0.05 * t  # slow upward trend
  seasonal = 5 * np.sin(2 * np.pi * t / 12)  # annual seasonality
  noise = np.random.normal(0, 1.5, size=T)
  baseline = 100
  series = baseline + trend + seasonal + noise
  return series.astype(np.float32)


# ---------------------------------
# RankRAG Components (Minimal)
# ---------------------------------

@dataclass
class RankRAGConfig:
  top_k_retrieve: int = 10
  top_k_classify: int = 5
  vocab_max_features: int = 5000


class SimpleReranker(nn.Module):
  """A very small reranker: linear layer over TF-IDF similarities & length.
  Input features per (query, doc):
      [cosine_sim, doc_len]
  """
  def __init__(self):
      super().__init__()
      self.layer = nn.Linear(2, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.layer(x).squeeze(-1)


class RankRAG:
  def __init__(self, cfg: RankRAGConfig):
      self.cfg = cfg
      self.vectorizer = TfidfVectorizer(max_features=cfg.vocab_max_features)
      self.docs: List[str] = []
      self.doc_labels: List[str] = []
      self.doc_matrix = None  # TF-IDF matrix
      # Reranker
      self.reranker = SimpleReranker()
      self.reranker_opt = torch.optim.Adam(self.reranker.parameters(), lr=3e-3)
      # Simple classifier over retrieved context
      self.classifier: Optional[Pipeline] = None

  # 1) Instruction Tuning (hook)
  def instruction_tune(self, instructions: str):
      # Placeholder for LLM instruction-tuning.
      # In this prototype we simply store it (could be used for prompts).
      self.instructions = instructions

  # 2) Context-Rich Fine Tuning (hook)
  def fine_tune_context(self, extra_corpus: List[str]):
      # Placeholder. In a real system, you'd update embeddings or doc representations.
      # Here, we append extra data and re-fit TF-IDF.
      self.docs += extra_corpus
      self.doc_labels += ["unknown"] * len(extra_corpus)
      self.doc_matrix = self.vectorizer.fit_transform(self.docs)

  def fit_corpus(self, docs: List[str], labels: List[str]):
      self.docs = list(docs)
      self.doc_labels = list(labels)
      self.doc_matrix = self.vectorizer.fit_transform(self.docs)

  # 3) Retrieval
  def retrieve(self, query: str, top_k: Optional[int] = None) -> List[int]:
      if self.doc_matrix is None:
          raise RuntimeError("Call fit_corpus first.")
      q_vec = self.vectorizer.transform([query])
      sims = cosine_similarity(q_vec, self.doc_matrix).ravel()
      k = top_k or self.cfg.top_k_retrieve
      idx = np.argsort(-sims)[:k]
      return idx.tolist()

  # 4) Reranking (learn tiny model from preferences)
  def train_reranker(self, train_pairs: List[Tuple[str, int, int]], epochs: int = 5):
      """Train on triplets: (query, pos_idx, neg_idx) where pos should rank above neg."""
      for _ in range(epochs):
          random.shuffle(train_pairs)
          total_loss = 0.0
          for q, pos, neg in train_pairs:
              q_vec = self.vectorizer.transform([q])
              sims = cosine_similarity(q_vec, self.doc_matrix).ravel()

              feats_pos = np.array([sims[pos], len(self.docs[pos])], dtype=np.float32)
              feats_neg = np.array([sims[neg], len(self.docs[neg])], dtype=np.float32)

              X = torch.from_numpy(np.stack([feats_pos, feats_neg]))  # shape (2, 2)
              scores = self.reranker(X)  # shape (2,)
              # Hinge loss: want score_pos > score_neg by margin
              loss = torch.relu(1.0 - (scores[0] - scores[1]))
              self.reranker_opt.zero_grad()
              loss.backward()
              self.reranker_opt.step()
              total_loss += float(loss.item())
      return total_loss

  def rerank(self, query: str, candidates: List[int], top_k: Optional[int] = None) -> List[int]:
      q_vec = self.vectorizer.transform([query])
      sims = cosine_similarity(q_vec, self.doc_matrix[candidates]).ravel()
      # feature: [cosine_sim, doc_len]
      feats = np.stack([sims, [len(self.docs[i]) for i in candidates]], axis=1).astype(np.float32)
      with torch.no_grad():
          scores = self.reranker(torch.from_numpy(feats)).numpy()
      order = np.argsort(-scores)
      k = top_k or len(candidates)
      return [candidates[i] for i in order[:k]]

  # 5) Classify Top‑k Emission Sectors
  def fit_classifier(self):
      y = np.array(self.doc_labels)
      # Use TF-IDF features directly
      clf = Pipeline([
          ("scaler", StandardScaler(with_mean=False)),  # sparse-compatible
          ("lr", LogisticRegression(max_iter=1000, multi_class='auto')),
      ])
      clf.fit(self.doc_matrix, y)
      self.classifier = clf

  def classify_topk(self, topk_idx: List[int]) -> List[Tuple[str, float]]:
      if self.classifier is None:
          raise RuntimeError("Call fit_classifier() first.")
      # Aggregate TF-IDF vectors of top-k
      sub = self.doc_matrix[topk_idx]
      agg = np.asarray(sub.mean(axis=0))
      # Predict probabilities
      proba = self.classifier.named_steps['lr'].predict_proba(agg)
      classes = self.classifier.named_steps['lr'].classes_
      scores = list(zip(classes, proba.ravel().tolist()))
      scores.sort(key=lambda x: -x[1])
      return scores[:5]

  def embed_context(self, idx_list: List[int]) -> np.ndarray:
      """Return a dense embedding for the retrieved context (mean TF-IDF)."""
      sub = self.doc_matrix[idx_list]
      vec = np.asarray(sub.mean(axis=0)).ravel()
      return vec


# ---------------------------------
# LSTM Forecaster
# ---------------------------------

class EmissionDataset(Dataset):
  def __init__(self, series: np.ndarray, context_vec: np.ndarray, lookback: int = 12):
      self.series = series
      self.context_vec = context_vec.astype(np.float32)
      self.lookback = lookback
      self.n = len(series) - lookback

  def __len__(self):
      return self.n

  def __getitem__(self, idx):
      x_seq = self.series[idx:idx+self.lookback]  # (L,)
      y = self.series[idx+self.lookback]          # scalar next-step
      x = np.concatenate([x_seq, self.context_vec])  # append context as static features
      return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)


class LSTMRegressor(nn.Module):
  def __init__(self, lookback: int, context_dim: int, hidden: int = 64, num_layers: int = 1):
      super().__init__()
      self.lookback = lookback
      self.context_dim = context_dim

      # Sequence branch for time-series part
      self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=num_layers, batch_first=True)
      # MLP head that fuses LSTM output with static context
      self.fc = nn.Sequential(
          nn.Linear(hidden + context_dim, 64),
          nn.ReLU(),
          nn.Linear(64, 1)
      )

  def forward(self, x):
      # x shape: (B, lookback + context_dim)
      seq = x[:, :self.lookback].unsqueeze(-1)  # (B, L, 1)
      ctx = x[:, self.lookback:]                # (B, context_dim)
      out, _ = self.lstm(seq)
      h = out[:, -1, :]                         # (B, hidden)
      fused = torch.cat([h, ctx], dim=1)
      yhat = self.fc(fused)
      return yhat


# ---------------------------
# Demo / Wiring it together
# ---------------------------

def demo():
  print("\n==== RankRAG + LSTM Demo ====")

  # 1) Build corpus
  docs, labels = make_synthetic_context_corpus(n_docs=300)
  rag = RankRAG(RankRAGConfig(top_k_retrieve=12, top_k_classify=5, vocab_max_features=4000))
  rag.fit_corpus(docs, labels)

  # 2) Optional: instruction tuning & fine-tuning hooks
  rag.instruction_tune("Classify emission contexts accurately and retrieve sector-specific evidence.")
  rag.fine_tune_context(["energy storage batteries reduce peaker plant emissions via demand shifting."])

  # 3) Train a tiny reranker with synthetic preferences
  #    We'll pretend queries prefer documents from the correct sector
  train_pairs = []
  queries = [
      ("coal plant ramping and grid dispatch", "power"),
      ("clinker kiln heat rate in cement", "industry"),
      ("diesel trucks vs BEV fleet", "transport"),
      ("home heating retrofit with insulation", "residential"),
      ("livestock methane reduction", "agriculture"),
  ]
  for q, target in queries:
      # pos indices: docs whose label == target; neg: different label
      pos_idx = [i for i, lab in enumerate(rag.doc_labels) if lab == target]
      neg_idx = [i for i, lab in enumerate(rag.doc_labels) if lab != target]
      # sample a few pairs
      for _ in range(8):
          if not pos_idx or not neg_idx: continue
          train_pairs.append((q, random.choice(pos_idx), random.choice(neg_idx)))
  _ = rag.train_reranker(train_pairs, epochs=3)

  # 4) Fit classifier on entire corpus
  rag.fit_classifier()

  # 5) Run retrieval + rerank for a working query
  query = "forecast power sector emissions using grid and coal generator context"
  initial = rag.retrieve(query, top_k=rag.cfg.top_k_retrieve)
  reranked = rag.rerank(query, initial, top_k=rag.cfg.top_k_retrieve)

  # 6) Classify top‑k emissions
  topk_scores = rag.classify_topk(reranked[:rag.cfg.top_k_classify])
  print("Top‑k emission sector probabilities:")
  for lab, sc in topk_scores:
      print(f"  {lab:12s}  {sc:.3f}")

  # 7) Build context embedding for LSTM conditioning
  ctx_vec = rag.embed_context(reranked[:rag.cfg.top_k_classify]).astype(np.float32)

  # 8) Synthetic emissions time series
  series = make_synthetic_emission_series(T=360)  # 30 years monthly
  lookback = 12
  dataset = EmissionDataset(series, ctx_vec, lookback=lookback)
  ntrain = int(0.8 * len(dataset))
  train_set, test_set = torch.utils.data.random_split(dataset, [ntrain, len(dataset)-ntrain], generator=torch.Generator().manual_seed(42))
  train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

  model = LSTMRegressor(lookback=lookback, context_dim=len(ctx_vec), hidden=64, num_layers=1)
  optim = torch.optim.Adam(model.parameters(), lr=1e-3)
  loss_fn = nn.MSELoss()

  # 9) Train briefly (tiny demo)
  for epoch in range(5):
      model.train()
      tr_loss = 0.0
      for xb, yb in train_loader:
          optim.zero_grad()
          yhat = model(xb)
          loss = loss_fn(yhat, yb)
          loss.backward()
          optim.step()
          tr_loss += float(loss.item()) * len(xb)
      tr_loss /= len(train_loader.dataset)

      # Eval
      model.eval()
      with torch.no_grad():
          te_loss = 0.0
          for xb, yb in test_loader:
              yhat = model(xb)
              te_loss += float(loss_fn(yhat, yb).item()) * len(xb)
          te_loss /= len(test_loader.dataset)
      print(f"Epoch {epoch+1:02d} | Train MSE {tr_loss:.3f} | Test MSE {te_loss:.3f}")

  # 10) One-step forecast example (last window)
  model.eval()
  last_seq = torch.tensor(np.concatenate([series[-lookback:], ctx_vec]).astype(np.float32)).unsqueeze(0)
  with torch.no_grad():
      pred = model(last_seq).item()
  print(f"\nOne‑step ahead forecast: {pred:.2f}")
  print(f"Last observed value     : {series[-1]:.2f}")

  print("\nDone.")


if __name__ == "__main__":
  demo()
