# LLM-Fraud-Transactions-Officer

This project provides an **open‑source** Streamlit application and supporting Python script that leverage an LLM (e.g., Meta’s Llama 2 chat models) to convert raw transaction details into human‑readable descriptions, risk indicators, and contextual fraud interpretations.

---

## 🔍 Goals & Purpose

1. **Streamline Fraud Review**: Help Anti‑Fraud Officers quickly understand individual transactions by auto‑generating:
   - A concise summary
   - Key risk indicators
   - Contextual insights and fraud patterns
2. **Interactive Data Entry**: Allow users to input transaction fields manually (or in future via CSV upload) and receive immediate feedback, including warnings when required details are missing.
3. **Open‑Source & Extensible**: Use freely available LLMs and Hugging Face resources so organizations can self‑host, customize prompts, and integrate into existing workflows without API costs.

---

## 🛠️ Technologies & Libraries

- **Python 3.8+**
- **Streamlit** for building the interactive web UI
- **Transformers** (Hugging Face) for model loading and text generation
- **Torch** for model execution (GPU/CPU)
- **Pandas** for any CSV ingestion or data handling

---

## ⚙️ Setup & Execution

1. **Clone the repo**
   ```bash
   git clone https://github.com/your‑org/transaction‑describer.git
   cd transaction‑describer
   ```

2. **Create a virtual environment & install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   export HUGGINGFACE_HUB_TOKEN="<your_token>"    # optional for public models
   export MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"  # or another HF model path
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run llm_transaction_describer.py
   ```

5. **Use the web interface**
   - Input transaction fields (ID, amount, location, device, merchant, timestamp, account age).
   - Click **Describe Transaction**.
   - View the generated summary, risk indicators, and contextual interpretation.

---

## 🚀 Next Steps

- **Batch CSV upload** and bulk processing
- **Advanced input validation** (timestamp parsing, value ranges)
- **Deployment** via Docker or cloud platforms
- **Caching & rate‑limiting** for production readiness
- **UI enhancements**: history tracking, filtering, and exporting results

---

*Feel free to contribute, file issues, or suggest improvements!*


