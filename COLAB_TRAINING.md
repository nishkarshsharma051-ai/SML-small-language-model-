# Ting Ling Ling - Google Colab Training Notebook

Run this notebook in Google Colab with a free T4 GPU runtime to fine-tune **Ting Ling Ling** fast!

---

### Step 1: Install Dependencies
```bash
!pip install -q torch transformers datasets peft accelerate trl
```

---

### Step 2: Clone or Upload Project Files
If training directly in Colab, upload `data/hf_sft_train.jsonl` and `data/hf_sft_val.jsonl` to your Colab session files, or run this block to build the dataset:

```python
# Upload data_builder.py and study_data.py, then build dataset:
!python3 data_builder.py
```

---

### Step 3: Run Main Fine-Tuning (`train_core.py`)
Run the training script using GPU acceleration:

```python
!python3 train_core.py --base-model "Qwen/Qwen2.5-0.5B-Instruct" --output-dir "hf_local_model" --epochs 5 --lr 2e-4
```

---

### Step 4: Evaluate Model Performance
Test the fine-tuned model directly inside Colab:

```python
!python3 eval_core.py --model-dir "hf_local_model"
```

---

### Step 5: Zip and Download Model Artifacts
Zip the fine-tuned LoRA model weights to download back to your local workspace:

```python
!zip -r ting_ling_ling_colab_model.zip hf_local_model/
from google.colab import files
files.download("ting_ling_ling_colab_model.zip")
```
