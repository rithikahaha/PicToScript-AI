[README (2).md](https://github.com/user-attachments/files/25640344/README.2.md)
# PicToScript-AI — Image Captioning with Xception + LSTM

A deep learning project that automatically generates natural language captions for images. Built and trained on the Flickr8k dataset using a CNN + LSTM architecture, with BLEU score evaluation and a live Gradio demo.

---

## Architecture

The model combines two inputs — image features and partial captions — to predict the next word one at a time until a full caption is generated.

**Image encoder:**
- Xception (pretrained on ImageNet, top layer removed) extracts a 2048-dimensional feature vector from each image
- A Dense layer (256 units, ReLU) compresses this down with 0.5 Dropout for regularization

**Caption decoder:**
- An Embedding layer maps each word token to a 256-dimensional vector
- An LSTM (256 units) processes the token sequence and learns language patterns
- The image and caption representations are merged and passed through a final softmax Dense layer to predict the next word

---

## Data Pipeline

1. **Dataset** — Flickr8k: 8,091 images, each with 5 human-written captions (~40,000 captions total)
2. **Preprocessing** — captions are lowercased, punctuation stripped, and wrapped with `<start>` / `<end>` tokens
3. **Train / Test split** — 90/10 split (7,281 training images, 810 test images), seeded for reproducibility
4. **Tokenization** — fitted on training captions only; vocabulary size ~8,400 words
5. **Feature extraction** — Xception processes every image once and saves features to disk to avoid recomputing during training
6. **Sequence generation** — each caption is expanded into multiple (image, partial caption → next word) training pairs

---

## Results

Evaluated on 200 test images using corpus BLEU score:

| Metric | Score |
|--------|-------|
| BLEU-1 | 0.534 |
| BLEU-2 | 0.342 |
| BLEU-3 | 0.214 |
| BLEU-4 | 0.132 |

A BLEU-1 of 0.534 means roughly half the words in the generated captions match the reference captions — competitive for a single-layer LSTM trained from scratch on 7k images.

---

## Gradio Demo

After training, the model is deployed as an interactive web app using Gradio. Upload any image and the model generates a caption in real time. Running `demo.launch(share=True)` produces a public link valid for 72 hours.

---

## Setup

Designed to run on **Google Colab** with a T4 GPU.

1. Open `PicToScript.ipynb` in Google Colab
2. Set runtime to GPU: Runtime > Change runtime type > T4 GPU
3. Run all cells top to bottom

You'll need a Kaggle account to download the dataset. The notebook will prompt you to upload your `kaggle.json` API token.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Image Encoder | Xception (CNN) |
| Caption Decoder | LSTM |
| Evaluation | NLTK (BLEU) |
| Demo | Gradio |
| Environment | Google Colab (T4 GPU) |

---

## Project Structure

```
├── PicToScript.ipynb   # Full pipeline — preprocessing to demo
├── requirements.txt    # Dependencies
└── README.md
```
