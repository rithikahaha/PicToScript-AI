
# PicToScript-AI — Image Captioning with Xception + LSTM

A deep learning project that automatically generates natural language captions for images, built and trained on the Flickr8k dataset.

---

## Project Overview

PicToScript-AI takes any image as input and outputs a human-readable caption describing what's happening in it. The model learns to "read" an image by combining visual features extracted from a CNN with a language model that generates words one at a time — mimicking how a person might describe a photo.

---

## Problem Statement

Automatically describing the content of an image in natural language is a classic challenge at the intersection of computer vision and NLP. A good image captioning model needs to:
- Understand what objects and actions are present in an image
- Generate a grammatically correct and contextually relevant sentence
- Generalise to images it has never seen before

This project tackles that problem using a CNN encoder + LSTM decoder architecture trained end-to-end on real-world image-caption pairs.

---

## Dataset

**Flickr8k** — a widely used benchmark dataset for image captioning.

| Property | Value |
|----------|-------|
| Total images | 8,091 |
| Captions per image | 5 (human written) |
| Total captions | ~40,000 |
| Train split | 7,281 images (90%) |
| Test split | 810 images (10%) |

Captions are lowercased, stripped of punctuation, and wrapped with `<start>` and `<end>` tokens before training.

---

## Model Architecture

The model takes two inputs — an image and a partial caption — and predicts the next word. This is repeated until the `<end>` token is generated.

**Image Encoder (Xception CNN)**
- Xception pretrained on ImageNet, top layer removed
- Outputs a 2048-dimensional feature vector per image
- Features are extracted once and saved to disk before training
- A Dense layer (256 units, ReLU) + Dropout(0.5) compresses the features

**Caption Decoder (LSTM)**
- Embedding layer maps each word token to a 256-dimensional vector
- Dropout(0.5) applied for regularisation
- LSTM (256 units) processes the token sequence
- Image and caption representations are merged (element-wise add)
- Final Dense layer with softmax predicts the next word over the full vocabulary

**Training**
- Loss: Categorical Crossentropy
- Optimiser: Adam
- Epochs: 20
- Environment: Google Colab T4 GPU (~25 minutes)

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

## Sample Outputs

**Example 1**

![sample1](https://github.com/user-attachments/assets/0dbe9b8f-ac86-4910-9dc9-217ccf953f71)

**Example 2**

![sample2](https://github.com/user-attachments/assets/f87fdf29-646f-4cbc-b0e0-8c1ddc611a39)

**Example 3**

![sample3](https://github.com/user-attachments/assets/7295a67c-73ee-4c73-8aa0-cf07a6484968)

### Observations

The model performs well on common, straightforward scenes — Example 1 correctly identifies the skateboarder and the action, only getting the location slightly wrong. However it struggles with unusual or complex images, as seen in Examples 2 and 3 where it misidentifies gender, colour, and context.

This is expected behaviour for a single-layer LSTM trained on 7k images. The model has learned general patterns (people, actions, locations) but lacks the capacity to pick up on finer visual details. Performance could be improved with a larger dataset, attention mechanisms, or a Transformer-based decoder.

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

## Gradio Demo

After training, the model is deployed as an interactive web app using Gradio. Upload any image and the model generates a caption in real time. Running `demo.launch(share=True)` produces a public link valid for 72 hours.

---

## Setup

1. Open `PicToScript.ipynb` in Google Colab
2. Set runtime to GPU: Runtime > Change runtime type > T4 GPU
3. Run all cells top to bottom

You'll need a Kaggle account to download the dataset. The notebook will prompt you to upload your `kaggle.json` API token.

---

## Project Structure

```
├── PicToScript.ipynb   # Full pipeline — preprocessing to demo
├── requirements.txt    # Dependencies
├── sample1.png         # Sample output 1
├── sample2.png         # Sample output 2
├── sample3.png         # Sample output 3
└── README.md
```
