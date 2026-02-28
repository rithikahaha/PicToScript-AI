# PicToScript-AI — Image Captioning with Xception + LSTM

Automatically generates captions for images using a CNN + LSTM architecture, trained on the Flickr8k dataset.

## How it works

- **Xception** (pretrained on ImageNet) extracts a 2048-dimensional feature vector from each image
- An **LSTM** language model takes those features and generates a caption word by word
- Trained on ~7,300 images from the Flickr8k dataset over 20 epochs
- Evaluated using **BLEU score** (BLEU-1 through BLEU-4)
- Includes a **Gradio web demo** where you can upload any image and get a caption in real time

## Project Structure

```
├── PicToScript.ipynb   # Main notebook — end to end pipeline
├── requirements.txt    # Dependencies
└── README.md
```

## Setup

This project is designed to run on **Google Colab** with a T4 GPU.

1. Open `PicToScript.ipynb` in Google Colab
2. Set runtime to GPU: Runtime > Change runtime type > T4 GPU
3. Run all cells from top to bottom

You'll need a Kaggle account to download the Flickr8k dataset. The notebook will prompt you to upload your `kaggle.json` API token.

## Results

| Metric | Score |
|--------|-------|
| BLEU-1 | 0.534 |
| BLEU-2 | 0.342 |
| BLEU-3 | 0.214 |
| BLEU-4 | 0.132 |

## Tech Stack

- Python
- TensorFlow / Keras
- Xception (CNN)
- LSTM
- NLTK (BLEU scoring)
- Gradio (live demo)
- Google Colab
