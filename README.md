# Neural_Style_Transfer
# ��� Neural Style Transfer (NST)

This project blends the *content* of one image with the *style* of another using **Neural Style Transfer** based on a pre-trained VGG19 network.

---

## ��� Setup

### ��� Option 1: Automatic Setup

```bash
zsh setup_neural_style_transfer.sh
```

### ��� Option 2: Manual Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ��� How to Use

1. Replace `content.jpg` and `style.jpg` with your own images.
2. Run:

```bash
python main.py
```

3. After training, check `output.png` for the result!

---

## ���️ Features

- Content Reconstruction using VGG19 features
- Style Reconstruction using Gram Matrices
- Basic training loop with Adam optimizer

---

## ��� License

MIT License
