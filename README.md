# KP - Basic Cell Migration Analysis (alpha)

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-alpha-orange.svg)
![GitHub repo size](https://img.shields.io/github/repo-size/bigaxon/kinetic-pathology-basic-analyzer)
![Last commit](https://img.shields.io/github/last-commit/bigaxon/kinetic-pathology-basic-analyzer)

> ⚠️ This is an **alpha release** (`v1.0.0-alpha`). Expect changes and possible bugs. Feedback is welcome!

***

Comprehensive in situ cell migration analyses based on Carbonell et al, 2005b. Does in 7 seconds what took me 7 days to do during my dissertation. 😭

*Reference:*
Carbonell, W. S., Murase, S.-I., Horwitz, A. F., & Mandell, J. W. (2005). Migration of perilesional microglia after focal brain injury and modulation by CC chemokine receptor 5: An in situ time-lapse confocal imaging study. Journal of Neuroscience, 25(30), 7040–7047. Full text: https://www.jneurosci.org/content/25/30/7040

---

## 📦 Installation

Clone this repo and install dependencies:

```bash
git clone https://github.com/bigaxon/kinetic-pathology-basic-analyzer.git
cd kinetic-pathology-basic-analyzer
pip install -r requirements.txt
```
***

## ▶️ Usage
Run the script on a sample movie:

```bash
python kinetic_basic_repo_app.py --input path/to/movie.mov --out results/
```
Options:
 - input → path to input movie file (MOV, MP4, etc.)
 - out → output directory for results

***

## 📂 Project Structure

```bash
kinetic-pathology-basic-analyzer/
├── kinetic_basic_repo_app.py      # Core analysis script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── LICENSE                        # License file
└── .gitignore                     # Ignored files
```

***

## 🚀 Roadmap
- [ ] Figure out what the fuck I am doing
- [ ] Do it
- [ ] If I love it, do it for the rest of my life

***

## ⚠️ Assumptions
This analyzer is optimized for use with MP4 or MOV files with the following constraints:
  1. vital labeling of cells with fluorescent IB4 lectin
  2. 20x objective
  3. 90 second interval

***

## 🎥 Sample Videos
(from Carbonell et al., 2005b)
  1. [MOVIE1_TEST.MOV](https://drive.google.com/file/d/1S594j7KRyqeTHFsbO16WXPlp8TC5ld19/view?usp=sharing) - thalamus 24h after lesion (inflammatory condition)
  2. [MOVIE2_TEST.MOV](https://drive.google.com/file/d/139URdppe6fwoIbUhToub4YzLTGKqzNmQ/view?usp=sharing) - thalamus 24h after sham lesion (naive control)

***

## 🙌🏽 Acknowledgements
  1. My co-CTOs: ChatGPT 5 & Claude Sonnet 4
  2. OpenCV, scikit-image, pandas, matplotlib
  3. All y'all

