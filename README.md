
# Clarix — Support Ticket Classification System

> **FUTURE_ML_02** · Future Interns Machine Learning Track · Task 2

---

## 📌 Project Overview

**Clarix** is an intelligent support ticket classification system that automatically categorises customer support tickets into 6 categories and assigns priority levels (High / Medium / Low) using Natural Language Processing and Machine Learning.

Built as part of the **Future Interns ML Internship — Task 2**.

---

## 🚀 Live Dashboard

🔗 **[https://future-ml-02.netlify.app](https://future-ml-02.netlify.app)**

The interactive dashboard includes:
- Dataset overview with charts and heatmaps
- Model performance comparison across 4 ML algorithms
- Confusion matrices and per-class metrics
- Live ticket classifier with real-time predictions

---

## 🗂️ Dataset

- **600** synthetically generated support tickets
- **6 Categories:** Billing, Technical Support, Account Management, Feature Request, Shipping & Delivery, General Inquiry
- **3 Priority Levels:** High, Medium, Low
- **Average ticket length:** 8.8 words (post-cleaning)

---

## 🤖 Models Trained

| Model | Category Accuracy | Weighted F1 | CV Accuracy |
|---|---|---|---|
| Logistic Regression | 100% | 100% | 100% |
| Linear SVM | 100% | 100% | 100% |
| Random Forest | 100% | 100% | 100% |
| Naive Bayes | 100% | 100% | 100% |

> All models achieved perfect category classification. Priority classification is a harder task with macro F1 ~43% due to class overlap.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Scikit-learn | ML models & pipelines |
| TF-IDF Vectorizer | Text feature extraction |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Data visualisation |
| HTML / CSS / JS | Interactive dashboard |
| Netlify | Live deployment |
| GitHub | Version control |

---

## 📁 File Structure

```
FUTURE_ML_02/
├── support_ticket_classifier.py   # Main ML pipeline + chart generation
├── index.html                     # Clarix interactive dashboard
├── 01_dataset_overview.png        # Dataset charts
├── 02_text_statistics.png         # Text analysis charts
├── 03_model_comparison.png        # Model performance comparison
├── 04_confusion_matrix.png        # Category confusion matrix
├── 05_per_class_metrics.png       # Per-class precision/recall/F1
├── 06_priority_results.png        # Priority classification results
└── 07_live_demo.png               # Live prediction showcase
```

---

## ▶️ How to Run

1. **Clone the repository**
```bash
git clone https://github.com/madhurimauppathil-pixel/FUTURE_ML_02.git
cd FUTURE_ML_02
```

2. **Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

3. **Run the classifier**
```bash
python support_ticket_classifier.py
```

4. **View the dashboard**
Open `index.html` in your browser or visit the live link above.

---

## 📊 Key Results

- ✅ **Category Classification** — 100% accuracy across all 4 models
- ⚠️ **Priority Classification** — ~43% macro F1 (challenging due to subjective priority labels)
- 🔍 **Best insight** — Technical Support tickets have the highest average word count (10.0 words), suggesting more complex issues

---

## 👩‍💻 Author

**Madhurima Mani** · ML Intern · Future Interns

---

*Future Interns Machine Learning Track — Task 2 · Support Ticket Classification*
