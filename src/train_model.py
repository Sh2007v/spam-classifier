"""Train, compare and evaluate spam classifiers (guidance steps 2-10).
 
Usage (from the project root):
    python src/train_model.py
    python src/train_model.py --data data/SMSSpamCollection     # real UCI file
"""
import argparse
import os
 
import joblib
import matplotlib
 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
 
from preprocess import clean_text, load_dataset, prepare
 
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RANDOM_STATE = 42
 
 
def p(*parts):
    return os.path.normpath(os.path.join(ROOT, *parts))
 
 
# --------------------------------------------------------------------------- explore
def explore(df, charts_dir):
    """Step 3: missing values, duplicates, class balance, length, common words."""
    info = {
        "rows": len(df),
        "missing": int(df["message"].isna().sum()),
        "duplicates": int(df.duplicated(subset="message").sum()),
        "ham": int((df.label == "ham").sum()),
        "spam": int((df.label == "spam").sum()),
    }
    info["spam_pct"] = 100 * info["spam"] / info["rows"]
    df = df.assign(length=df["message"].str.len(), words=df["message"].str.split().str.len())
    info["avg_len_ham"] = df.loc[df.label == "ham", "length"].mean()
    info["avg_len_spam"] = df.loc[df.label == "spam", "length"].mean()
 
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="label", order=["ham", "spam"], hue="label",
                  palette=["#4C78A8", "#E45756"], legend=False, ax=ax)
    for c in ax.containers:
        ax.bar_label(c)
    ax.set_title("Class distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "class_distribution.png"), dpi=130)
    plt.close(fig)
 
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x="length", hue="label", bins=30, element="step",
                 palette={"ham": "#4C78A8", "spam": "#E45756"}, ax=ax)
    ax.set_title("Message length (characters)")
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "message_length.png"), dpi=130)
    plt.close(fig)
 
    common = {}
    for label in ("ham", "spam"):
        words = " ".join(df.loc[df.label == label, "clean"]).split()
        common[label] = pd.Series(words).value_counts().head(10)
    return info, common
 
 
# --------------------------------------------------------------------------- models
def make_models():
    """Model -> (estimator, parameter grid searched with 5-fold CV on the TRAIN set only)."""
    return {
        "Multinomial Naive Bayes": (MultinomialNB(), {"alpha": [1.0, 0.3, 0.1, 0.03]}),
        "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced",
                                                   random_state=RANDOM_STATE), {"C": [1, 3, 10]}),
        "Linear SVM": (LinearSVC(class_weight="balanced", random_state=RANDOM_STATE), {"C": [0.1, 0.3, 1]}),
    }
 
 
def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
 
 
def run_experiment(X_train_text, X_test_text, y_train, y_test, model, ngram, max_features):
    vec = TfidfVectorizer(ngram_range=ngram, max_features=max_features)
    Xtr = vec.fit_transform(X_train_text)  # fit on TRAIN only (no leakage)
    Xte = vec.transform(X_test_text)
    model.fit(Xtr, y_train)
    return vec, model, metrics(y_test, model.predict(Xte))
 
 
def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
 
 
def top_spam_words(vec, lr_model, n=15):
    names = np.array(vec.get_feature_names_out())
    coef = lr_model.coef_[0]
    order = np.argsort(coef)
    return list(zip(names[order[::-1][:n]], coef[order[::-1][:n]])), \
        list(zip(names[order[:n]], coef[order[:n]]))
 
 
def md_table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)
 
 
# --------------------------------------------------------------------------- main
def main(data_path):
    charts_dir = p("results", "charts")
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(p("models"), exist_ok=True)
 
    raw = load_dataset(data_path)
    n_before = len(raw)
    raw = raw.dropna(subset=["message"]).drop_duplicates(subset="message").reset_index(drop=True)
    df = prepare(raw)
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)  # nothing left after cleaning
 
    synthetic = os.path.basename(data_path) == "spam_dataset.csv"
    info, common = explore(df, charts_dir)
    print(f"Loaded {n_before} rows -> {len(df)} after removing duplicates/empty messages")
    print(f"Spam share: {info['spam_pct']:.1f}%  (imbalanced data: do not rely on accuracy alone)")
 
    # Step 5: stratified 80/20 split
    idx_train, idx_test = train_test_split(df.index, test_size=0.2, stratify=df["target"],
                                           random_state=RANDOM_STATE)
    train, test = df.loc[idx_train], df.loc[idx_test]
    y_train, y_test = train["target"].values, test["target"].values
 
    # Step 6/7: TF-IDF (unigrams+bigrams, 5000 features) and three models
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    Xtr = vec.fit_transform(train["clean"])
    Xte = vec.transform(test["clean"])
 
    results, fitted, preds, best_params = {}, {}, {}, {}
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    for name, (model, grid) in make_models().items():
        gs = GridSearchCV(model, grid, scoring="f1", cv=cv)  # tuned on training data only
        gs.fit(Xtr, y_train)
        fitted[name], best_params[name] = gs.best_estimator_, gs.best_params_
        preds[name] = gs.predict(Xte)
        results[name] = metrics(y_test, preds[name])
        print(f"{name:26s} {gs.best_params_}  " + "  ".join(f"{k}={v:.3f}" for k, v in results[name].items()))
 
    # Best model: highest spam F1 (ties broken by precision, since blocking real mail is worse)
    best_name = max(results, key=lambda n: (round(results[n]["f1"], 6), results[n]["precision"]))
    best_model, best_pred = fitted[best_name], preds[best_name]
    print(f"Best model: {best_name}")
 
    cm = confusion_matrix(y_test, best_pred)
    plot_confusion(cm, f"Confusion matrix - {best_name}", os.path.join(charts_dir, "confusion_matrix.png"))
 
    # model comparison chart
    comp = pd.DataFrame(results).T[["precision", "recall", "f1"]]
    ax = comp.plot(kind="bar", figsize=(7, 4), color=["#4C78A8", "#F58518", "#54A24B"], rot=15)
    ax.set_ylim(0, 1.05)
    ax.set_title("Spam-class metrics by model")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "model_comparison.png"), dpi=130)
    plt.close()
 
    # Step 9: error analysis
    errors = test.assign(pred=best_pred)
    fp = errors[(errors.target == 0) & (errors.pred == 1)]
    fn = errors[(errors.target == 1) & (errors.pred == 0)]
    n_words = lambda s: s.str.split().str.len()  # noqa: E731
    short_share = lambda d: (100 * (n_words(d["message"]) <= 5).mean()) if len(d) else 0.0  # noqa: E731
 
    # Step 10: before/after comparison (ngrams / model / class weights)
    exp_rows = []
    probes = ["You won a free prize! Click now to claim.",
              "Urgent! Your account is blocked. Verify your details immediately."]
    exps = [
        ("1. Baseline: all stop words removed, NB alpha=1, unigrams", "clean_basic", MultinomialNB(), (1, 1)),
        ("2. + bigrams", "clean_basic", MultinomialNB(), (1, 2)),
        ("3. + keep spam-signal words (you, your, now, ...)", "clean", MultinomialNB(), (1, 2)),
        ("4. + CV-tuned alpha (final Naive Bayes)", "clean",
         MultinomialNB(alpha=best_params["Multinomial Naive Bayes"]["alpha"]), (1, 2)),
        ("5. Logistic Regression without class weights", "clean", LogisticRegression(max_iter=2000), (1, 2)),
        ("6. Logistic Regression with class_weight=balanced (C=1)", "clean",
         LogisticRegression(max_iter=2000, class_weight="balanced"), (1, 2)),
    ]
    for label, col, m, ng in exps:
        v_, m_, r = run_experiment(train[col], test[col], y_train, y_test, m, ng, 5000)
        clean_fn = (lambda t: clean_text(t, keep_words=())) if col == "clean_basic" else clean_text
        if hasattr(m_, "predict_proba"):
            pp = m_.predict_proba(v_.transform([clean_fn(t) for t in probes]))[:, 1]
            probe_txt = " / ".join(f"{x:.2f}" for x in pp)
        else:
            probe_txt = "-"
        exp_rows.append([label, f"{r['accuracy']:.3f}", f"{r['precision']:.3f}",
                         f"{r['recall']:.3f}", f"{r['f1']:.3f}", probe_txt])
 
    # top spam words (from logistic regression coefficients)
    spam_w, ham_w = top_spam_words(vec, fitted["Logistic Regression"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    words, coefs = zip(*spam_w[::-1])
    ax.barh(words, coefs, color="#E45756")
    ax.set_title("Top words/phrases indicating spam (LogReg)")
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "top_spam_words.png"), dpi=130)
    plt.close(fig)
 
    # Save model + vectorizer
    joblib.dump(best_model, p("models", "spam_model.pkl"))
    joblib.dump(vec, p("models", "vectorizer.pkl"))
    joblib.dump({"best_model": best_name}, p("models", "model_info.pkl"))
 
    # ---------------------------------------------------------------- evaluation.md
    L = []
    L.append("# Evaluation Report\n")
    L.append(f"Dataset: `{os.path.basename(data_path)}` | Split: 80% train / 20% test, stratified, "
             f"random_state={RANDOM_STATE} | Features: TF-IDF, ngram_range=(1,2), max_features=5000\n")
    L.append("## Dataset summary\n")
    L.append(md_table([
        ["Total messages (after cleaning)", info["rows"]],
        ["Ham", info["ham"]], ["Spam", f"{info['spam']} ({info['spam_pct']:.1f}%)"],
        ["Duplicates left", info["duplicates"]], ["Missing values", info["missing"]],
        ["Avg length ham / spam (chars)", f"{info['avg_len_ham']:.0f} / {info['avg_len_spam']:.0f}"],
        ["Train / test size", f"{len(train)} / {len(test)}"],
    ], ["Item", "Value"]))
    L.append("\n\nMost common words (after cleaning):\n")
    for label in ("ham", "spam"):
        L.append(f"- **{label}**: " + ", ".join(f"{w} ({c})" for w, c in common[label].items()))
    L.append("\n## Model comparison (metrics are for the spam class, on unseen test data)\n")
    L.append(md_table(
        [[n, str(best_params[n]), f"{r['accuracy']:.3f}", f"{r['precision']:.3f}",
          f"{r['recall']:.3f}", f"{r['f1']:.3f}"] for n, r in results.items()],
        ["Model", "Best params (5-fold CV)", "Accuracy", "Precision", "Recall", "F1"]))
    L.append(f"\n\n**Selected model: {best_name}** (highest spam F1; saved to `models/`).\n")
    L.append("![Model comparison](charts/model_comparison.png)\n")
    L.append("## Classification report (selected model)\n")
    L.append("```\n" + classification_report(y_test, best_pred, target_names=["ham", "spam"], digits=3) + "```\n")
    L.append("## Confusion matrix (selected model)\n")
    L.append(md_table([["Actual ham", cm[0, 0], cm[0, 1]], ["Actual spam", cm[1, 0], cm[1, 1]]],
                      ["", "Predicted ham", "Predicted spam"]))
    L.append("\n\n![Confusion matrix](charts/confusion_matrix.png)\n")
    L.append("## Error analysis\n")
    L.append(f"- False positives (real message marked spam): **{len(fp)}**")
    L.append(f"- False negatives (spam missed): **{len(fn)}**")
    L.append(f"- Share of messages with 5 words or fewer: all test {short_share(test):.0f}%, "
             f"false positives {short_share(fp):.0f}%, false negatives {short_share(fn):.0f}%")
    L.append("- Which error matters more? Blocking a real message (false positive) is usually worse, "
             "so precision is watched closely, but low recall lets spam reach users.\n")
    L.append("### False positives\n")
    L.extend([f"- `{m}`" for m in fp["message"].head(15)] or ["- none"])
    L.append("\n### False negatives\n")
    if synthetic:
        L.append("_Note: most of these read like ordinary chat. They are labeled spam because of the ~2% label noise "
                 "built into the synthetic dataset, so they show the recall ceiling rather than a model weakness._\n")
    L.extend([f"- `{m}`" for m in fn["message"].head(15)] or ["- none"])
    L.append("\n## Improving the model (before vs after)\n")
    L.append(md_table(exp_rows, ["Configuration", "Accuracy", "Precision", "Recall", "F1", "Spam prob. of 2 obvious spam examples"]))
    L.append("\n\n## Top words that indicate spam (Logistic Regression)\n")
    L.append(", ".join(f"`{w}` ({c:.2f})" for w, c in spam_w))
    L.append("\n\nTop words that indicate ham: " + ", ".join(f"`{w}`" for w, _ in ham_w[:10]) + "\n")
    L.append("![Top spam words](charts/top_spam_words.png)\n")
    L.append("## Limitations\n")
    if synthetic:
        L.append("- The dataset is synthetic (template-generated), so scores are optimistic compared with real traffic.")
        L.append("- It deliberately contains ~2% flipped labels, which caps recall: some false negatives are mislabeled messages.")
        L.append("- Only English and light Hinglish are represented.")
    else:
        L.append("- Trained on one dataset (SMS Spam Collection: mostly UK English SMS from the 2000s). "
                 "Other regions, languages and newer spam styles (e.g. Indian loan or KYC phishing) are underrepresented.")
    L.append("- Bag-of-words models ignore context; spam that avoids typical spam words is easily missed.")
    L.append("- Spammers adapt, so the model needs periodic retraining on fresh data.\n")
    with open(p("results", "evaluation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("Saved models/ and results/evaluation.md and results/charts/")
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=p("data", "spam_dataset.csv"))
    main(ap.parse_args().data)
 