import os
import re
import string
import joblib
import pandas as pd
import streamlit as st
BASE_DIR=r"D:\IOT\ML\NLP\HOTELREVIEWS"
OUT_DIR=os.path.join(BASE_DIR, "sentiment_artifacts")
ARTIFACT_PATH=os.path.join(OUT_DIR, "hotel_sentiment_pipeline.joblib")
COMPARE_CSV=os.path.join(OUT_DIR, "model_comparison.csv")
CM_PATH=os.path.join(OUT_DIR, "best_confusion_matrix.png")
EDA_TXT=os.path.join(OUT_DIR, "eda_summary.txt")
STOPWORDS=set("""
a an the is are was were am be been being and or but if then else
this that those these i me my we our you your he she it they to of in on at
as not no do does did doing done very so too can could should would will
""".split())
def clean_text(text: str)->str:
    text=str(text).lower()
    text=re.sub(r"http\S+|www\S+", " ", text)
    text=re.sub(r"\d+", " ", text)
    text=text.translate(str.maketrans("", "", string.punctuation))
    text=re.sub(r"[^a-z\s]", " ", text)
    text=re.sub(r"\s+", " ", text).strip()
    words=[w for w in text.split() if w not in STOPWORDS and len(w)>2]
    return " ".join(words)
st.set_page_config(page_title="Hotel Review Sentiment (NLP)",layout="centered")
@st.cache_resource
def load_artifact(path: str):
    return joblib.load(path)
def main():
    st.title("Hotel Reviews Sentiment Analysis (NLP) by FERDIANND C SUNGOH ,BATCH-11")
    st.write("Predict sentiment using a trained **scikit-learn pipeline**.")
    if not os.path.exists(ARTIFACT_PATH):
        st.error("Trained model not found.\n\n""Please run training (notebook/script) to create:\n"f"`{ARTIFACT_PATH}`")
        st.stop()
    artifact=load_artifact(ARTIFACT_PATH)
    pipeline=artifact["pipeline"]
    label_map=artifact.get("label_map", {"Negative": 0, "Positive": 1})
    if all(isinstance(k, str) for k in label_map.keys()):
        inv_label={v: k for k, v in label_map.items()}
    else:
        inv_label=dict(label_map)
    best_model_name=artifact.get("best_model_name", "Unknown")
    low_thr=artifact.get("low_quantile_thr", None)
    high_thr=artifact.get("high_quantile_thr", None)
    st.success(f"Best Model: `{best_model_name}`")
    st.sidebar.header("Model Information")
    st.sidebar.success(best_model_name)
    if low_thr is not None and high_thr is not None:
        st.sidebar.markdown("Training Labeling:Rating Percentiles")
        st.sidebar.write(f"Negative: rating ≤ **{float(low_thr):.3f}**")
        st.sidebar.write(f"Positive: rating ≥ **{float(high_thr):.3f}**")
    else:
        st.sidebar.markdown("**Training Labeling: Text Based**")
        st.sidebar.write("Positive_Review → Positive")
        st.sidebar.write("Negative_Review → Negative")
    tab1,tab2=st.tabs(["Predict","Reports"])
    with tab1:
        st.subheader("Sentiment Prediction")
        mode=st.radio("Choose input mode",["Single Review", "Bulk Reviews"],horizontal=True)
        if mode=="Single Review":
            text=st.text_area("Enter hotel review",height=160,placeholder="Example: The room was clean and staff were friendly...")
            if st.button("Predict Sentiment"):
                if not text.strip():
                    st.warning("Please enter review text.")
                else:
                    pred=pipeline.predict([text])[0]
                    label=inv_label.get(int(pred), str(pred))

                    if str(label).lower()=="positive":
                        st.success(f"Sentiment:{label}")
                    else:
                        st.error(f"Sentiment:{label}")
        else:
            bulk=st.text_area(
                "Enter one review per line",height=220,placeholder="Review 1...\nReview 2...\nReview 3...")
            if st.button("Predict All"):
                reviews=[r.strip() for r in bulk.split("\n") if r.strip()]
                if not reviews:
                    st.warning("Please enter at least one review.")
                else:
                    preds=pipeline.predict(reviews)
                    labels=[inv_label.get(int(p), str(p)) for p in preds]
                    out=pd.DataFrame({"Review": reviews, "Prediction": labels})
                    st.dataframe(out, use_container_width=True)                
                    st.download_button("Download Predictions CSV",data=out.to_csv(index=False).encode("utf-8"),file_name="hotel_sentiment_predictions.csv",mime="text/csv")
    with tab2:
        st.subheader("Reports")
        st.info(f"Best Model Selected: `{best_model_name}`")
        if os.path.exists(COMPARE_CSV):
            st.markdown("Model Comparison")
            st.dataframe(pd.read_csv(COMPARE_CSV), use_container_width=True)
        else:
            st.warning("model_comparison.csv not found. Train model to generate it.")
        if os.path.exists(EDA_TXT):
            st.markdown("EDA Summary")
            with open(EDA_TXT, "r", encoding="utf-8") as f:
                st.text(f.read())
        else:
            st.info("EDA summary not found (optional).")
        st.markdown("Confusion Matrix")
        if os.path.exists(CM_PATH):
            st.image(CM_PATH, caption=f"Confusion Matrix - {best_model_name}", use_column_width=True)
        else:
            st.info("Confusion matrix image not found. Save it during training to display here.")
if __name__ == "__main__":
    main()
