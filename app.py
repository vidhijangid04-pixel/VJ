
import streamlit as st
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
st.set_page_config(layout="wide", page_title="Employee Attrition Dashboard")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

def label_encode_df(df, encoders=None):
    encs = {} if encoders is None else encoders.copy()
    out = df.copy()
    for col in out.select_dtypes(include=['object','category']).columns:
        if encoders and col in encoders:
            le = encoders[col]
            out[col] = le.transform(out[col].astype(str).fillna("Missing"))
        else:
            le = LabelEncoder()
            out[col] = out[col].astype(str).fillna("Missing")
            out[col] = le.fit_transform(out[col])
            encs[col] = le
    return out, encs

def encode_single_df_with_encoders(df, encoders):
    out = df.copy()
    for col, le in encoders.items():
        if col in out.columns:
            out[col] = out[col].astype(str).fillna("Missing")
            unseen = set(out[col].unique()) - set(le.classes_)
            if unseen:
                le_classes = list(le.classes_) + list(unseen)
                le.classes_ = np.array(le_classes, dtype=object)
            out[col] = le.transform(out[col])
    return out

def download_link(df, filename="predictions.csv"):
    b = BytesIO()
    df.to_csv(b, index=False)
    b.seek(0)
    st.download_button("Download CSV", data=b, file_name=filename, mime="text/csv")

SAMPLE_CSV = "EA.csv"
if os.path.exists(SAMPLE_CSV):
    df_raw = load_data(SAMPLE_CSV)
else:
    df_raw = pd.DataFrame()

st.title("Employee Attrition — HR Analytics Dashboard")
tabs = st.tabs(["Dashboard", "Models (Train & Evaluate)", "Predict (Upload & Predict)", "About / README"])

with tabs[0]:
    st.header("Interactive Dashboard — retention insights")
    if df_raw.empty:
        st.warning("No sample dataset found. Upload a dataset in Predict tab to use dashboard.")
    else:
        df = df_raw.copy()
        satisfaction_cols = [c for c in df.columns if 'satisf' in c.lower() or 'satisfaction' in c.lower()]
        sat_col = satisfaction_cols[0] if satisfaction_cols else None
        jobrole_col = None
        candidates = [c for c in df.columns if 'job' in c.lower() and 'role' in c.lower()]
        if candidates:
            jobrole_col = candidates[0]
        else:
            if 'JobRole' in df.columns:
                jobrole_col = 'JobRole'

        st.sidebar.subheader("Filters (global)")
        if jobrole_col:
            jobroles = sorted(df[jobrole_col].dropna().unique().tolist())
            selected_roles = st.sidebar.multiselect("Job Role (multi-select)", options=jobroles, default=jobroles)
        else:
            selected_roles = None

        if sat_col:
            sat_min = float(df[sat_col].min())
            sat_max = float(df[sat_col].max())
            sat_range = st.sidebar.slider("Satisfaction slider", min_value=sat_min, max_value=sat_max, value=(sat_min, sat_max))
        else:
            sat_range = None

        df_f = df.copy()
        if jobrole_col and selected_roles is not None:
            df_f = df_f[df_f[jobrole_col].isin(selected_roles)]
        if sat_col and sat_range:
            df_f = df_f[(df_f[sat_col] >= sat_range[0]) & (df_f[sat_col] <= sat_range[1])]

        if jobrole_col:
            attr_by_role = df_f.groupby(jobrole_col)['Attrition'].apply(lambda x: (x.astype(str).str.lower().isin(['yes','1','true'])).mean()).reset_index(name='AttritionRate')
            fig1 = px.bar(attr_by_role.sort_values('AttritionRate',ascending=False), x=jobrole_col, y='AttritionRate', title="Attrition Rate by Job Role", labels={'AttritionRate':'Attrition Rate'})
            st.plotly_chart(fig1, use_container_width=True)

        if sat_col and 'MonthlyIncome' in df_f.columns:
            df_plot = df_f.copy()
            df_plot['Attrition_flag'] = df_plot['Attrition'].astype(str).str.lower().map({'yes':1,'no':0,'1':1,'0':0,'true':1,'false':0}).fillna(0).astype(int)
            fig2 = px.scatter(df_plot, x=sat_col, y='MonthlyIncome', color='Attrition_flag', title=f"Satisfaction vs Monthly Income (colored by Attrition)")
            st.plotly_chart(fig2, use_container_width=True)

        tenure_candidates = [c for c in df.columns if 'tenure' in c.lower() or 'years' in c.lower() or 'service' in c.lower()]
        tenure_col = tenure_candidates[0] if tenure_candidates else None
        if tenure_col:
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(x=df_f[tenure_col], nbinsx=20, name='All Employees', opacity=0.6))
            if 'Attrition' in df_f.columns:
                left = df_f[df_f['Attrition'].astype(str).str.lower().isin(['yes','1','true'])][tenure_col]
                fig3.add_trace(go.Histogram(x=left, nbinsx=20, name='Left (Attrition=Yes)', opacity=0.75))
            fig3.update_layout(barmode='overlay', title="Tenure Distribution (with Attrition overlay)", xaxis_title=tenure_col)
            st.plotly_chart(fig3, use_container_width=True)

        num = df_f.select_dtypes(include=[np.number])
        if not num.empty:
            corr = num.corr()
            fig4, ax4 = plt.subplots(figsize=(10,6))
            sns.heatmap(corr, ax=ax4, cmap='coolwarm', center=0)
            ax4.set_title("Correlation matrix (numeric features)")
            st.pyplot(fig4)

        overtime_cols = [c for c in df.columns if 'overtime' in c.lower()]
        ot_col = overtime_cols[0] if overtime_cols else None
        if ot_col:
            stacked = df_f.groupby([ot_col, 'Attrition']).size().reset_index(name='count')
            stacked['Attrition'] = stacked['Attrition'].astype(str)
            fig5 = px.bar(stacked, x=ot_col, y='count', color='Attrition', title=f"Attrition counts by {ot_col}", barmode='stack')
            st.plotly_chart(fig5, use_container_width=True)

        st.markdown('---')
        st.subheader("Top action points (automated suggestions)")
        suggestions = []
        if sat_col:
            low_sat = df_f[df_f[sat_col] < df[sat_col].quantile(0.25)]
            if not low_sat.empty and 'Attrition' in low_sat.columns:
                rate = (low_sat['Attrition'].astype(str).str.lower().isin(['yes','1','true'])).mean()
                suggestions.append(f"Employees with low {sat_col} have attrition rate ~{rate:.2f}. Consider targeted engagement/surveys.")
        if jobrole_col:
            high_roles = attr_by_role.sort_values('AttritionRate', ascending=False).head(3)[jobrole_col].tolist()
            suggestions.append(f"High-risk roles: {', '.join(high_roles)} — investigate role-specific interventions (compensation, career path).")
        if suggestions:
            for s in suggestions:
                st.info(s)
        else:
            st.write("No automated suggestions available (insufficient columns).")

with tabs[1]:
    st.header("Model training & evaluation (cv=5, stratified)")
    if df_raw.empty:
        st.warning("No sample dataset available. Upload a dataset in the Predict tab to use model training.")
    else:
        df0 = df_raw.copy()
        st.write("Columns detected:", df0.columns.tolist())
        df_enc, encoders = label_encode_df(df0)
        X = df_enc.drop(columns=['Attrition'])
        y = df_enc['Attrition']

        test_size = st.slider("Test set size (percent)", min_value=10, max_value=40, value=20)
        if st.button("Run training & evaluation"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, stratify=y, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            models = {
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
            }

            results = []
            roc_data = {}
            for name, model in models.items():
                with st.spinner(f"Training {name}..."):
                    oof_pred = cross_val_predict(model, X_train, y_train, cv=cv, method='predict')
                    train_acc = accuracy_score(y_train, oof_pred)
                    model.fit(X_train, y_train)
                    y_test_pred = model.predict(X_test)
                    test_acc = accuracy_score(y_test, y_test_pred)
                    prec = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
                    rec = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
                    f1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_test)[:,1]
                        auc = roc_auc_score(y_test, y_proba)
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        roc_data[name] = (fpr, tpr, auc)
                    else:
                        auc = None
                        roc_data[name] = (None, None, None)

                    results.append({"Model":name, "Train Acc":train_acc, "Test Acc":test_acc,
                                    "Precision":prec, "Recall":rec, "F1":f1, "AUC":auc})

                    cm_train = confusion_matrix(y_train, oof_pred)
                    cm_test = confusion_matrix(y_test, y_test_pred)
                    fig, axes = plt.subplots(1,2, figsize=(10,4))
                    sns.heatmap(cm_train, annot=True, fmt='d', ax=axes[0], cmap='Blues', cbar=False)
                    axes[0].set_title(f"{name} — Training (OOF) CM")
                    axes[0].set_xlabel("Predicted")
                    axes[0].set_ylabel("Actual")
                    sns.heatmap(cm_test, annot=True, fmt='d', ax=axes[1], cmap='OrRd', cbar=False)
                    axes[1].set_title(f"{name} — Test CM")
                    axes[1].set_xlabel("Predicted")
                    axes[1].set_ylabel("Actual")
                    st.pyplot(fig)

                    if hasattr(model, "feature_importances_"):
                        fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
                        fig2, ax2 = plt.subplots(figsize=(8,4))
                        sns.barplot(x=fi.values, y=fi.index, ax=ax2)
                        ax2.set_title(f"{name} — Top features")
                        st.pyplot(fig2)

            res_df = pd.DataFrame(results).set_index("Model")
            st.subheader("Model comparison (test metrics)")
            st.dataframe(res_df.style.highlight_max(axis=0))
            if any([v[0] is not None for v in roc_data.values()]):
                fig_roc = go.Figure()
                for k,(fpr,tpr,auc) in roc_data.items():
                    if fpr is not None:
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{k} (AUC={auc:.3f})"))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
                fig_roc.update_layout(title="ROC — Test set", xaxis_title="FPR", yaxis_title="TPR", width=800, height=500)
                st.plotly_chart(fig_roc)
            st.session_state['encoders'] = encoders
            st.session_state['trained_models'] = models
            st.session_state['last_results'] = res_df

with tabs[2]:
    st.header("Upload dataset and predict Attrition")
    uploaded = st.file_uploader("Upload CSV file with same columns as training data", type=['csv'])
    if uploaded is not None:
        new_df = pd.read_csv(uploaded)
        st.write("Preview of uploaded data:")
        st.dataframe(new_df.head())
        encoders = st.session_state.get('encoders', None)
        if encoders is None and not df_raw.empty:
            _, encoders = label_encode_df(df_raw)
            st.info("No encoders from training — fitted on sample dataset.")
        if encoders is None:
            st.error("No encoders available. Upload data with numeric/coded categorical or run Models tab first.")
        else:
            new_enc = new_df.copy()
            new_enc = encode_single_df_with_encoders(new_enc, encoders)
            models = st.session_state.get('trained_models', None)
            if models is None:
                df_enc_sample, encs = label_encode_df(df_raw)
                Xs = df_enc_sample.drop(columns=['Attrition'])
                ys = df_enc_sample['Attrition']
                rf = RandomForestClassifier(n_estimators=200, random_state=42)
                rf.fit(Xs, ys)
                models = {"Random Forest": rf}
                st.session_state['trained_models'] = models
                st.session_state['encoders'] = encs
            model_names = list(models.keys())
            sel_model = st.selectbox("Choose model for prediction", options=model_names)
            model = models[sel_model]
            X_new = new_enc.reindex(columns=model.feature_names_in_, fill_value=0)
            preds = model.predict(X_new)
            if 'Attrition' in encoders:
                le_attr = encoders['Attrition']
                try:
                    pred_labels = le_attr.inverse_transform(preds.astype(int))
                except Exception:
                    pred_labels = preds.astype(str)
            else:
                pred_labels = preds.astype(str)
            new_df['Predicted_Attrition'] = pred_labels
            st.write("Predictions added:")
            st.dataframe(new_df.head())
            b = BytesIO()
            new_df.to_csv(b, index=False)
            b.seek(0)
            st.download_button("Download predictions CSV", data=b, file_name="predicted_with_attrition.csv", mime="text/csv")

with tabs[3]:
    st.header("About / README")
    st.markdown("""
    This app contains:
    - Dashboard (5 charts + filters)
    - Models tab (train with cv=5)
    - Predict tab (upload & predict)
    Deploy on Streamlit Cloud by adding this repo to GitHub and setting main file to app.py.
    """)
