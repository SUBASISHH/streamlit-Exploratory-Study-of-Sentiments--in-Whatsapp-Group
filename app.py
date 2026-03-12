import streamlit as st
import pandas as pd
import re
import zipfile
import io
import os
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
from transformers import BertTokenizer, BertForSequenceClassification 
import emoji  # Make sure to run: pip install emoji

# --- GLOBAL SETTINGS ---
MEDIA_FOLDER = "Media"
MODEL_DIR = 'bert_emotion_model' 

if not os.path.exists(MEDIA_FOLDER):
    os.makedirs(MEDIA_FOLDER)

# --- FUNCTION DEFINITIONS ---

def load_data_from_zip(zip_file):
    with zipfile.ZipFile(zip_file) as z:
        filenames = z.namelist()
        chat_file = next((f for f in filenames if f.endswith('.txt')), None)
        if not chat_file:
            st.error("No .txt file found in the ZIP!")
            return None
        with z.open(chat_file) as f:
            content = io.TextIOWrapper(f, encoding='utf-8')
            pattern = r'^\[?(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?\s?[APMapm]?)\]?\s(?:-\s)?(.*?):\s(.*)$'
            chat_data = []
            for line in content:
                line = line.strip()
                if not line: continue 
                match = re.match(pattern, line)
                if match:
                    chat_data.append(match.groups())
                elif chat_data:
                    last_list = list(chat_data[-1])
                    last_list[3] += " " + line
                    chat_data[-1] = tuple(last_list)
    df = pd.DataFrame(chat_data, columns=['Date', 'Time', 'User', 'Message'])
    df['Has_Media'] = df['Message'].str.contains(r'<Media omitted>|attached|sticker', case=False)
    return df

@st.cache_data
def prepare_datasets():
    mapping_df = pd.read_csv('emotion_dataset.csv')
    emotion_map = {'Depressed': 0, 'Sad': 1, 'Neutral': 2, 'Happy': 3, 'Excitement': 4}
    return emotion_map

def get_loaded_model():
    if os.path.exists(MODEL_DIR):
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()  
        return model, tokenizer
    else:
        st.sidebar.error("❌ BERT model folder not found! Please run train_bert.py first.")
        st.stop()

def get_refined_visual_timelines(df, folder_path, window=2):
    photo_data = []
    sticker_data = []
    for i in range(len(df)):
        if df.iloc[i]['Has_Media']:
            msg = df.iloc[i]['Message']
            file_match = re.search(r'([\w-]+\.(?:jpg|jpeg|png|heic|webp))', msg, re.IGNORECASE)
            if file_match:
                filename = file_match.group(1)
                full_path = os.path.join(folder_path, filename)
                if os.path.exists(full_path):
                    start = max(0, i - window)
                    end = min(len(df), i + window + 1)
                    context_slice = df.iloc[start:end]
                    refined_vibe = context_slice['Detected_Emotion'].mode()[0]
                    entry = {
                        'Path': full_path, 'Timestamp': f"{df.iloc[i]['Date']} {df.iloc[i]['Time']}",
                        'User': df.iloc[i]['User'], 'Refined_Emotion': refined_vibe,
                        'Context': context_slice[['User', 'Message']]
                    }
                    if filename.lower().endswith('.webp'): sticker_data.append(entry)
                    else: photo_data.append(entry)
    return photo_data, sticker_data

# --- UI CONFIG ---
st.set_page_config(layout="wide")
st.sidebar.title("WhatsApp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Upload your WhatsApp Export (ZIP)", type="zip")

if st.sidebar.button("🗑️ Clear Local Media Folder"):
    if os.path.exists(MEDIA_FOLDER):
        shutil.rmtree(MEDIA_FOLDER)
        os.makedirs(MEDIA_FOLDER)
        st.sidebar.success("Media folder cleared!")
        st.rerun()

# --- MAIN APP LOGIC ---
if uploaded_file is not None:
    # 1. Extract Media
    with zipfile.ZipFile(uploaded_file, 'r') as z:
        for file_info in z.infolist():
            if file_info.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                filename = os.path.basename(file_info.filename)
                if filename:
                    with z.open(file_info) as source, open(os.path.join(MEDIA_FOLDER, filename), "wb") as target:
                        shutil.copyfileobj(source, target)
    
    # 2. Load Data
    uploaded_file.seek(0)
    df = load_data_from_zip(uploaded_file)
    
    if df is not None:
        # 3. Predict Emotions with BERT (FAST BATCH LOGIC - logic preserved)
        with st.status("🧠 Analyzing Emotions with BERT...") as status:
            emotion_map = prepare_datasets()
            model, tokenizer = get_loaded_model()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            messages = df['Message'].astype(str).tolist()
            all_predictions = []
            batch_size = 16 
            prog_bar = st.progress(0)
            for i in range(0, len(messages), batch_size):
                batch_msgs = messages[i : i + batch_size]
                inputs = tokenizer(batch_msgs, return_tensors="pt", truncation=True, max_length=64, padding='max_length').to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    all_predictions.extend(batch_preds)
                prog_bar.progress(min((i + batch_size) / len(messages), 1.0))
            df['Emotion_Idx'] = all_predictions
            idx_to_emotion = {v: k for k, v in emotion_map.items()}
            df['Detected_Emotion'] = df['Emotion_Idx'].map(idx_to_emotion)
            prog_bar.empty()
            status.update(label="BERT Analysis Complete!", state="complete")

        # --- DASHBOARD TOP ROW (CARDS) ---
        st.header("HOME > DASHBOARD")
        col1, col2, col3, col4 = st.columns(4)
        
        # Total Messages
        col1.metric("TOTAL TRAFFIC", f"{len(df)}", "+5% messages")
        # Active Users
        distinct_users = df['User'].nunique()
        col2.metric("ACTIVE USERS", f"{distinct_users}", f"Out of {len(df)} msgs")
        # Emoji Count
        all_emojis = [c for msg in df['Message'] for c in str(msg) if emoji.is_emoji(c)]
        col3.metric("EMOJI USAGE", f"{len(all_emojis)}", "Total Emojis")
        # Performance/Health (Mock sentiment score)
        col4.metric("VIBE SCORE", "60%", "+2.5% Sentiment")

        st.markdown("---")

        # --- SECOND ROW (LEFT: STATUS, RIGHT: TRAJECTORY) ---
        row2_left, row2_right = st.columns([1, 2])
        
        with row2_left:
            st.subheader("Frequently Used Emojis")
            if all_emojis:
                emoji_df = pd.Series(all_emojis).value_counts().head(10).reset_index()
                emoji_df.columns = ['Emoji', 'Count']
                st.dataframe(emoji_df, use_container_width=True, hide_index=True)
            else:
                st.info("No emojis found in chat.")
            
            st.info("🌦️ WEATHER UPDATES: Clear Sky")
            st.success("🏥 HEALTH CARE: Sentiment Stable")

        with row2_right:
            st.subheader("PERCENTAGE TRAJECTORY")
            df['Date_dt'] = pd.to_datetime(df['Date'], dayfirst=True)
            emotion_trend = df.groupby(['Date_dt', 'Detected_Emotion']).size().unstack(fill_value=0)
            st.line_chart(emotion_trend, use_container_width=True)

        # --- THIRD ROW (ACTIVITY BAR CHART) ---
        st.subheader("TOTAL ACTIVITY per USER")
        user_counts = df['User'].value_counts().reset_index()
        user_counts.columns = ['User', 'Message_Count']
        st.bar_chart(user_counts.set_index('User').head(15), color="#2ecc71")

        # --- ORIGINAL ANALYSIS FEATURES (BELOW DASHBOARD) ---
        st.divider()
        
        # Emotion Analysis Data Preview
        st.subheader("Emotion Analysis Results")
        st.dataframe(df[['User', 'Message', 'Detected_Emotion']], use_container_width=True)

        # Media Timeline
        photo_list, sticker_list = get_refined_visual_timelines(df, MEDIA_FOLDER)
        if photo_list or sticker_list:
            st.header("🖼️ Media Timeline")
            tab1, tab2 = st.tabs([f"Photos ({len(photo_list)})", f"Stickers ({len(sticker_list)})"])
            with tab1:
                for item in photo_list:
                    c1, c2 = st.columns([1, 2])
                    with c1: st.image(item['Path'], use_container_width=True)
                    with c2:
                        st.markdown(f"**Vibe:** `{item['Refined_Emotion']}`")
                        st.caption(f"By {item['User']} | {item['Timestamp']}")
                        with st.expander("Show Context"): st.table(item['Context'])
                    st.divider()
            with tab2:
                cols = st.columns(4)
                for idx, item in enumerate(sticker_list):
                    with cols[idx % 4]: st.image(item['Path'], caption=f"Mood: {item['Refined_Emotion']}")

        # Segmentation Logic (preserved)
        st.header("👥 Group Member Segmentation")
        total_msgs = len(df)
        def custom_segmentation(p):
            if p >= 30.0: return 'Core Contributor (>30%)'
            if p >= 15.0: return 'Active Participant (15-30%)'
            if p >= 2.0:  return 'Occasional Contributor (2-15%)'
            return 'Lurker/Observer (<2%)'
        
        user_counts['Share_Percent'] = (user_counts['Message_Count'] / total_msgs) * 100
        user_counts['Participation_Level'] = user_counts['Share_Percent'].apply(custom_segmentation)
        summary = user_counts['Participation_Level'].value_counts()
        summary = summary.reindex(['Core Contributor (>30%)', 'Active Participant (15-30%)', 'Occasional Contributor (2-15%)', 'Lurker/Observer (<2%)']).fillna(0)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
            ax_pie.pie(summary, labels=summary.index, autopct=lambda p: '{:.0f}'.format(p * sum(summary)/100) if p > 0 else '', startangle=140, colors=sns.color_palette('rocket', 4), explode=[0.05]*4)
            st.pyplot(fig_pie)
        with c2: st.dataframe(user_counts[['User', 'Participation_Level', 'Share_Percent']].sort_values(by='Share_Percent', ascending=False), use_container_width=True, hide_index=True)

        # Heatmap (preserved)
        st.header("🕒 Group Activity Patterns")
        df['Day_of_Week'] = df['Date_dt'].dt.day_name()
        df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
        heatmap_data = df.groupby(['Day_of_Week', 'Hour']).size().unstack(fill_value=0)
        for h in range(24): 
            if h not in heatmap_data.columns: heatmap_data[h] = 0
        heatmap_data = heatmap_data[sorted(heatmap_data.columns)].reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        fig_heat, ax_heat = plt.subplots(figsize=(20, 8))
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d', ax=ax_heat)
        st.pyplot(fig_heat)

        # User-Emotion Matrix (preserved)
        st.header("🎯 User-Emotion Correlation Matrix")
        user_vibe_matrix = pd.crosstab(df['User'], df['Detected_Emotion']).loc[user_counts['User'].head(10)]
        c_m1, c_m2 = st.columns([2, 1])
        with c_m1:
            fig_matrix, ax_matrix = plt.subplots(figsize=(12, 8))
            sns.heatmap(user_vibe_matrix, annot=True, fmt='d', cmap='RdPu', ax=ax_matrix)
            st.pyplot(fig_matrix)
        with c_m2:
            for user in user_vibe_matrix.index:
                top_emo = user_vibe_matrix.loc[user].idxmax()
                st.write(f"**{user}**")
                st.success(f"{top_emo}")

else:
    st.info("Please upload a WhatsApp export ZIP file in the sidebar to begin.")