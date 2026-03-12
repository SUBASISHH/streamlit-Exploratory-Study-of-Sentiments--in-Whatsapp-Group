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
import emoji

# ============================================================
# GLOBAL SETTINGS — Folder paths and model directory config
# ============================================================
MEDIA_FOLDER = "Media"
MODEL_DIR = 'bert_emotion_model'


# ============================================================
# DATA LOADING — Parses WhatsApp export ZIP into a DataFrame.
#                Skips system messages using pattern matching.
# ============================================================
def load_data_from_zip(zip_file):
    with zipfile.ZipFile(zip_file) as z:
        filenames = z.namelist()
        chat_file = next((f for f in filenames if f.endswith('.txt')), None)
        if not chat_file:
            st.error("No .txt file found in the ZIP!")
            return None
        with z.open(chat_file) as f:
            content = io.TextIOWrapper(f, encoding='utf-8')
            pattern = r'^\[?(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?\s?[APMapm]{0,2})\]?\s(?:-\s)?(?:~\s)?(.*?):\s(.*)$'
            system_patterns = r'<Media omitted>|end-to-end encrypted|Messages to this group|was added|left|changed the subject|changed this group|security code|created group|added you'
            chat_data = []
            for line in content:
                line = line.strip()
                if not line:
                    continue
                match = re.match(pattern, line)
                if match:
                    groups = match.groups()
                    if not re.search(system_patterns, groups[3], re.IGNORECASE):
                        chat_data.append(groups)
                elif chat_data:
                    last_list = list(chat_data[-1])
                    last_list[3] += " " + line
                    chat_data[-1] = tuple(last_list)

    df = pd.DataFrame(chat_data, columns=['Date', 'Time', 'User', 'Message'])
    df['Has_Media'] = df['Message'].str.contains(r'<Media omitted>|attached|sticker', case=False)
    return df


# ============================================================
# EMOTION LABEL MAP — Maps emotion string labels to indices
# ============================================================
@st.cache_data
def prepare_datasets():
    emotion_map = {'Depressed': 0, 'Sad': 1, 'Neutral': 2, 'Happy': 3, 'Excitement': 4}
    return emotion_map


# ============================================================
# MODEL LOADING — Loads trained BERT model and tokenizer.
#                 Cached with @st.cache_resource so it only
#                 loads once across reruns.
# ============================================================
@st.cache_resource
def get_loaded_model():
    if os.path.exists(MODEL_DIR):
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
        return model, tokenizer
    else:
        st.sidebar.error("❌ BERT model folder not found! Please run train_bert.py first.")
        st.stop()


# ============================================================
# BERT INFERENCE — Runs batch emotion prediction on all
#                  messages. Uses adaptive batch size based
#                  on whether GPU or CPU is available.
# ============================================================
def run_bert_inference(df):
    emotion_map = prepare_datasets()
    model, tokenizer = get_loaded_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    batch_size = 64 if device.type == 'cuda' else 32

    messages = df['Message'].astype(str).tolist()
    all_predictions = []
    prog_bar = st.progress(0, text="Analyzing emotions...")

    for i in range(0, len(messages), batch_size):
        batch_msgs = messages[i: i + batch_size]
        inputs = tokenizer(
            batch_msgs, return_tensors="pt",
            truncation=True, max_length=64, padding='max_length'
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_predictions.extend(batch_preds)
        prog_bar.progress(min((i + batch_size) / len(messages), 1.0), text="Analyzing emotions...")

    prog_bar.empty()
    df['Emotion_Idx'] = all_predictions
    idx_to_emotion = {v: k for k, v in emotion_map.items()}
    df['Detected_Emotion'] = df['Emotion_Idx'].map(idx_to_emotion)
    return df


# ============================================================
# MEDIA TIMELINE — Matches image/sticker files to their
#                  surrounding emotional context using a
#                  sliding window over nearby messages.
# ============================================================
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
                        'Path': full_path,
                        'Timestamp': f"{df.iloc[i]['Date']} {df.iloc[i]['Time']}",
                        'User': df.iloc[i]['User'],
                        'Refined_Emotion': refined_vibe,
                        'Context': context_slice[['User', 'Message']]
                    }
                    if filename.lower().endswith('.webp'):
                        sticker_data.append(entry)
                    else:
                        photo_data.append(entry)
    return photo_data, sticker_data


# ============================================================
# EMOTION COLOR MAP — Emoji badges for each emotion label
# ============================================================
EMOTION_COLORS = {
    'Happy': '🟢',
    'Excitement': '🟡',
    'Neutral': '🔵',
    'Sad': '🟠',
    'Depressed': '🔴',
}


# ============================================================
# UI CONFIG — Page layout, title, and custom CSS styling
# ============================================================
st.set_page_config(layout="wide", page_title="WhatsApp Analyzer", page_icon="💬")

st.markdown("""
<style>
    .stMetric { background: #1e1e2e; border-radius: 12px; padding: 12px; }
    .media-vibe-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    .section-header {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR — File uploader, cache clear button, user filter
# ============================================================
st.sidebar.title("💬 WhatsApp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Upload your WhatsApp Export (ZIP)", type="zip")

if st.sidebar.button("🗑️ Clear Cache & Media"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    if os.path.exists(MEDIA_FOLDER):
        shutil.rmtree(MEDIA_FOLDER)
        os.makedirs(MEDIA_FOLDER)
    st.sidebar.success("Cache and media cleared!")
    st.rerun()

user_filter = None


# ============================================================
# MAIN APP LOGIC — Runs when a ZIP file is uploaded
# ============================================================
if uploaded_file is not None:

    if not os.path.exists(MEDIA_FOLDER):
        os.makedirs(MEDIA_FOLDER)

    # --------------------------------------------------------
    # STEP 1: MEDIA EXTRACTION — Unzips images and stickers
    #         into the local Media folder (runs only once)
    # --------------------------------------------------------
    if 'media_extracted' not in st.session_state:
        with zipfile.ZipFile(uploaded_file, 'r') as z:
            for file_info in z.infolist():
                if file_info.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    filename = os.path.basename(file_info.filename)
                    if filename:
                        out_path = os.path.join(MEDIA_FOLDER, filename)
                        with z.open(file_info) as source, open(out_path, "wb") as target:
                            shutil.copyfileobj(source, target)
        st.session_state['media_extracted'] = True

    # --------------------------------------------------------
    # STEP 2: DATA LOADING — Parses chat .txt into DataFrame
    #         and caches it in session_state
    # --------------------------------------------------------
    if 'df_raw' not in st.session_state:
        uploaded_file.seek(0)
        df_raw = load_data_from_zip(uploaded_file)
        if df_raw is None:
            st.stop()
        st.session_state['df_raw'] = df_raw

    df_raw = st.session_state['df_raw']

    # --------------------------------------------------------
    # STEP 3: BERT EMOTION INFERENCE — Predicts emotion for
    #         every message in batches. Cached in session_state
    #         so it only runs once per uploaded file.
    # --------------------------------------------------------
    if 'df_analyzed' not in st.session_state:
        with st.status("🧠 Analyzing Emotions with BERT...", expanded=True) as status:
            df_analyzed = run_bert_inference(df_raw.copy())
            st.session_state['df_analyzed'] = df_analyzed
            status.update(label="✅ BERT Analysis Complete!", state="complete")

    df = st.session_state['df_analyzed']

    # --------------------------------------------------------
    # SIDEBAR USER FILTER — Multi-select to scope dashboard
    #                        to specific participants
    # --------------------------------------------------------
    all_users = sorted(df['User'].unique().tolist())
    selected_users = st.sidebar.multiselect(
        "👤 Filter by User",
        options=all_users,
        default=all_users,
        help="Select users to include in the dashboard"
    )
    if selected_users:
        df = df[df['User'].isin(selected_users)]

    # --------------------------------------------------------
    # PRECOMPUTE SHARED METRICS — Date, hour, emoji, and
    #                              user count data used across
    #                              multiple dashboard sections
    # --------------------------------------------------------
    df['Date_dt'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df['Hour'] = pd.to_datetime(df['Time'].str.strip(), format='mixed', errors='coerce').dt.hour
    df['Day_of_Week'] = df['Date_dt'].dt.day_name()

    all_emojis = [c for msg in df['Message'] for c in str(msg) if emoji.is_emoji(c)]

    user_counts = df['User'].value_counts().reset_index()
    user_counts.columns = ['User', 'Message_Count']
    total_msgs = len(df)
    user_counts['Share_Percent'] = (user_counts['Message_Count'] / total_msgs) * 100

    positive_count = df['Detected_Emotion'].isin(['Happy', 'Excitement']).sum()
    vibe_score = round((positive_count / len(df)) * 100, 1) if len(df) > 0 else 0

    # --------------------------------------------------------
    # DASHBOARD HEADER + TOP METRIC CARDS
    # --------------------------------------------------------
    st.header("💬 WhatsApp Chat Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📨 Total Messages", f"{len(df):,}")
    col2.metric("👥 Active Users", f"{df['User'].nunique()}")
    col3.metric("😄 Emoji Usage", f"{len(all_emojis):,}", "Total Emojis")
    col4.metric("✨ Vibe Score", f"{vibe_score}%", f"+ve emotion ratio")

    st.markdown("---")

    # --------------------------------------------------------
    # ROW 2: Emoji frequency table (left) +
    #         Emotion trend line chart over time (right)
    # --------------------------------------------------------
    row2_left, row2_right = st.columns([1, 2])

    with row2_left:
        st.subheader("🔥 Top Emojis Used")
        if all_emojis:
            emoji_df = pd.Series(all_emojis).value_counts().head(10).reset_index()
            emoji_df.columns = ['Emoji', 'Count']
            st.dataframe(emoji_df, use_container_width=True, hide_index=True)
        else:
            st.info("No emojis found in chat.")

    with row2_right:
        st.subheader("📈 Emotion Trend Over Time")
        emotion_trend = df.groupby(['Date_dt', 'Detected_Emotion']).size().unstack(fill_value=0)
        st.line_chart(emotion_trend, use_container_width=True)

    # --------------------------------------------------------
    # USER ACTIVITY — Bar chart of message count per user
    # --------------------------------------------------------
    st.subheader("📊 Messages per User")
    st.bar_chart(user_counts.set_index('User').head(15)['Message_Count'], color="#2ecc71")

    st.divider()

    # --------------------------------------------------------
    # EMOTION RESULTS TABLE — Full DataFrame preview with
    #                          BERT-predicted emotion labels
    # --------------------------------------------------------
    st.subheader("🔍 Emotion Analysis Results")
    with st.expander("View Full Table", expanded=False):
        st.dataframe(df[['User', 'Message', 'Detected_Emotion']], use_container_width=True)

    # --------------------------------------------------------
    # DOWNLOAD — Export analyzed results as a CSV file
    # --------------------------------------------------------
    csv_data = df[['User', 'Message', 'Detected_Emotion']].to_csv(index=False)
    st.download_button(
        label="⬇️ Download Analysis as CSV",
        data=csv_data,
        file_name="whatsapp_analysis.csv",
        mime="text/csv"
    )

    st.divider()

    # --------------------------------------------------------
    # MEDIA TIMELINE — Photos and stickers matched to their
    #                   emotional context via sliding window.
    #                   Hidden behind a toggle to avoid
    #                   loading all images on every rerun.
    # --------------------------------------------------------
    st.header("🖼️ Media Timeline")

    if 'show_media_vibes' not in st.session_state:
        st.session_state['show_media_vibes'] = False

    btn_label = "🔍 Reveal Vibes for Media" if not st.session_state['show_media_vibes'] else "🙈 Hide Media Vibes"
    if st.button(btn_label, type="primary", use_container_width=False):
        st.session_state['show_media_vibes'] = not st.session_state['show_media_vibes']

    if st.session_state['show_media_vibes']:
        if 'photo_list' not in st.session_state or 'sticker_list' not in st.session_state:
            with st.spinner("🔎 Scanning media files and assigning vibes..."):
                photo_list, sticker_list = get_refined_visual_timelines(df, MEDIA_FOLDER)
                st.session_state['photo_list'] = photo_list
                st.session_state['sticker_list'] = sticker_list
        else:
            photo_list = st.session_state['photo_list']
            sticker_list = st.session_state['sticker_list']

        if photo_list or sticker_list:
            tab1, tab2 = st.tabs([f"📷 Photos ({len(photo_list)})", f"🎭 Stickers ({len(sticker_list)})"])

            with tab1:
                if photo_list:
                    for item in photo_list:
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.image(item['Path'], use_container_width=True)
                        with c2:
                            emotion = item['Refined_Emotion']
                            icon = EMOTION_COLORS.get(emotion, '⚪')
                            st.markdown(f"**Vibe:** {icon} `{emotion}`")
                            st.caption(f"👤 {item['User']}  |  🕐 {item['Timestamp']}")
                            with st.expander("💬 Show Context Messages"):
                                st.table(item['Context'])
                        st.divider()
                else:
                    st.info("No photo files found in the media folder matching chat messages.")

            with tab2:
                if sticker_list:
                    cols = st.columns(4)
                    for idx, item in enumerate(sticker_list):
                        with cols[idx % 4]:
                            st.image(item['Path'], use_container_width=True)
                            emotion = item['Refined_Emotion']
                            icon = EMOTION_COLORS.get(emotion, '⚪')
                            st.caption(f"{icon} {emotion}")
                else:
                    st.info("No sticker files found.")
        else:
            st.warning("⚠️ No media found. Make sure your ZIP includes image files and the chat references them.")
    else:
        st.info("Click **Reveal Vibes for Media** above to analyze the emotional context of photos and stickers.")

    st.divider()

    # --------------------------------------------------------
    # GROUP SEGMENTATION — Classifies users into participation
    #                       tiers based on message share %
    # --------------------------------------------------------
    st.header("👥 Group Member Segmentation")

    def custom_segmentation(p):
        if p >= 30.0: return 'Core Contributor (>30%)'
        if p >= 15.0: return 'Active Participant (15-30%)'
        if p >= 2.0:  return 'Occasional Contributor (2-15%)'
        return 'Lurker/Observer (<2%)'

    user_counts['Participation_Level'] = user_counts['Share_Percent'].apply(custom_segmentation)
    summary = user_counts['Participation_Level'].value_counts()
    summary = summary.reindex([
        'Core Contributor (>30%)', 'Active Participant (15-30%)',
        'Occasional Contributor (2-15%)', 'Lurker/Observer (<2%)'
    ]).fillna(0)

    c1, c2 = st.columns([1, 1])
    with c1:
        fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
        ax_pie.pie(
            summary,
            labels=summary.index,
            autopct=lambda p: '{:.0f}'.format(p * sum(summary) / 100) if p > 0 else '',
            startangle=140,
            colors=sns.color_palette('rocket', 4),
            explode=[0.05] * 4
        )
        st.pyplot(fig_pie)
    with c2:
        st.dataframe(
            user_counts[['User', 'Participation_Level', 'Share_Percent']].sort_values(
                by='Share_Percent', ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # HEATMAP — Message frequency by day-of-week and hour.
    #            Also surfaces peak day and peak hour metrics.
    # --------------------------------------------------------
    st.header("🕒 Group Activity Patterns")

    heatmap_data = df.groupby(['Day_of_Week', 'Hour']).size().unstack(fill_value=0)
    for h in range(24):
        if h not in heatmap_data.columns:
            heatmap_data[h] = 0
    heatmap_data = heatmap_data[sorted(heatmap_data.columns)].reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ).fillna(0)

    peak_hour = int(heatmap_data.sum().idxmax())
    peak_day = heatmap_data.sum(axis=1).idxmax()
    h1, h2 = st.columns(2)
    h1.metric("🏆 Most Active Day", peak_day)
    h2.metric("⏰ Peak Hour", f"{peak_hour}:00 – {peak_hour + 1}:00")

    fig_heat, ax_heat = plt.subplots(figsize=(20, 8))
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d', ax=ax_heat)
    ax_heat.set_title("Messages by Day & Hour", fontsize=16)
    st.pyplot(fig_heat)

    # --------------------------------------------------------
    # USER-EMOTION MATRIX — Crosstab heatmap showing emotion
    #                        distribution per user. Right panel
    #                        highlights each user's top emotion.
    # --------------------------------------------------------
    st.header("🎯 User–Emotion Correlation Matrix")
    top_users = user_counts['User'].head(10).tolist()
    user_vibe_matrix = pd.crosstab(df['User'], df['Detected_Emotion']).reindex(
        index=[u for u in top_users if u in df['User'].unique()]
    ).fillna(0)

    c_m1, c_m2 = st.columns([2, 1])
    with c_m1:
        fig_matrix, ax_matrix = plt.subplots(figsize=(12, 8))
        sns.heatmap(user_vibe_matrix, annot=True, fmt='.0f', cmap='RdPu', ax=ax_matrix)
        ax_matrix.set_title("User vs Emotion Heatmap", fontsize=14)
        st.pyplot(fig_matrix)
    with c_m2:
        st.subheader("Dominant Emotion per User")
        for user in user_vibe_matrix.index:
            row = user_vibe_matrix.loc[user]
            if row.sum() > 0:
                top_emo = row.idxmax()
                icon = EMOTION_COLORS.get(top_emo, '⚪')
                st.write(f"**{user}**")
                st.success(f"{icon} {top_emo}")

else:
    st.markdown("## 💬 WhatsApp Chat Analyzer")
    st.info("👈 Upload a WhatsApp export ZIP file in the sidebar to begin.")
    st.markdown("""
    **How to export your WhatsApp chat:**
    1. Open any WhatsApp chat or group
    2. Tap **⋮ Menu → More → Export chat**
    3. Choose **Include Media** (optional)
    4. Save or share the `.zip` file
    5. Upload it here!
    """)