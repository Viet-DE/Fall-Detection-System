'''

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import albumentations as A
from pathlib import Path
import tempfile
from collections import deque
from ultralytics import YOLO
import time
from datetime import datetime
import os

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Hệ thống Dự đoán Fall",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# Custom CSS - Theo thiết kế trong ảnh
# ============================================================
st.markdown("""
    <style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&display=swap');
    
    /* Global */
    * {
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Main container background */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #ffffff;
    }
    
    /* Header gradient */
    .header-gradient {
        background: linear-gradient(90deg, #4158D0 0%, #C850C0 46%, #FFCC70 100%);
        padding: 2rem 0;
        text-align: center;
        margin: -1rem -1rem 2rem -1rem;
    }
    
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #1a1a2e;
        padding: 1rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d2d44;
        color: #ffffff;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        border: none;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4158D0 0%, #C850C0 100%);
        color: white;
    }
    
    /* Section headers */
    h2, h3 {
        color: #ffffff !important;
    }
    
    /* Cards/Containers */
    .upload-container {
        background: #2d2d44;
        border: 2px dashed #4158D0;
        border-radius: 15px;
        padding: 3rem 2rem;
        text-align: center;
        min-height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .result-container {
        background: #2d2d44;
        border-radius: 15px;
        padding: 2rem;
        min-height: 300px;
    }
    
    .status-safe {
        background: linear-gradient(135deg, #0f5132 0%, #198754 100%);
        color: white;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .status-danger {
        background: linear-gradient(135deg, #842029 0%, #dc3545 100%);
        color: white;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .stats-box {
        background: #1a1a2e;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #4158D0 0%, #C850C0 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(65, 88, 208, 0.4);
    }
    
    /* Camera placeholder */
    .camera-placeholder {
        background: #1a1a2e;
        border: 2px solid #2d2d44;
        border-radius: 10px;
        padding: 3rem;
        text-align: center;
        color: #888;
        min-height: 400px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    /* History section */
    .history-container {
        background: #2d2d44;
        border-radius: 15px;
        padding: 2rem;
        margin-top: 2rem;
    }
    
    /* File uploader - chỉ ẩn khi chưa active */
    .stFileUploader {
        background: transparent;
    }
    
    /* Ẩn text "Drag and drop" */
    [data-testid="stFileUploader"] section {
        background: #2d2d44;
        border-radius: 10px;
        padding: 1rem;
    }
    
    [data-testid="stFileUploader"] section > div:first-child {
        display: none;
    }
    
    /* Style browse button */
    [data-testid="stFileUploader"] button {
        background: linear-gradient(90deg, #4158D0 0%, #C850C0 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #2d2d44;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# Model Definition
# ============================================================
class EfficientNetLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, dropout=0.5):
        super(EfficientNetLSTM, self).__init__()
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        self.efficientnet = efficientnet_v2_s(weights=weights)
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(256*2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        x = x.view(batch_size * num_frames, c, h, w)
        features = self.efficientnet(x)
        features = features.view(batch_size, num_frames, -1)
        lstm_out, _ = self.lstm(features)
        final_features = lstm_out[:, -1, :]
        output = self.fc(final_features)
        return output.squeeze()

# ============================================================
# Helper Functions
# ============================================================
@st.cache_resource
def load_model(model_path):
    """Load the fall detection model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetLSTM(hidden_size=256, num_layers=2, dropout=0.5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

@st.cache_resource
def load_yolo_model():
    """Load YOLO model for person detection"""
    return YOLO("yolov8n.pt")

def get_transform():
    """Get image transformation pipeline"""
    return A.Compose([
        A.Resize(height=288, width=288),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToTensorV2(),
    ])

def extract_frames_from_video(video_path, num_frames=16):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    if total_frames >= num_frames:
        index_frames = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        index_frame = np.arange(0, total_frames)
        index_frame_lack = np.full(num_frames - total_frames, index_frame[-1])
        index_frames = np.concatenate((index_frame, index_frame_lack))
    
    for index_frame in index_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index_frame)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return np.array(frames)

def detect_person_bbox(frame_bgr, yolo_model, conf_thres=0.5):
    """Detect person bounding box using YOLO"""
    res = yolo_model(frame_bgr, verbose=False, conf=conf_thres)
    boxes = res[0].boxes.data.cpu().numpy()
    
    for x1, y1, x2, y2, cf, cl in boxes:
        if int(cl) == 0:  # person class
            if x2 > x1 and y2 > y1:
                return (int(x1), int(y1), int(x2), int(y2))
    return None

def frames_to_tensor(frame_rgb, transform):
    """Convert frame to tensor"""
    frame_tensor = transform(image=frame_rgb)['image']
    return frame_tensor

def predict_video(model, frames, transform, device, threshold=0.5):
    """Predict if video contains a fall"""
    video_tensor = transform(images=frames)['images']
    video_tensor = video_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(video_tensor)
        probability = torch.sigmoid(output).item()
    
    prediction = 'Fall' if probability > threshold else 'No Fall'
    confidence = probability if probability > 0.5 else (1 - probability)
    
    return prediction, confidence, probability

# ============================================================
# Main App
# ============================================================
def main():
    # Initialize session state variables
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'detection_log' not in st.session_state:
        st.session_state.detection_log = []
    if 'fall_count' not in st.session_state:
        st.session_state.fall_count = 0
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'current_status' not in st.session_state:
        st.session_state.current_status = "An toàn"
    if 'fall_video_writer' not in st.session_state:
        st.session_state.fall_video_writer = None
    if 'fall_recording' not in st.session_state:
        st.session_state.fall_recording = False
    if 'fall_video_count' not in st.session_state:
        st.session_state.fall_video_count = 0
    if 'saved_fall_videos' not in st.session_state:
        st.session_state.saved_fall_videos = []
    if 'video_analysis_history' not in st.session_state:
        st.session_state.video_analysis_history = []
    if 'fall_start_time' not in st.session_state:
        st.session_state.fall_start_time = None
    if 'fall_end_time' not in st.session_state:
        st.session_state.fall_end_time = None
    if 'fall_frame_count' not in st.session_state:
        st.session_state.fall_frame_count = 0
    if 'current_video_filename' not in st.session_state:
        st.session_state.current_video_filename = None
    if 'video_analyzed' not in st.session_state:
        st.session_state.video_analyzed = False
    if 'video_analysis_result' not in st.session_state:
        st.session_state.video_analysis_result = None
    
    # Header với gradient giống ảnh
    st.markdown("""
        <div class="header-gradient">
            <h1 class="main-title">🎈 Hệ thống Dự đoán Fall</h1>
            <p class="subtitle">Công nghệ AI tiên tiến để phát hiện nguy cơ té ngã</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load models
    model_path = "best_model_efficientnet_lstm_v2.pth"
    num_frames = 32
    threshold = 0.5
    conf_thres = 0.5
    
    try:
        with st.spinner("Đang tải models..."):
            model, device = load_model(model_path)
            yolo_model = load_yolo_model()
            transform = get_transform()
    except Exception as e:
        st.error(f"Lỗi khi tải models: {str(e)}")
        return
    
    # Tabs chính
    tab1, tab2 = st.tabs(["📹 Dự đoán qua Video", "📹 Dự đoán Realtime"])
    
    # ============================================================
    # TAB 1: VIDEO PREDICTION
    # ============================================================
    with tab1:
        st.markdown("## 📹 Phân tích Video")
        
        col_left, col_right = st.columns([7, 3])
        
        with col_left:
            st.markdown("### Tải lên video")
            
            # Khởi tạo session state cho upload
            if 'uploaded_video' not in st.session_state:
                st.session_state.uploaded_video = None
            
            if st.session_state.uploaded_video is None:
                # Hiển thị placeholder và file uploader
                st.markdown("""
                    <div class="camera-placeholder">
                        <div style="font-size: 4rem;">📷</div>
                        <p style="font-size: 1.2rem; margin-top: 1rem;">Camera chưa được kích hoạt</p>
                    </div>
                """, unsafe_allow_html=True)
                uploaded_file = st.file_uploader(
                    "Chọn video để phân tích",
                    type=['mp4', 'avi', 'mov', 'mkv'],
                    label_visibility="collapsed",
                    key="video_uploader"
                )
                
                if uploaded_file is not None:
                    st.session_state.uploaded_video = uploaded_file
                    # Reset trạng thái phân tích khi tải video mới
                    st.session_state.video_analyzed = False
                    st.session_state.video_analysis_result = None
                    st.rerun()
                
                
            else:
                # Hiển thị video đã upload
                st.video(st.session_state.uploaded_video)
                st.caption(f"✓ Đã tải: {st.session_state.uploaded_video.name}")
                if st.button("🔄 Chọn video khác", width='stretch'):
                    st.session_state.uploaded_video = None
                    # Reset trạng thái phân tích
                    st.session_state.video_analyzed = False
                    st.session_state.video_analysis_result = None
                    st.rerun()
            
            uploaded_file = st.session_state.uploaded_video
        
        with col_right:
            st.markdown("### Kết quả phân tích")
            
            if uploaded_file is None:
                st.markdown("""
                    <div class="result-container">
                        <p style="text-align: center; color: #888; margin-top: 6rem;">
                            Chưa có video để phân tích
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.button("🔵 Bắt đầu phân tích", disabled=True, width='stretch')
            else:
                # Nếu chưa phân tích hoặc đổi video mới
                if not st.session_state.video_analyzed or st.session_state.video_analysis_result is None:
                    if st.button("🔵 Bắt đầu phân tích", width='stretch'):
                        # Save video
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        tfile.write(uploaded_file.read())
                        video_path = tfile.name
                        
                        with st.spinner("Đang phân tích video..."):
                            progress_bar = st.progress(0)
                            
                            # Extract frames
                            progress_bar.progress(30)
                            frames = extract_frames_from_video(video_path, num_frames)
                            
                            # Predict
                            progress_bar.progress(70)
                            prediction, confidence, probability = predict_video(
                                model, frames, transform, device, threshold
                            )
                            
                            progress_bar.progress(100)
                            time.sleep(0.3)
                            progress_bar.empty()
                        
                        # Lấy thông tin video
                        cap_info = cv2.VideoCapture(video_path)
                        total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = int(cap_info.get(cv2.CAP_PROP_FPS))
                        duration_sec = total_frames / fps if fps > 0 else 0
                        cap_info.release()
                        
                        # Lưu kết quả vào session state
                        st.session_state.video_analyzed = True
                        st.session_state.video_analysis_result = {
                            'prediction': prediction,
                            'confidence': confidence,
                            'probability': probability,
                            'video_name': uploaded_file.name,
                            'total_frames': total_frames,
                            'fps': fps,
                            'duration_sec': duration_sec
                        }
                        
                        # Lưu vào lịch sử
                        analysis_time = datetime.now().strftime('%d/%m/%Y - %H:%M')
                        st.session_state.video_analysis_history.insert(0, {
                            'time': analysis_time,
                            'video_name': uploaded_file.name,
                            'result': 'Fall detected' if prediction == 'Fall' else 'Safe',
                            'confidence': confidence,
                            'probability': probability,
                            'is_fall': prediction == 'Fall'
                        })
                        
                        st.rerun()
                else:
                    # Hiển thị thông tin chi tiết thay cho nút
                    result = st.session_state.video_analysis_result
                    
                    st.markdown("**📊 Thông tin chi tiết:**")
                    st.markdown(f"• **Tên video:** `{result['video_name']}`")
                    st.markdown(f"• **Thời lượng:** `{int(result['duration_sec']//60):02d}:{int(result['duration_sec']%60):02d}s`")
                    st.markdown(f"• **Tổng số khung hình:** `{result['total_frames']}`")
                    st.markdown(f"• **Tỷ lệ khung hình (FPS):** `{result['fps']}`")
                    
                    if result['prediction'] == 'Fall':
                        st.markdown(f"• **Kết quả tổng thể:** 🟥 **Phát hiện NGÃ**")
                    else:
                        st.markdown(f"• **Kết quả tổng thể:** 🟩 **Không phát hiện ngã**")
                    
                    st.markdown(f"• **Độ tin cậy (Confidence):** `{result['confidence']:.2f}`")
        
        # History section
        st.markdown("---")
        st.markdown("## 📊 Lịch sử cảnh báo")
        
        if len(st.session_state.video_analysis_history) > 0:
            # Header của bảng
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            with col1:
                st.markdown("**Thời gian phân tích**")
            with col2:
                st.markdown("**Tên video**")
            with col3:
                st.markdown("**Kết quả**")
            with col4:
                st.markdown("**Confidence**")
            
            st.markdown("---")
            
            # Hiển thị từng dòng lịch sử
            for entry in st.session_state.video_analysis_history:
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
                    st.text(entry['time'])
                
                with col2:
                    st.text(entry['video_name'])
                
                with col3:
                    if entry['is_fall']:
                        st.markdown("⚠️ **<span style='color: #ff4444;'>Fall detected</span>**", unsafe_allow_html=True)
                    else:
                        st.markdown("✅ **<span style='color: #44ff44;'>Safe</span>**", unsafe_allow_html=True)
                
                with col4:
                    st.text(f"{entry['confidence']:.2f}")
                
                st.markdown("---")
            
            # Nút xóa lịch sử
            if st.button("🗑️ Xóa lịch sử", key="clear_history"):
                st.session_state.video_analysis_history = []
                st.rerun()
        else:
            st.markdown("""
                <div class="history-container">
                    <p style="text-align: center; color: #888;">
                        📋 Chưa có cảnh báo nào được ghi nhận
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    # ============================================================
    # TAB 2: REALTIME DETECTION
    # ============================================================
    with tab2:
        st.markdown("## 📹 Giám sát Realtime")
        
        col_left, col_right = st.columns([7, 3])
        
        with col_left:
            # st.markdown("### Camera feed")
            frame_placeholder = st.empty()
            
            if not st.session_state.camera_active:
                frame_placeholder.markdown("""
                    <div class="camera-placeholder">
                        <div style="font-size: 4rem;">📷</div>
                        <p style="font-size: 1.2rem; margin-top: 1rem;">Camera chưa được kích hoạt</p>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("🎬 Bật Camera", width='stretch', type="primary"):
                    st.session_state.camera_active = True
                    st.session_state.start_time = time.time()
                    # st.session_state.fall_count = 0
                    # st.session_state.detection_log = []
                    st.rerun()
            else:
                if st.button("⏹️ Dừng giám sát", width='stretch', type="secondary"):
                    st.session_state.camera_active = False
                    st.session_state.end_time = time.time()  # Lưu thời gian kết thúc
                    st.rerun()
        
        with col_right:
            # Status box
            st.markdown("### Trạng thái hiện tại")
            status_container = st.empty()
            
            if st.session_state.current_status == "An toàn":
                status_container.markdown("""
                    <div class="status-safe">
                        <h4 style="margin: 0;">🟢 An toàn</h4>
                        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Không phát hiện nguy cơ té ngã</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                status_container.markdown("""
                    <div class="status-danger">
                        <h4 style="margin: 0;">🔴 Nguy hiểm</h4>
                        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Phát hiện nguy cơ té ngã</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### Nhật ký phát hiện")
            log_container = st.empty()
            
            # Thống kê (chỉ hiển thị khi camera TẮT)
            stats_container = st.empty()
        
        # History section placeholders
        st.markdown("---")
        st.markdown("## 📊 Video cảnh fall")
        
        video_history_placeholder = st.empty()
        
        # Realtime detection loop
        if st.session_state.camera_active:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                frame_placeholder.error("❌ Không thể mở webcam")
                st.session_state.camera_active = False
                st.rerun()
            else:
                buffer = deque(maxlen=num_frames)
                frames_add_video = deque(maxlen=30)
                smooth_frame = deque(maxlen = 5)
                output_dir = Path("fall_videos")
                output_dir.mkdir(exist_ok=True)
                no_detect_count = 0
                try:
                    while st.session_state.camera_active:
                        ret, frame_bgr = cap.read()
                        if not ret:
                            break
                        
                        frames_add_video.append(frame_bgr)
                        # Hiển thị thời gian ở góc dưới trái
                        h, w = frame_bgr.shape[:2]
                        current_time = datetime.now().strftime('%d/%m/%Y - %H:%M:%S')
                        cv2.putText(frame_bgr, current_time, (10, h - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        
                        # Thêm frame vào buffer để phân tích
                        frame_tensor = frames_to_tensor(frame_rgb, transform)
                        buffer.append(frame_tensor)
                        # Phát hiện người trong frame
                        bbox = detect_person_bbox(frame_bgr, yolo_model, conf_thres)
                        
                        # === TRƯỜNG HỢP 1: KHÔNG PHÁT HIỆN NGƯỜI ===
                        if bbox is None:
                            no_detect_count += 1
                            
                            if no_detect_count >= 10:
                                buffer.clear()
                                no_detect_count = 0
                                frames_add_video.clear()
                                st.session_state.smoothed_prob = 0.0
                                cv2.putText(frame_bgr, "Khong phat hien nguoi", 
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                                st.session_state.current_status = "An toàn"
                                
                                # Nếu đang ghi video fall -> dừng ghi và lưu
                                if st.session_state.fall_recording:
                                    st.session_state.fall_video_writer.release()
                                    st.session_state.fall_recording = False
                                    st.session_state.fall_video_writer = None

                                    time.sleep(0.1)
                                            
                                    video_path = Path(st.session_state.current_video_filename)
                                    if video_path.exists() and video_path.stat().st_size > 0:
                                        st.session_state.saved_fall_videos.append({
                                            'filename': st.session_state.current_video_filename,
                                            'start_time': st.session_state.fall_start_time,
                                            'end_time': st.session_state.fall_end_time,
                                            'frame_count': st.session_state.fall_frame_count
                                        })
                            else:
                                # # Chưa đủ 10 frames -> Vẫn predict nếu đủ frames trong buffer
                                # cv2.putText(frame_bgr, f"Mat nguoi ({no_detect_count}/10)", 
                                #         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                                
                                if len(buffer) == num_frames:
                                    # Chạy model
                                    video_tensor = torch.stack(list(buffer)).unsqueeze(0).to(device)
                                    with torch.no_grad():
                                        output = model(video_tensor)
                                        prob = torch.sigmoid(output).item()
                                    
                                    # prob_to_binary = 1 if prob > 0.5 else 0
                                    # smooth_frame.append(prob_to_binary)
                                    
                                    # count_fall_frame = sum(smooth_frame)
                                    # count_nofall_frame = len(smooth_frame) - count_fall_frame
                                    # # Xác định fall
                                    # is_fall = True if count_fall_frame > count_nofall_frame else False

                                    is_fall = True if prob > 0.5 else False
                                    label = "TE NGA!" if is_fall else "Binh thuong"
                                    color = (0, 0, 255) if is_fall else (0, 255, 0)
                                    
                                    # Cập nhật trạng thái
                                    st.session_state.current_status = "Nguy hiểm" if is_fall else "An toàn"
                                    
                                    # Vẽ label (không có bbox)
                                    cv2.putText(frame_bgr, f"{label} ({prob*100:.1f}%)", (10, 70),
                                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                                    
                                    # Xử lý ghi video fall
                                    if is_fall:
                                        # Bắt đầu ghi video (nếu chưa ghi)
                                        if not st.session_state.fall_recording:
                                            st.session_state.fall_video_count += 1
                                            video_filename = output_dir / f"fall_{st.session_state.fall_video_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                                            
                                            h, w = frame_bgr.shape[:2] 
                                            fourcc = cv2.VideoWriter_fourcc(*'avc1')
                                            writer = cv2.VideoWriter(str(video_filename), fourcc, 20.0, (w, h))
                                            
                                            st.session_state.fall_video_writer = writer
                                            st.session_state.fall_recording = True
                                            st.session_state.fall_frame_count = 0
                                            st.session_state.fall_start_time = current_time
                                            st.session_state.current_video_filename = str(video_filename)
                                            
                                            # Ghi buffer frames trước đó
                                            for old_frame in frames_add_video:
                                                st.session_state.fall_video_writer.write(old_frame)
                                        
                                        # Ghi frame hiện tại
                                        st.session_state.fall_video_writer.write(frame_bgr)
                                        st.session_state.fall_frame_count += 1
                                        
                                        # Tăng số lần fall (chỉ tính lần đầu)
                                        if len(st.session_state.detection_log) == 0 or st.session_state.detection_log[0]['type'] != 'Té ngã':
                                            st.session_state.fall_count += 1
                                        
                                        # Ghi log
                                        st.session_state.detection_log.insert(0, {
                                            'time': current_time,
                                            'type': 'Té ngã'
                                        })
                                    else:
                                        # Nếu đang ghi video fall -> dừng và lưu
                                        if st.session_state.fall_recording:
                                            st.session_state.fall_end_time = current_time
                                            st.session_state.fall_video_writer.release()
                                            st.session_state.fall_video_writer = None
                                            time.sleep(0.1)
                                            
                                            # Chỉ lưu video nếu đủ dài
                                            MIN_FALL_FRAMES = 10
                                            if st.session_state.fall_frame_count >= MIN_FALL_FRAMES:
                                                video_path = Path(st.session_state.current_video_filename)
                                                if video_path.exists() and video_path.stat().st_size > 0:
                                                    st.session_state.saved_fall_videos.append({
                                                        'filename': st.session_state.current_video_filename,
                                                        'start_time': st.session_state.fall_start_time,
                                                        'end_time': st.session_state.fall_end_time,
                                                        'frame_count': st.session_state.fall_frame_count
                                                    })
                                            else:
                                                # Video quá ngắn -> xóa file
                                                try:
                                                    video_path = Path(st.session_state.current_video_filename)
                                                    if video_path.exists():
                                                        video_path.unlink()
                                                except:
                                                    pass
                                            
                                            st.session_state.fall_recording = False
                                        
                                        # Ghi log
                                        st.session_state.detection_log.insert(0, {
                                            'time': current_time,
                                            'type': 'Bình thường'
                                        })
                                    
                                    # Giới hạn log
                                    if len(st.session_state.detection_log) > 50:
                                        st.session_state.detection_log.pop()
                            
                        # === TRƯỜNG HỢP 2: PHÁT HIỆN NGƯỜI ===
                        else: 
                            no_detect_count = 0  # Reset counter
                            
                            # === PREDICT NẾU ĐỦ FRAMES ===
                            if len(buffer) == num_frames:
                                # Chạy model
                                video_tensor = torch.stack(list(buffer)).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    output = model(video_tensor)
                                    prob = torch.sigmoid(output).item()
                                # prob_to_binary = 1 if prob > 0.5 else 0
                                # smooth_frame.append(prob_to_binary)
                                    
                                # count_fall_frame = sum(smooth_frame)
                                # count_nofall_frame = len(smooth_frame) - count_fall_frame
                                # # Xác định fall
                                # is_fall = True if count_fall_frame > count_nofall_frame else False

                                is_fall = True if prob > 0.5 else False
                                
                                label = "TE NGA!" if is_fall else "Binh thuong"
                                color = (0, 0, 255) if is_fall else (0, 255, 0)
                                
                                # Cập nhật trạng thái
                                st.session_state.current_status = "Nguy hiểm" if is_fall else "An toàn"
                                
                                # Xử lý ghi video fall
                                if is_fall:
                                    # Bắt đầu ghi video (nếu chưa ghi)
                                    if not st.session_state.fall_recording:
                                        st.session_state.fall_video_count += 1
                                        video_filename = output_dir / f"fall_{st.session_state.fall_video_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                                        
                                        h, w = frame_bgr.shape[:2] 
                                        fourcc = cv2.VideoWriter_fourcc(*'avc1')
                                        writer = cv2.VideoWriter(str(video_filename), fourcc, 20.0, (w, h))
                                        
                                        st.session_state.fall_video_writer = writer
                                        st.session_state.fall_recording = True
                                        st.session_state.fall_frame_count = 0
                                        st.session_state.fall_start_time = current_time
                                        st.session_state.current_video_filename = str(video_filename)
                                        
                                        # Ghi buffer frames trước đó
                                        for old_frame in frames_add_video:
                                            st.session_state.fall_video_writer.write(old_frame)
                                    
                                    # Ghi frame hiện tại
                                    st.session_state.fall_video_writer.write(frame_bgr)
                                    st.session_state.fall_frame_count += 1
                                    
                                    # Tăng số lần fall (chỉ tính lần đầu)
                                    if len(st.session_state.detection_log) == 0 or st.session_state.detection_log[0]['type'] != 'Té ngã':
                                        st.session_state.fall_count += 1
                                    
                                    # Ghi log
                                    st.session_state.detection_log.insert(0, {
                                        'time': current_time,
                                        'type': 'Té ngã'
                                    })
                                else:
                                    # Nếu đang ghi video fall -> dừng và lưu
                                    if st.session_state.fall_recording:
                                        st.session_state.fall_end_time = current_time
                                        st.session_state.fall_video_writer.release()
                                        st.session_state.fall_video_writer = None
                                        time.sleep(0.1)
                                        
                                        # Chỉ lưu video nếu đủ dài
                                        MIN_FALL_FRAMES = 10
                                        if st.session_state.fall_frame_count >= MIN_FALL_FRAMES:
                                            video_path = Path(st.session_state.current_video_filename)
                                            if video_path.exists() and video_path.stat().st_size > 0:
                                                st.session_state.saved_fall_videos.append({
                                                    'filename': st.session_state.current_video_filename,
                                                    'start_time': st.session_state.fall_start_time,
                                                    'end_time': st.session_state.fall_end_time,
                                                    'frame_count': st.session_state.fall_frame_count
                                                })
                                        else:
                                            # Video quá ngắn -> xóa file
                                            try:
                                                video_path = Path(st.session_state.current_video_filename)
                                                if video_path.exists():
                                                    video_path.unlink()
                                            except:
                                                pass
                                        
                                        st.session_state.fall_recording = False
                                    
                                    # Ghi log
                                    st.session_state.detection_log.insert(0, {
                                        'time': current_time,
                                        'type': 'Bình thường'
                                    })
                                
                                # Giới hạn log
                                if len(st.session_state.detection_log) > 50:
                                    st.session_state.detection_log.pop()
                                
                                # Vẽ khung và nhãn
                                x1, y1, x2, y2 = bbox
                                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 3)
                                cv2.putText(frame_bgr, f"{label} ({prob*100:.1f}%)", (x1, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                
                            
                            # Đang thu thập frames (chưa đủ để phân tích)
                            else:
                                x1, y1, x2, y2 = bbox
                                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)
                                cv2.putText(frame_bgr, f"Thu thap: {len(buffer)}/{num_frames}", 
                                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        # Display frame
                        frame_rgb_display = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb_display, channels="RGB", width='stretch')
                        
                        # Update status and stats dynamically
                        if st.session_state.current_status == "An toàn":
                            status_container.markdown("""
                                <div class="status-safe">
                                    <h4 style="margin: 0; font-size: 1.2rem;">🟢 An toàn</h4>
                                    <p style="margin: 0.3rem 0 0 0; font-size: 0.85rem;">Không phát hiện nguy cơ té ngã</p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            status_container.markdown("""
                                <div class="status-danger">
                                    <h4 style="margin: 0; font-size: 1.2rem;">🔴 Nguy hiểm</h4>
                                    <p style="margin: 0.3rem 0 0 0; font-size: 0.85rem;">Phát hiện nguy cơ té ngã</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # CẬP NHẬT NHẬT KÝ REAL-TIME (trong col_right)
                        log_html = ""
                        for log_entry in st.session_state.detection_log[:15]:  # Hiển thị 15 dòng gần nhất
                            if log_entry['type'] == 'Té ngã':
                                log_html += f'<p style="margin: 0.4rem 0; color: #ff4444; font-size: 1.1rem;">🔴 {log_entry["time"]}: <strong>Té ngã</strong></p>'
                            else:
                                log_html += f'<p style="margin: 0.4rem 0; color: #44ff44; font-size: 1.1rem;">🟢 {log_entry["time"]}: Bình thường</p>'
                        
                        if log_html:
                            log_container.markdown(f"""
                                <div style="background: #2d2d44; padding: 1rem; border-radius: 10px; max-height: 350px; overflow-y: auto;">
                                    <div style="line-height: 1.8;">
                                        {log_html}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            log_container.caption("📋 Chưa có nhật ký")
                        
                except Exception as e:
                    frame_placeholder.error(f"Lỗi: {str(e)}")
                finally:
                    # Dừng ghi video nếu đang ghi
                    if st.session_state.fall_recording and st.session_state.fall_video_writer:
                        st.session_state.fall_video_writer.release()
                        st.session_state.fall_recording = False
                        st.session_state.fall_video_writer = None
                    
                    cap.release()
                    st.session_state.camera_active = False
        else:
            # KHI CAMERA TẮT: Hiển thị thống kê tổng hợp và nhật ký thu gọn
            if st.session_state.start_time:
                # Tính thời gian giám sát
                if hasattr(st.session_state, 'end_time'):
                    total_elapsed = int(st.session_state.end_time - st.session_state.start_time)
                else:
                    total_elapsed = int(time.time() - st.session_state.start_time)
                
                hours = total_elapsed // 3600
                minutes = (total_elapsed % 3600) // 60
                seconds = total_elapsed % 60
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                # Hiển thị thống kê tổng hợp
                stats_container.markdown("### Thống kê tổng hợp")
                stats_container.markdown(f"""
                    <div class="stats-box">
                        <p style="margin: 0.5rem 0;"><strong>⏱️ Thời gian giám sát:</strong> {time_str}</p>
                        <p style="margin: 0.5rem 0;"><strong>⚠️ Cảnh báo hôm nay:</strong> {st.session_state.fall_count}</p>
                        <p style="margin: 0.5rem 0;"><strong>📹 Số Video Fall:</strong> {len(st.session_state.saved_fall_videos)}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Nhật ký thu gọn (chỉ hiển thị text)
            log_html = ""
            for log_entry in st.session_state.detection_log[:10]:  # Chỉ 10 dòng khi tắt
                if log_entry['type'] == 'Té ngã':
                    log_html += f'<p style="margin: 0.3rem 0; color: #ff4444;">🔴 {log_entry["time"]}: <strong>Té ngã</strong></p>'
                else:
                    log_html += f'<p style="margin: 0.3rem 0; color: #44ff44;">🟢 {log_entry["time"]}: Bình thường</p>'
            
            if log_html:
                log_container.markdown(f"""
                    <div style="background: #2d2d44; padding: 1rem; border-radius: 10px; max-height: 200px; overflow-y: auto;">
                        <div style="font-size: 0.85rem; line-height: 1.6;">
                            {log_html}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                log_container.caption("📋 Chưa có nhật ký")
            
            # Hiển thị video fall history
            with video_history_placeholder.container():
                if len(st.session_state.saved_fall_videos) > 0:
                    
                    for video_idx, video_info in enumerate(st.session_state.saved_fall_videos):
                        with st.expander(f"⚠️ Cảnh té ngã {video_idx + 1}", expanded=(video_idx==0)):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                video_path = Path(video_info['filename'])
                                with open(video_path, 'rb') as f:
                                    video_bytes = f.read()
                                st.video(video_bytes)
                                st.download_button(
                                    label="📥 Tải video",
                                    data=video_bytes,
                                    file_name=video_path.name,
                                    mime="video/mp4",
                                    width='stretch',
                                    key=f"download_static_{video_idx}"
                                )
                            
                            with col2:
                                st.markdown(f"**📍 Thời gian ngã:**")
                                st.markdown(f"🕐 Bắt đầu: `{video_info['start_time']}`")
                                st.markdown(f"🕐 Kết thúc: `{video_info['end_time']}`")
                                st.markdown(f"📊 Số frames: `{video_info.get('frame_count', 0)}`")
                                
                                # Tính thời lượng (giả sử 20 fps)
                                duration_sec = video_info.get('frame_count', 0) / 20.0
                                st.markdown(f"⏱️ Thời lượng: `~{duration_sec:.1f}s`")
                else:
                    st.markdown("""
                                <div class="history-container">
                                    <p style="text-align: center; color: #888;">
                                        📋 Hãy bật camera và ngã để không thấy thông báo này
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

'''

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import albumentations as A
from pathlib import Path
import tempfile
from collections import deque
from ultralytics import YOLO
import time
from datetime import datetime
import os

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Hệ thống Dự đoán Fall",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# Custom CSS - Theo thiết kế trong ảnh
# ============================================================
st.markdown("""
    <style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&display=swap');
    
    /* Global */
    * {
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Main container background */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #ffffff;
    }
    
    /* Header gradient */
    .header-gradient {
        background: linear-gradient(90deg, #4158D0 0%, #C850C0 46%, #FFCC70 100%);
        padding: 2rem 0;
        text-align: center;
        margin: -1rem -1rem 2rem -1rem;
    }
    
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #1a1a2e;
        padding: 1rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d2d44;
        color: #ffffff;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        border: none;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4158D0 0%, #C850C0 100%);
        color: white;
    }
    
    /* Section headers */
    h2, h3 {
        color: #ffffff !important;
    }
    
    /* Cards/Containers */
    .upload-container {
        background: #2d2d44;
        border: 2px dashed #4158D0;
        border-radius: 15px;
        padding: 3rem 2rem;
        text-align: center;
        min-height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .result-container {
        background: #2d2d44;
        border-radius: 15px;
        padding: 2rem;
        min-height: 300px;
    }
    
    .status-safe {
        background: linear-gradient(135deg, #0f5132 0%, #198754 100%);
        color: white;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .status-danger {
        background: linear-gradient(135deg, #842029 0%, #dc3545 100%);
        color: white;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .stats-box {
        background: #1a1a2e;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #4158D0 0%, #C850C0 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(65, 88, 208, 0.4);
    }
    
    /* Camera placeholder */
    .camera-placeholder {
        background: #1a1a2e;
        border: 2px solid #2d2d44;
        border-radius: 10px;
        padding: 3rem;
        text-align: center;
        color: #888;
        min-height: 400px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    /* History section */
    .history-container {
        background: #2d2d44;
        border-radius: 15px;
        padding: 2rem;
        margin-top: 2rem;
    }
    
    /* File uploader - chỉ ẩn khi chưa active */
    .stFileUploader {
        background: transparent;
    }
    
    /* Ẩn text "Drag and drop" */
    [data-testid="stFileUploader"] section {
        background: #2d2d44;
        border-radius: 10px;
        padding: 1rem;
    }
    
    [data-testid="stFileUploader"] section > div:first-child {
        display: none;
    }
    
    /* Style browse button */
    [data-testid="stFileUploader"] button {
        background: linear-gradient(90deg, #4158D0 0%, #C850C0 100%) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #2d2d44;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# Model Definition
# ============================================================
class EfficientNetLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, dropout=0.5):
        super(EfficientNetLSTM, self).__init__()
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        self.efficientnet = efficientnet_v2_s(weights=weights)
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(256*2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        x = x.view(batch_size * num_frames, c, h, w)
        features = self.efficientnet(x)
        features = features.view(batch_size, num_frames, -1)
        lstm_out, _ = self.lstm(features)
        final_features = lstm_out[:, -1, :]
        output = self.fc(final_features)
        return output.squeeze()

# ============================================================
# Helper Functions
# ============================================================
@st.cache_resource
def load_model(model_path):
    """Load the fall detection model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetLSTM(hidden_size=256, num_layers=2, dropout=0.5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

@st.cache_resource
def load_yolo_model():
    """Load YOLO model for person detection"""
    return YOLO("yolov8n.pt")

def get_transform():
    """Get image transformation pipeline"""
    return A.Compose([
        A.Resize(height=288, width=288),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToTensorV2(),
    ])

def extract_frames_from_video(video_path, num_frames=16):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    if total_frames >= num_frames:
        index_frames = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        index_frame = np.arange(0, total_frames)
        index_frame_lack = np.full(num_frames - total_frames, index_frame[-1])
        index_frames = np.concatenate((index_frame, index_frame_lack))
    
    for index_frame in index_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index_frame)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return np.array(frames)

def detect_person_bbox(frame_bgr, yolo_model, conf_thres=0.5):
    """Detect person bounding box using YOLO"""
    res = yolo_model(frame_bgr, verbose=False, conf=conf_thres)
    boxes = res[0].boxes.data.cpu().numpy()
    
    for x1, y1, x2, y2, cf, cl in boxes:
        if int(cl) == 0:  # person class
            if x2 > x1 and y2 > y1:
                return (int(x1), int(y1), int(x2), int(y2))
    return None

def frames_to_tensor(frame_rgb, transform):
    """Convert frame to tensor"""
    frame_tensor = transform(image=frame_rgb)['image']
    return frame_tensor

def predict_video(model, frames, transform, device, threshold=0.5):
    """Predict if video contains a fall"""
    video_tensor = transform(images=frames)['images']
    video_tensor = video_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(video_tensor)
        probability = torch.sigmoid(output).item()
    
    prediction = 'Fall' if probability > threshold else 'No Fall'
    confidence = probability if probability > 0.5 else (1 - probability)
    
    return prediction, confidence, probability

# ============================================================
# Main App
# ============================================================
def main():
    # Initialize session state variables
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'detection_log' not in st.session_state:
        st.session_state.detection_log = []
    if 'fall_count' not in st.session_state:
        st.session_state.fall_count = 0
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'current_status' not in st.session_state:
        st.session_state.current_status = "An toàn"
    if 'fall_video_writer' not in st.session_state:
        st.session_state.fall_video_writer = None
    if 'fall_recording' not in st.session_state:
        st.session_state.fall_recording = False
    if 'fall_video_count' not in st.session_state:
        st.session_state.fall_video_count = 0
    if 'saved_fall_videos' not in st.session_state:
        st.session_state.saved_fall_videos = []
    if 'video_analysis_history' not in st.session_state:
        st.session_state.video_analysis_history = []
    if 'fall_start_time' not in st.session_state:
        st.session_state.fall_start_time = None
    if 'fall_end_time' not in st.session_state:
        st.session_state.fall_end_time = None
    if 'fall_frame_count' not in st.session_state:
        st.session_state.fall_frame_count = 0
    if 'current_video_filename' not in st.session_state:
        st.session_state.current_video_filename = None
    if 'video_analyzed' not in st.session_state:
        st.session_state.video_analyzed = False
    if 'video_analysis_result' not in st.session_state:
        st.session_state.video_analysis_result = None
    
    # Header với gradient giống ảnh
    st.markdown("""
        <div class="header-gradient">
            <h1 class="main-title">🎈 Hệ thống Dự đoán Fall</h1>
            <p class="subtitle">Công nghệ AI tiên tiến để phát hiện nguy cơ té ngã</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load models
    model_path = "best_model_efficientnet_lstm_v2.pth"
    num_frames = 32
    threshold = 0.5
    conf_thres = 0.5
    
    try:
        with st.spinner("Đang tải models..."):
            model, device = load_model(model_path)
            yolo_model = load_yolo_model()
            transform = get_transform()
    except Exception as e:
        st.error(f"Lỗi khi tải models: {str(e)}")
        return
    
    # Tabs chính
    tab1, tab2 = st.tabs(["📹 Dự đoán qua Video", "📹 Dự đoán Realtime"])
    
    # ============================================================
    # TAB 1: VIDEO PREDICTION
    # ============================================================
    with tab1:
        st.markdown("## 📹 Phân tích Video")
        
        col_left, col_right = st.columns([7, 3])
        
        with col_left:
            st.markdown("### Tải lên video")
            
            # Khởi tạo session state cho upload
            if 'uploaded_video' not in st.session_state:
                st.session_state.uploaded_video = None
            
            if st.session_state.uploaded_video is None:
                # Hiển thị placeholder và file uploader
                st.markdown("""
                    <div class="camera-placeholder">
                        <div style="font-size: 4rem;">📷</div>
                        <p style="font-size: 1.2rem; margin-top: 1rem;">Camera chưa được kích hoạt</p>
                    </div>
                """, unsafe_allow_html=True)
                uploaded_file = st.file_uploader(
                    "Chọn video để phân tích",
                    type=['mp4', 'avi', 'mov', 'mkv'],
                    label_visibility="collapsed",
                    key="video_uploader"
                )
                
                if uploaded_file is not None:
                    st.session_state.uploaded_video = uploaded_file
                    # Reset trạng thái phân tích khi tải video mới
                    st.session_state.video_analyzed = False
                    st.session_state.video_analysis_result = None
                    st.rerun()
                
                
            else:
                # Hiển thị video đã upload
                st.video(st.session_state.uploaded_video)
                st.caption(f"✓ Đã tải: {st.session_state.uploaded_video.name}")
                if st.button("🔄 Chọn video khác", use_container_width=True):
                    st.session_state.uploaded_video = None
                    # Reset trạng thái phân tích
                    st.session_state.video_analyzed = False
                    st.session_state.video_analysis_result = None
                    st.rerun()
            
            uploaded_file = st.session_state.uploaded_video
        
        with col_right:
            st.markdown("### Kết quả phân tích")
            
            if uploaded_file is None:
                st.markdown("""
                    <div class="result-container">
                        <p style="text-align: center; color: #888; margin-top: 6rem;">
                            Chưa có video để phân tích
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.button("🔵 Bắt đầu phân tích", disabled=True, use_container_width=True)
            else:
                # Nếu chưa phân tích hoặc đổi video mới
                if not st.session_state.video_analyzed or st.session_state.video_analysis_result is None:
                    if st.button("🔵 Bắt đầu phân tích", use_container_width=True):
                        # Save video
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        tfile.write(uploaded_file.read())
                        video_path = tfile.name
                        
                        with st.spinner("Đang phân tích video..."):
                            progress_bar = st.progress(0)
                            
                            # Extract frames
                            progress_bar.progress(30)
                            frames = extract_frames_from_video(video_path, num_frames)
                            
                            # Predict
                            progress_bar.progress(70)
                            prediction, confidence, probability = predict_video(
                                model, frames, transform, device, threshold
                            )
                            
                            progress_bar.progress(100)
                            time.sleep(0.3)
                            progress_bar.empty()
                        
                        # Lấy thông tin video
                        cap_info = cv2.VideoCapture(video_path)
                        total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = int(cap_info.get(cv2.CAP_PROP_FPS))
                        duration_sec = total_frames / fps if fps > 0 else 0
                        cap_info.release()
                        
                        # Lưu kết quả vào session state
                        st.session_state.video_analyzed = True
                        st.session_state.video_analysis_result = {
                            'prediction': prediction,
                            'confidence': confidence,
                            'probability': probability,
                            'video_name': uploaded_file.name,
                            'total_frames': total_frames,
                            'fps': fps,
                            'duration_sec': duration_sec
                        }
                        
                        # Lưu vào lịch sử
                        analysis_time = datetime.now().strftime('%d/%m/%Y - %H:%M')
                        st.session_state.video_analysis_history.insert(0, {
                            'time': analysis_time,
                            'video_name': uploaded_file.name,
                            'result': 'Fall detected' if prediction == 'Fall' else 'Safe',
                            'confidence': confidence,
                            'probability': probability,
                            'is_fall': prediction == 'Fall'
                        })
                        
                        st.rerun()
                else:
                    # Hiển thị thông tin chi tiết thay cho nút
                    result = st.session_state.video_analysis_result
                    
                    st.markdown("**📊 Thông tin chi tiết:**")
                    st.markdown(f"• **Tên video:** `{result['video_name']}`")
                    st.markdown(f"• **Thời lượng:** `{int(result['duration_sec']//60):02d}:{int(result['duration_sec']%60):02d}s`")
                    st.markdown(f"• **Tổng số khung hình:** `{result['total_frames']}`")
                    st.markdown(f"• **Tỷ lệ khung hình (FPS):** `{result['fps']}`")
                    
                    if result['prediction'] == 'Fall':
                        st.markdown(f"• **Kết quả tổng thể:** 🟥 **Phát hiện NGÃ**")
                    else:
                        st.markdown(f"• **Kết quả tổng thể:** 🟩 **Không phát hiện ngã**")
                    
                    st.markdown(f"• **Độ tin cậy (Confidence):** `{result['confidence']:.2f}`")
        
        # History section
        st.markdown("---")
        st.markdown("## 📊 Lịch sử cảnh báo")
        
        if len(st.session_state.video_analysis_history) > 0:
            # Header của bảng
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            with col1:
                st.markdown("**Thời gian phân tích**")
            with col2:
                st.markdown("**Tên video**")
            with col3:
                st.markdown("**Kết quả**")
            with col4:
                st.markdown("**Confidence**")
            
            st.markdown("---")
            
            # Hiển thị từng dòng lịch sử
            for entry in st.session_state.video_analysis_history:
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
                    st.text(entry['time'])
                
                with col2:
                    st.text(entry['video_name'])
                
                with col3:
                    if entry['is_fall']:
                        st.markdown("⚠️ **<span style='color: #ff4444;'>Fall detected</span>**", unsafe_allow_html=True)
                    else:
                        st.markdown("✅ **<span style='color: #44ff44;'>Safe</span>**", unsafe_allow_html=True)
                
                with col4:
                    st.text(f"{entry['confidence']:.2f}")
                
                st.markdown("---")
            
            # Nút xóa lịch sử
            if st.button("🗑️ Xóa lịch sử", key="clear_history"):
                st.session_state.video_analysis_history = []
                st.rerun()
        else:
            st.markdown("""
                <div class="history-container">
                    <p style="text-align: center; color: #888;">
                        📋 Chưa có cảnh báo nào được ghi nhận
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    # ============================================================
    # TAB 2: REALTIME DETECTION
    # ============================================================
    with tab2:
        st.markdown("## 📹 Giám sát Realtime")
        
        col_left, col_right = st.columns([7, 3])
        
        with col_left:
            # st.markdown("### Camera feed")
            frame_placeholder = st.empty()
            
            if not st.session_state.camera_active:
                frame_placeholder.markdown("""
                    <div class="camera-placeholder">
                        <div style="font-size: 4rem;">📷</div>
                        <p style="font-size: 1.2rem; margin-top: 1rem;">Camera chưa được kích hoạt</p>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("🎬 Bật Camera", use_container_width=True, type="primary"):
                    st.session_state.camera_active = True
                    st.session_state.start_time = time.time()
                    # st.session_state.fall_count = 0
                    # st.session_state.detection_log = []
                    st.rerun()
            else:
                if st.button("⏹️ Dừng giám sát", use_container_width=True, type="secondary"):
                    st.session_state.camera_active = False
                    st.session_state.end_time = time.time()  # Lưu thời gian kết thúc
                    st.rerun()
        
        with col_right:
            # Status box
            st.markdown("### Trạng thái hiện tại")
            status_container = st.empty()
            
            if st.session_state.current_status == "An toàn":
                status_container.markdown("""
                    <div class="status-safe">
                        <h4 style="margin: 0;">🟢 An toàn</h4>
                        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Không phát hiện nguy cơ té ngã</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                status_container.markdown("""
                    <div class="status-danger">
                        <h4 style="margin: 0;">🔴 Nguy hiểm</h4>
                        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Phát hiện nguy cơ té ngã</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### Nhật ký phát hiện")
            log_container = st.empty()
            
            # Thống kê (chỉ hiển thị khi camera TẮT)
            stats_container = st.empty()
        
        # History section placeholders
        st.markdown("---")
        st.markdown("## 📊 Video cảnh fall")
        
        video_history_placeholder = st.empty()
        
        # Realtime detection loop
        if st.session_state.camera_active:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                frame_placeholder.error("❌ Không thể mở webcam")
                st.session_state.camera_active = False
                st.rerun()
            else:
                buffer = deque(maxlen=num_frames)
                frames_add_video = deque(maxlen=30)
                smooth_frame = deque(maxlen = 5)
                output_dir = Path("fall_videos")
                output_dir.mkdir(exist_ok=True)
                no_detect_count = 0
                try:
                    while st.session_state.camera_active:
                        ret, frame_bgr = cap.read()
                        if not ret:
                            break
                        
                        frames_add_video.append(frame_bgr)
                        # Hiển thị thời gian ở góc dưới trái
                        h, w = frame_bgr.shape[:2]
                        current_time = datetime.now().strftime('%d/%m/%Y - %H:%M:%S')
                        cv2.putText(frame_bgr, current_time, (10, h - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        
                        # Thêm frame vào buffer để phân tích
                        frame_tensor = frames_to_tensor(frame_rgb, transform)
                        buffer.append(frame_tensor)
                        # Phát hiện người trong frame
                        bbox = detect_person_bbox(frame_bgr, yolo_model, conf_thres)
                        
                        # === TRƯỜNG HỢP 1: KHÔNG PHÁT HIỆN NGƯỜI ===
                        if bbox is None:
                            no_detect_count += 1
                            
                            if no_detect_count >= 10:
                                buffer.clear()
                                no_detect_count = 0
                                frames_add_video.clear()
                                st.session_state.smoothed_prob = 0.0
                                cv2.putText(frame_bgr, "Khong phat hien nguoi", 
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                                st.session_state.current_status = "An toàn"
                                
                                # Nếu đang ghi video fall -> dừng ghi và lưu
                                if st.session_state.fall_recording:
                                    st.session_state.fall_video_writer.release()
                                    st.session_state.fall_recording = False
                                    st.session_state.fall_video_writer = None

                                    time.sleep(0.1)
                                            
                                    video_path = Path(st.session_state.current_video_filename)
                                    if video_path.exists() and video_path.stat().st_size > 0:
                                        st.session_state.saved_fall_videos.append({
                                            'filename': st.session_state.current_video_filename,
                                            'start_time': st.session_state.fall_start_time,
                                            'end_time': st.session_state.fall_end_time,
                                            'frame_count': st.session_state.fall_frame_count
                                        })
                            else:
                                # # Chưa đủ 10 frames -> Vẫn predict nếu đủ frames trong buffer
                                # cv2.putText(frame_bgr, f"Mat nguoi ({no_detect_count}/10)", 
                                #         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                                
                                if len(buffer) == num_frames:
                                    # Chạy model
                                    video_tensor = torch.stack(list(buffer)).unsqueeze(0).to(device)
                                    with torch.no_grad():
                                        output = model(video_tensor)
                                        prob = torch.sigmoid(output).item()
                                    
                                    # prob_to_binary = 1 if prob > 0.5 else 0
                                    # smooth_frame.append(prob_to_binary)
                                    
                                    # count_fall_frame = sum(smooth_frame)
                                    # count_nofall_frame = len(smooth_frame) - count_fall_frame
                                    # # Xác định fall
                                    # is_fall = True if count_fall_frame > count_nofall_frame else False

                                    is_fall = True if prob > 0.5 else False
                                    label = "TE NGA!" if is_fall else "Binh thuong"
                                    color = (0, 0, 255) if is_fall else (0, 255, 0)
                                    
                                    # Cập nhật trạng thái
                                    st.session_state.current_status = "Nguy hiểm" if is_fall else "An toàn"
                                    
                                    # Vẽ label (không có bbox)
                                    cv2.putText(frame_bgr, f"{label} ({prob*100:.1f}%)", (10, 70),
                                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                                    
                                    # Xử lý ghi video fall
                                    if is_fall:
                                        # Bắt đầu ghi video (nếu chưa ghi)
                                        if not st.session_state.fall_recording:
                                            st.session_state.fall_video_count += 1
                                            video_filename = output_dir / f"fall_{st.session_state.fall_video_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                                            
                                            h, w = frame_bgr.shape[:2] 
                                            fourcc = cv2.VideoWriter_fourcc(*'avc1')
                                            writer = cv2.VideoWriter(str(video_filename), fourcc, 20.0, (w, h))
                                            
                                            st.session_state.fall_video_writer = writer
                                            st.session_state.fall_recording = True
                                            st.session_state.fall_frame_count = 0
                                            st.session_state.fall_start_time = current_time
                                            st.session_state.current_video_filename = str(video_filename)
                                            
                                            # Ghi buffer frames trước đó
                                            for old_frame in frames_add_video:
                                                st.session_state.fall_video_writer.write(old_frame)
                                        
                                        # Ghi frame hiện tại
                                        st.session_state.fall_video_writer.write(frame_bgr)
                                        st.session_state.fall_frame_count += 1
                                        
                                        # Tăng số lần fall (chỉ tính lần đầu)
                                        if len(st.session_state.detection_log) == 0 or st.session_state.detection_log[0]['type'] != 'Té ngã':
                                            st.session_state.fall_count += 1
                                        
                                        # Ghi log
                                        st.session_state.detection_log.insert(0, {
                                            'time': current_time,
                                            'type': 'Té ngã'
                                        })
                                    else:
                                        # Nếu đang ghi video fall -> dừng và lưu
                                        if st.session_state.fall_recording:
                                            st.session_state.fall_end_time = current_time
                                            st.session_state.fall_video_writer.release()
                                            st.session_state.fall_video_writer = None
                                            time.sleep(0.1)
                                            
                                            # Chỉ lưu video nếu đủ dài
                                            MIN_FALL_FRAMES = 10
                                            if st.session_state.fall_frame_count >= MIN_FALL_FRAMES:
                                                video_path = Path(st.session_state.current_video_filename)
                                                if video_path.exists() and video_path.stat().st_size > 0:
                                                    st.session_state.saved_fall_videos.append({
                                                        'filename': st.session_state.current_video_filename,
                                                        'start_time': st.session_state.fall_start_time,
                                                        'end_time': st.session_state.fall_end_time,
                                                        'frame_count': st.session_state.fall_frame_count
                                                    })
                                            else:
                                                # Video quá ngắn -> xóa file
                                                try:
                                                    video_path = Path(st.session_state.current_video_filename)
                                                    if video_path.exists():
                                                        video_path.unlink()
                                                except:
                                                    pass
                                            
                                            st.session_state.fall_recording = False
                                        
                                        # Ghi log
                                        st.session_state.detection_log.insert(0, {
                                            'time': current_time,
                                            'type': 'Bình thường'
                                        })
                                    
                                    # Giới hạn log
                                    if len(st.session_state.detection_log) > 50:
                                        st.session_state.detection_log.pop()
                            
                        # === TRƯỜNG HỢP 2: PHÁT HIỆN NGƯỜI ===
                        else: 
                            no_detect_count = 0  # Reset counter
                            
                            # === PREDICT NẾU ĐỦ FRAMES ===
                            if len(buffer) == num_frames:
                                # Chạy model
                                video_tensor = torch.stack(list(buffer)).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    output = model(video_tensor)
                                    prob = torch.sigmoid(output).item()
                                # prob_to_binary = 1 if prob > 0.5 else 0
                                # smooth_frame.append(prob_to_binary)
                                    
                                # count_fall_frame = sum(smooth_frame)
                                # count_nofall_frame = len(smooth_frame) - count_fall_frame
                                # # Xác định fall
                                # is_fall = True if count_fall_frame > count_nofall_frame else False

                                is_fall = True if prob > 0.5 else False
                                
                                label = "TE NGA!" if is_fall else "Binh thuong"
                                color = (0, 0, 255) if is_fall else (0, 255, 0)
                                
                                # Cập nhật trạng thái
                                st.session_state.current_status = "Nguy hiểm" if is_fall else "An toàn"
                                
                                # Xử lý ghi video fall
                                if is_fall:
                                    # Bắt đầu ghi video (nếu chưa ghi)
                                    if not st.session_state.fall_recording:
                                        st.session_state.fall_video_count += 1
                                        video_filename = output_dir / f"fall_{st.session_state.fall_video_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                                        
                                        h, w = frame_bgr.shape[:2] 
                                        fourcc = cv2.VideoWriter_fourcc(*'avc1')
                                        writer = cv2.VideoWriter(str(video_filename), fourcc, 20.0, (w, h))
                                        
                                        st.session_state.fall_video_writer = writer
                                        st.session_state.fall_recording = True
                                        st.session_state.fall_frame_count = 0
                                        st.session_state.fall_start_time = current_time
                                        st.session_state.current_video_filename = str(video_filename)
                                        
                                        # Ghi buffer frames trước đó
                                        for old_frame in frames_add_video:
                                            st.session_state.fall_video_writer.write(old_frame)
                                    
                                    # Ghi frame hiện tại
                                    st.session_state.fall_video_writer.write(frame_bgr)
                                    st.session_state.fall_frame_count += 1
                                    
                                    # Tăng số lần fall (chỉ tính lần đầu)
                                    if len(st.session_state.detection_log) == 0 or st.session_state.detection_log[0]['type'] != 'Té ngã':
                                        st.session_state.fall_count += 1
                                    
                                    # Ghi log
                                    st.session_state.detection_log.insert(0, {
                                        'time': current_time,
                                        'type': 'Té ngã'
                                    })
                                else:
                                    # Nếu đang ghi video fall -> dừng và lưu
                                    if st.session_state.fall_recording:
                                        st.session_state.fall_end_time = current_time
                                        st.session_state.fall_video_writer.release()
                                        st.session_state.fall_video_writer = None
                                        time.sleep(0.1)
                                        
                                        # Chỉ lưu video nếu đủ dài
                                        MIN_FALL_FRAMES = 10
                                        if st.session_state.fall_frame_count >= MIN_FALL_FRAMES:
                                            video_path = Path(st.session_state.current_video_filename)
                                            if video_path.exists() and video_path.stat().st_size > 0:
                                                st.session_state.saved_fall_videos.append({
                                                    'filename': st.session_state.current_video_filename,
                                                    'start_time': st.session_state.fall_start_time,
                                                    'end_time': st.session_state.fall_end_time,
                                                    'frame_count': st.session_state.fall_frame_count
                                                })
                                        else:
                                            # Video quá ngắn -> xóa file
                                            try:
                                                video_path = Path(st.session_state.current_video_filename)
                                                if video_path.exists():
                                                    video_path.unlink()
                                            except:
                                                pass
                                        
                                        st.session_state.fall_recording = False
                                    
                                    # Ghi log
                                    st.session_state.detection_log.insert(0, {
                                        'time': current_time,
                                        'type': 'Bình thường'
                                    })
                                
                                # Giới hạn log
                                if len(st.session_state.detection_log) > 50:
                                    st.session_state.detection_log.pop()
                                
                                # Vẽ khung và nhãn
                                x1, y1, x2, y2 = bbox
                                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 3)
                                cv2.putText(frame_bgr, f"{label} ({prob*100:.1f}%)", (x1, y1 - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                                
                            
                            # Đang thu thập frames (chưa đủ để phân tích)
                            else:
                                x1, y1, x2, y2 = bbox
                                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)
                                cv2.putText(frame_bgr, f"Thu thap: {len(buffer)}/{num_frames}", 
                                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        # Display frame
                        frame_rgb_display = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb_display, channels="RGB", use_container_width=True)
                        
                        # Update status and stats dynamically
                        if st.session_state.current_status == "An toàn":
                            status_container.markdown("""
                                <div class="status-safe">
                                    <h4 style="margin: 0; font-size: 1.2rem;">🟢 An toàn</h4>
                                    <p style="margin: 0.3rem 0 0 0; font-size: 0.85rem;">Không phát hiện nguy cơ té ngã</p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            status_container.markdown("""
                                <div class="status-danger">
                                    <h4 style="margin: 0; font-size: 1.2rem;">🔴 Nguy hiểm</h4>
                                    <p style="margin: 0.3rem 0 0 0; font-size: 0.85rem;">Phát hiện nguy cơ té ngã</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # CẬP NHẬT NHẬT KÝ REAL-TIME (trong col_right)
                        log_html = ""
                        for log_entry in st.session_state.detection_log[:15]:  # Hiển thị 15 dòng gần nhất
                            if log_entry['type'] == 'Té ngã':
                                log_html += f'<p style="margin: 0.4rem 0; color: #ff4444; font-size: 1.1rem;">🔴 {log_entry["time"]}: <strong>Té ngã</strong></p>'
                            else:
                                log_html += f'<p style="margin: 0.4rem 0; color: #44ff44; font-size: 1.1rem;">🟢 {log_entry["time"]}: Bình thường</p>'
                        
                        if log_html:
                            log_container.markdown(f"""
                                <div style="background: #2d2d44; padding: 1rem; border-radius: 10px; max-height: 350px; overflow-y: auto;">
                                    <div style="line-height: 1.8;">
                                        {log_html}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            log_container.caption("📋 Chưa có nhật ký")
                        
                except Exception as e:
                    frame_placeholder.error(f"Lỗi: {str(e)}")
                finally:
                    # Dừng ghi video nếu đang ghi
                    if st.session_state.fall_recording and st.session_state.fall_video_writer:
                        st.session_state.fall_video_writer.release()
                        st.session_state.fall_recording = False
                        st.session_state.fall_video_writer = None
                    
                    cap.release()
                    st.session_state.camera_active = False
        else:
            # KHI CAMERA TẮT: Hiển thị thống kê tổng hợp và nhật ký thu gọn
            if st.session_state.start_time:
                # Tính thời gian giám sát
                if hasattr(st.session_state, 'end_time'):
                    total_elapsed = int(st.session_state.end_time - st.session_state.start_time)
                else:
                    total_elapsed = int(time.time() - st.session_state.start_time)
                
                hours = total_elapsed // 3600
                minutes = (total_elapsed % 3600) // 60
                seconds = total_elapsed % 60
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                # Hiển thị thống kê tổng hợp
                stats_container.markdown("### Thống kê tổng hợp")
                stats_container.markdown(f"""
                    <div class="stats-box">
                        <p style="margin: 0.5rem 0;"><strong>⏱️ Thời gian giám sát:</strong> {time_str}</p>
                        <p style="margin: 0.5rem 0;"><strong>⚠️ Cảnh báo hôm nay:</strong> {st.session_state.fall_count}</p>
                        <p style="margin: 0.5rem 0;"><strong>📹 Số Video Fall:</strong> {len(st.session_state.saved_fall_videos)}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Nhật ký thu gọn (chỉ hiển thị text)
            log_html = ""
            for log_entry in st.session_state.detection_log[:10]:  # Chỉ 10 dòng khi tắt
                if log_entry['type'] == 'Té ngã':
                    log_html += f'<p style="margin: 0.3rem 0; color: #ff4444;">🔴 {log_entry["time"]}: <strong>Té ngã</strong></p>'
                else:
                    log_html += f'<p style="margin: 0.3rem 0; color: #44ff44;">🟢 {log_entry["time"]}: Bình thường</p>'
            
            if log_html:
                log_container.markdown(f"""
                    <div style="background: #2d2d44; padding: 1rem; border-radius: 10px; max-height: 200px; overflow-y: auto;">
                        <div style="font-size: 0.85rem; line-height: 1.6;">
                            {log_html}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                log_container.caption("📋 Chưa có nhật ký")
            
            # Hiển thị video fall history
            with video_history_placeholder.container():
                if len(st.session_state.saved_fall_videos) > 0:
                    
                    for video_idx, video_info in enumerate(st.session_state.saved_fall_videos):
                        with st.expander(f"⚠️ Cảnh té ngã {video_idx + 1}", expanded=(video_idx==0)):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                video_path = Path(video_info['filename'])
                                with open(video_path, 'rb') as f:
                                    video_bytes = f.read()
                                st.video(video_bytes)
                                st.download_button(
                                    label="📥 Tải video",
                                    data=video_bytes,
                                    file_name=video_path.name,
                                    mime="video/mp4",
                                    use_container_width=True,
                                    key=f"download_static_{video_idx}"
                                )
                            
                            with col2:
                                st.markdown(f"**📍 Thời gian ngã:**")
                                st.markdown(f"🕐 Bắt đầu: `{video_info['start_time']}`")
                                st.markdown(f"🕐 Kết thúc: `{video_info['end_time']}`")
                                st.markdown(f"📊 Số frames: `{video_info.get('frame_count', 0)}`")
                                
                                # Tính thời lượng (giả sử 20 fps)
                                duration_sec = video_info.get('frame_count', 0) / 20.0
                                st.markdown(f"⏱️ Thời lượng: `~{duration_sec:.1f}s`")
                else:
                    st.markdown("""
                                <div class="history-container">
                                    <p style="text-align: center; color: #888;">
                                        📋 Hãy bật camera và ngã để không thấy thông báo này
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()