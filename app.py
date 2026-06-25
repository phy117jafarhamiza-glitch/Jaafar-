import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import time

# تحميل نموذج YOLOv8 المخصص لتقدير الوضعية (سيتم تحميله تلقائياً)
@st.cache_resource
def load_model():
    return YOLO('yolov8n-pose.pt')

model = load_model()

# دالة لحساب الزوايا
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

# --- واجهة المستخدم ---
st.set_page_config(page_title="التحليل البيوميكانيكي للقفز العالي", layout="wide")
st.title("🏃‍♂️ نظام YOLOv8 للتحليل اللحظي لفعالية القفز العالي")

st.sidebar.header("📊 بيانات اللاعب للمعايرة")
weight = st.sidebar.number_input("وزن اللاعب (كجم):", min_value=30.0, max_value=150.0, value=70.0, step=0.5)
height_cm = st.sidebar.number_input("طول اللاعب (سم):", min_value=120.0, max_value=230.0, value=185.0, step=1.0)
height_m = height_cm / 100.0

st.sidebar.header("📹 مصدر الفيديو")
source_option = st.sidebar.radio("اختر طريقة التحليل:", ("رفع ملف فيديو مسجل", "تشغيل البث المباشر (الكاميرا)"))

video_file = None
run_live = False

if source_option == "رفع ملف فيديو مسجل":
    video_file = st.sidebar.file_uploader("قم برفع فيديو القفزة:", type=["mp4", "mov", "avi"])
else:
    run_live = st.sidebar.checkbox("فتح / إغلاق الكاميرا")

col1, col2 = st.columns([2, 1])
with col1:
    view_window = st.image([])
with col2:
    st.subheader("📈 المتغيرات البيوميكانيكية")
    stat_velocity = st.empty()
    stat_angle = st.empty()
    stat_energy = st.empty()

if video_file is not None or run_live:
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
    else:
        cap = cv2.VideoCapture(0)
    
    prev_time = time.time()
    prev_com_y = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w, _ = frame.shape
        
        # التنبؤ باستخدام YOLOv8-Pose
        results = model(frame, verbose=False)
        
        if len(results) > 0 and results[0].keypoints is not None:
            # الحصول على إحداثيات النقاط للاعب الأول المكتشف
            kp = results[0].keypoints.xy[0].cpu().numpy()
            
            # التأكد من اكتشاف النقاط المطلوبة
            if len(kp) > 10:
                # ميديا بايب تستخدم ترتيب مختلف، هنا ترتيب نقاط YOLOv8:
                # 5: الكتف الأيسر, 6: الكتف الأيمن, 11: الورك الأيسر, 12: الورك الأيمن, 13: الركبة اليسرى, 15: الكاحل الأيسر
                try:
                    shoulder_l = kp[5]
                    hip_l = kp[11]
                    hip_r = kp[12]
                    knee_l = kp[13]
                    ankle_l = kp[15]
                    
                    # 1. المعايرة
                    player_height_pixels = np.linalg.norm(ankle_l - shoulder_l)
                    pixel_to_meter = height_m / player_height_pixels if player_height_pixels > 0 else 0.003
                    
                    # 2. مركز الكتلة التقريبي
                    com_x = (hip_l[0] + hip_r[0]) / 2
                    com_y = (hip_l[1] + hip_r[1]) / 2
                    
                    # 3. الحسابات البيوميكانيكية
                    current_time = time.time()
                    dt = current_time - prev_time
                    
                    if prev_com_y is not None and dt > 0:
                        dy_meters = (prev_com_y - com_y) * pixel_to_meter
                        velocity_y = dy_meters / dt
                        kinetic_energy = 0.5 * weight * (velocity_y ** 2)
                        arch_angle = calculate_angle(shoulder_l, hip_l, knee_l)
                        
                        # تحديث الشاشة
                        stat_velocity.metric(label="السرعة العمودية (م/ث)", value=f"{velocity_y:.2f} m/s")
                        stat_angle.metric(label="زاوية تقوس الظهر", value=f"{arch_angle:.1f}°")
                        stat_energy.metric(label="الطاقة الحركية العمودية (جول)", value=f"{kinetic_energy:.1f} J")
                    
                    prev_com_y = com_y
                    prev_time = current_time
                    
                except IndexError:
                    pass
        
        # رسم النتائج تلقائياً بواسطة YOLO
        annotated_frame = results[0].plot() if len(results) > 0 else frame
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        view_window.image(annotated_frame, channels="RGB")
        
    cap.release()
