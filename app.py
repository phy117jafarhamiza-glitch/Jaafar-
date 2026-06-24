import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import time

# إعدادات واجهة الميديا بايب لتتبع الجسم
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# دالة لحساب الزوايا بين المفاصل
def calculate_angle(a, b, c):
    a = np.array(a)  # نقطة البداية (مثلاً الكتف)
    b = np.array(b)  # النقطة المركزية (مثلاً الورك)
    c = np.array(c)  # نقطة النهاية (مثلاً الركبة)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

# --- تصميم واجهة المستخدم المستجيبة (Streamlit UI) ---
st.set_page_config(page_title="التحليل البيوميكانيكي للقفز العالي", layout="wide")

st.title("🏃‍♂️ نظام التحليل اللحظي لفعالية القفز العالي (Fosbury Flop)")
st.write("هذا التطبيق يستخدم الشبكات العصبية الالتفافية لتتبع حركة اللاعب واستخراج المتغيرات الميكانيكية لحظة بلحظة.")

# شريط جانبي لإدخال بيانات اللاعب
st.sidebar.header("📊 بيانات اللاعب للمعايرة")
weight = st.sidebar.number_input("وزن اللاعب (كجم):", min_value=30.0, max_value=150.0, value=70.0, step=0.5)
height_cm = st.sidebar.number_input("طول اللاعب (سم):", min_value=120.0, max_value=230.0, value=185.0, step=1.0)
height_m = height_cm / 100.0  # تحويل الطول لمتر

# خيارات الإدخال (بث مباشر أو رفع فيديو)
st.sidebar.header("📹 مصدر الفيديو")
source_option = st.sidebar.radio("اختر طريقة التحليل:", ("رفع ملف فيديو مسجل", "تشغيل البث المباشر (الكاميرا)"))

video_file = None
run_live = False

if source_option == "رفع ملف فيديو مسجل":
    video_file = st.sidebar.file_uploader("قم برفع فيديو القفزة (MP4, MOV, AVI):", type=["mp4", "mov", "avi"])
else:
    run_live = st.sidebar.checkbox("فتح / إغلاق الكاميرا")

# مكان عرض الفيديو والبيانات
col1, col2 = st.columns([2, 1])
with col1:
    view_window = st.image([])  # نافذة عرض الفيديو المعالج
with col2:
    st.subheader("📈 المتغيرات البيوميكانيكية اللحظية")
    stat_velocity = st.empty()
    stat_angle = st.empty()
    stat_energy = st.empty()

# --- معالجة الفيديو والذكاء الاصطناعي ---
if video_file is not None or run_live:
    
    # تحديد مصدر الفيديو
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
    else:
        # 0 تعني كاميرا الحاسوب أو الموبايل الافتراضية
        cap = cv2.VideoCapture(0)
    
    # متغيرات تتبع السرعة
    prev_time = time.time()
    prev_com_y = None
    pixel_to_meter = None
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # تحويل الألوان لأن OpenCV يقرأ BGR وميديا بايب تحتاج RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            
            # معالجة الإطار بالشبكة العصبية
            results = pose.process(frame)
            
            if results.pose_landmarks:
                # رسم الهيكل العظمي للاعب
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # استخراج النقاط الرئيسية
                landmarks = results.pose_landmarks.landmark
                
                # 1. حساب معامل المعايرة (Pixel-to-Meter) بناءً على طول اللاعب
                # سنقيس المسافة بالبكسل من الكاحل إلى الأذن كمؤشر لطول الجسم الكلي
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
                ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * h]
                player_height_pixels = np.linalg.norm(np.array(ankle) - np.array(ear))
                
                if player_height_pixels > 0:
                    pixel_to_meter = height_m / player_height_pixels
                
                # 2. تقريب مركز كتلة الجسم (Center of Mass) عبر نقاط الورك والجذع
                hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
                hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
                shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
                
                # مركز الكتلة التقريبي هو منتصف المسافة بين الحوض والكتف
                com_x = (hip_l[0] + hip_r[0]) / 2
                com_y = (hip_l[1] + hip_r[1]) / 2
                
                # 3. حساب السرعة العمودية (Vertical Velocity) والطاقة
                current_time = time.time()
                dt = current_time - prev_time
                
                if prev_com_y is not None and dt > 0 and pixel_to_meter is not None:
                    # فارق الحركة بالبكسل وتحويله إلى أمتار
                    dy_pixels = prev_com_y - com_y  # في الفيديو الأعلى قيمته Y أقل
                    dy_meters = dy_pixels * pixel_to_meter
                    velocity_y = dy_meters / dt  # السرعة م/ث
                    
                    # حساب الطاقة الحركية العمودية Ke = 0.5 * m * v^2
                    kinetic_energy = 0.5 * weight * (velocity_y ** 2)
                    
                    # 4. حساب زاوية تقوس الظهر (Arching Angle) أثناء العبور
                    # زاوية بين الكتف - الورك - الركبة
                    knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
                    arch_angle = calculate_angle(shoulder_l, hip_l, knee_l)
                    
                    # تحديث لوحة البيانات اللحظية
                    stat_velocity.metric(label="السرعة العمودية الحالية (م/ث)", value=f"{velocity_y:.2f} m/s")
                    stat_angle.metric(label="زاوية تقوس الظهر (درجة)", value=f"{arch_angle:.1f}°")
                    stat_energy.metric(label="الطاقة الحركية العمودية (جول)", value=f"{kinetic_energy:.1f} J")
                    
                    # رسم نقطة مركز الكتلة على الفيديو
                    cv2.circle(frame, (int(com_x), int(com_y)), 8, (255, 0, 0), -1)
                
                prev_com_y = com_y
                prev_time = current_time
            
            # عرض الإطار المعالج في المتصفح
            view_window.image(frame, channels="RGB")
            
    cap.release()
