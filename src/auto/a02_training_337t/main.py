import random
from config import Config
from utils import setup_environment, load_data, save_plot, add_doc_table, add_doc_figure
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost
import shap

def process_file(file_path):
    """ประมวลผลไฟล์ CSV แต่ละไฟล์"""
    print(f"Processing {file_path.name}...")
    
    # โหลดข้อมูลและตั้งค่าเริ่มต้น
    df = load_data(file_path)
    file_name = file_path.stem
    english_name = Config.FILE_NAME_MAPPING.get(file_name, file_name)
    doc = Document()
    doc.add_heading(f"SO2 Prediction Report - {file_name}", 0)
    
    # ทำความสะอาดหน่วยความจำก่อนเริ่มไฟล์ใหม่
    tf.keras.backend.clear_session()
    
    # EDA ก่อนประมวลผล
    explore_data_before(df, doc, file_name, english_name)
    
    # ประมวลผลข้อมูล
    df_clean = preprocess_data(df, doc, file_name)
    if df_clean.empty:
        doc.save(Config.OUTPUT_DIR / f"report_{file_name}_error.docx")
        return
    
    # เตรียมข้อมูลสำหรับโมเดล
    features, X, y = prepare_model_data(df_clean)
    if X is None:
        doc.save(Config.OUTPUT_DIR / f"report_{file_name}_error.docx")
        return
    
    # ฝึกโมเดล LSTM
    train_lstm(X, y, features, doc, file_name, english_name)
    
    # วิเคราะห์ความสำคัญของฟีเจอร์ด้วย XGBoost
    analyze_feature_importance(df_clean, features, doc, file_name, english_name)
    
    # บันทึกเอกสาร
    doc.save(Config.OUTPUT_DIR / f"report_{file_name}.docx")
    print(f"Completed processing {file_name}!")

def explore_data_before(df, doc, file_name, english_name):
    """สำรวจข้อมูลก่อนประมวลผล"""
    doc.add_heading("Initial Data Exploration", 1)
    doc.add_paragraph(f"Shape: {df.shape}")
    add_doc_table(doc, df.head(10), "Sample Data", "ตัวอย่างข้อมูล 10 แถวแรก")
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=True, cmap='Blues')
    save_plot(plt, f"heatmap_before_{english_name}", f"Missing Data - {english_name}")

def preprocess_data(df, doc, file_name):
    """ประมวลผลข้อมูลล่วงหน้า"""
    # ลบฟีเจอร์ที่มีข้อมูลขาดหายเกิน 50% (ยกเว้น SO2)
    missing_pct = df.isnull().mean()
    to_drop = [col for col in missing_pct.index if missing_pct[col] > 0.5 and col != 'SO2']
    if to_drop:
        df = df.drop(columns=to_drop)
        doc.add_paragraph(f"Dropped features: {to_drop}")
    
    # ตัดข้อมูลตาม SO2 ที่ไม่ใช่ NaN
    if 'SO2' in df.columns:
        df = df[df['SO2'].notna()]
    
    # เติมข้อมูลที่ขาดหาย
    all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
    df = df.reindex(all_dates).interpolate(method='linear').dropna()
    
    return df

def prepare_model_data(df):
    """เตรียมข้อมูลสำหรับโมเดล"""
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    features = [col for col in df.columns if col != 'SO2']
    
    df[features] = scaler_features.fit_transform(df[features])
    df['SO2'] = scaler_target.fit_transform(df[['SO2']])
    
    X, y = [], []
    for i in range(len(df) - Config.SEQ_LENGTH - Config.PRED_LENGTH + 1):
        X.append(df[features].iloc[i:i + Config.SEQ_LENGTH].values)
        y.append(df['SO2'].iloc[i + Config.SEQ_LENGTH:i + Config.SEQ_LENGTH + Config.PRED_LENGTH].values)
    
    return features, np.array(X), np.array(y)

def train_lstm(X, y, features, doc, file_name, english_name):
    """ฝึกโมเดล LSTM"""
    tscv = TimeSeriesSplit(n_splits=Config.SPLITS)
    model = create_lstm_model(len(features))
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(Config.BATCH_SIZE)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(Config.BATCH_SIZE)
        
        history = model.fit(train_dataset, epochs=Config.EPOCHS, validation_data=test_dataset, verbose=0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        save_plot(plt, f"loss_fold_{fold+1}_{english_name}", f"Fold {fold+1} Loss")

def create_lstm_model(input_dim):
    """สร้างโมเดล LSTM"""
    tf.keras.backend.clear_session()
    model = Sequential([
        LSTM(128, input_shape=(Config.SEQ_LENGTH, input_dim), return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(Config.PRED_LENGTH)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def analyze_feature_importance(df, features, doc, file_name, english_name):
    """วิเคราะห์ความสำคัญของฟีเจอร์ด้วย XGBoost"""
    X_flat = df[features].values
    y_flat = df['SO2'].values
    
    xgb_model = xgboost.XGBRegressor(random_state=Config.SEED_VALUE)
    xgb_model.fit(X_flat, y_flat)
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_flat[-100:])
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_flat[-100:], feature_names=features, plot_type="bar", show=False)
    save_plot(plt, f"shap_{english_name}", f"SHAP Importance - {english_name}")

def main():
    """ฟังก์ชันหลัก"""
    Config.OUTPUT_DIR.mkdir(exist_ok=True)
    setup_environment()
    
    for file_path in Config.INPUT_DIR.glob("*.csv"):
        process_file(file_path)

if __name__ == "__main__":
    main()