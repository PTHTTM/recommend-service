# Hybrid Recommendation System

Hệ thống gợi ý sản phẩm sử dụng kỹ thuật Hybrid kết hợp **Collaborative Filtering (ALS)** và **Content-Based (PhoBERT)** cho nền tảng thương mại điện tử.

## 🎯 Tổng quan

Hệ thống cung cấp 2 loại gợi ý chính:
- **Homepage**: Gợi ý cá nhân hóa dựa trên lịch sử người dùng (ALS)
- **Product Detail**: Gợi ý sản phẩm tương tự (Hybrid: 60% Content + 40% Collaborative)


## 📁 Cấu trúc thư mục

```
recommend-service/
├── models/                     # Trained ML models
│   ├── als_model.pkl          # ALS collaborative filtering
│   └── phobert_embeddings.pkl # PhoBERT content vectors
├── data/                      # Training datasets
│   ├── interactions.csv       # User-item interactions (weight: 1-3)
│   └── products.csv          # Product metadata
│
├── when_need/                
│   ├── check_gpu.py          # GPU availability check
│   ├── check_occur.py        # Co-occurrence analysis
│   └── length_of_text.py     # Text length statistics
├── load_data.py              # Data loading utilities
├── mf.py                     # ALS implementation
├── phoBERT_content.py        # PhoBERT embedding generation
├── eval.py                   # Model evaluation
├── evaluator.py              # Evaluation metrics
├── api.py                    # FastAPI service (Eureka-enabled)
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Cài đặt

### 1. Yêu cầu hệ thống
- Python 3.8+
- CUDA 11.x (optional, cho GPU acceleration)
- MySQL 8.0+ (nếu sử dụng DB thay vì CSV)

### 2. Cài đặt dependencies

```bash
# Tạo virtual environment
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
# hoặc 
myenv\Scripts\activate  # Windows

# Cài đặt packages
pip install -r requirements.txt
```

### 3. Kiểm tra GPU (optional)

```bash
python when_need/check_gpu.py
```

## 📊 Chuẩn bị dữ liệu

### Format file CSV

**interactions.csv**:
```csv
user_id,product_id,weight
user_001,prod_123,3
user_002,prod_456,1
```
- `weight`: 1 (click), 2 (add to cart), 3 (purchase)

**products.csv**:
```csv
product_id,name,category,brand,description
prod_123,iPhone 15,Smartphone,Apple,Điện thoại thông minh...
```

### Kiểm tra độ dài text (cho PhoBERT)

```bash
python when_need/length_of_text.py
```

Điều chỉnh `max_length` trong PhoBERT:
- 95th percentile < 256 → dùng `max_length=256`
- 95th percentile > 256 → tăng lên 384 hoặc 512

## 🔧 Training Models

### 1. Train ALS Model (Collaborative Filtering)

```bash
python mf.py
```

**Hyperparameters** (trong `mf.py`):
```python
AlternatingLeastSquares(
    n_factors=50,        # Số chiều latent vectors (50-100)
    n_iterations=15,     # Số epoch (10-20)
    reg_param=0.01,      # L2 regularization (0.001-0.1)
    alpha=40            # Confidence scaling (10-50)
)
```

**Output**: `models/als_model.pkl` (~50-200MB tùy dataset size)

### 2. Generate PhoBERT Embeddings (Content-Based)

```bash
python phoBERT_content.py
```

**Output**: `models/phobert_embeddings.pkl` (~500MB-2GB)

**Lưu ý**: 
- PhoBERT yêu cầu ~4GB RAM (CPU) hoặc ~2GB VRAM (GPU)
- Quá trình encode có thể mất 5-30 phút 

### 3. Evaluation

```bash
python eval.py
```

**Metrics**:
- Hit Rate@10: Tỷ lệ user có ít nhất 1 sản phẩm đúng trong top 10
- Precision@10: Độ chính xác trung bình
- NDCG@10: Ranking quality
- Coverage: Tỷ lệ sản phẩm được recommend
- Diversity: Đa dạng giữa các sản phẩm gợi ý

**Output**: 
- `evaluation_results.pkl`
- `evaluation_summary.csv`
- `evaluation_comparison.png` (biểu đồ so sánh)

**Tiêu chí đạt chuẩn**:
- Hit Rate > 50%: Tốt
- Precision > 10%: Chấp nhận được
- NDCG > 50%: Tốt
- Coverage > 30%: Đủ đa dạng

## 🌐 Chạy API Service

### 1. Cấu hình Eureka (optional)

Sửa trong `api.py`:
```python
eureka_client.init(
    eureka_server="http://localhost:8761/eureka",  # Eureka server URL
    app_name="RECOMMEND-SERVICE",
    instance_port=8888
)
```

### 2. Start service

```bash
python api.py
# hoặc
uvicorn api:app --host 0.0.0.0 --port 8888 --reload
```

### 3. Test endpoints

**Homepage recommendations** (guest):
```bash
curl "http://localhost:8888/recommend/homepage?top_k=10"
```

**Homepage recommendations** (logged-in):
```bash
curl "http://localhost:8888/recommend/homepage?user_id=user_001&top_k=10"
```

**Product detail recommendations** (hybrid):
```bash
curl "http://localhost:8888/recommend/product-detail/prod_123?user_id=user_001&top_k=10"
```

**Health check**:
```bash
curl "http://localhost:8888/health"
```

**Admin metrics**:
```bash
curl "http://localhost:8888/admin/metrics"
```

**Trigger training** (background job):
```bash
curl -X POST "http://localhost:8888/train" \
  -H "Content-Type: application/json" \
  -d '{"force_retrain_all": false}'
```

## 📈 Monitoring & Tuning

### ALS Hyperparameter Tuning

**Triệu chứng**: Hit Rate thấp (<30%)
- ✅ Tăng `alpha=60` (tăng confidence weighting)
- ✅ Tăng `n_factors=100` (tăng capacity)
- ✅ Thu thập thêm dữ liệu interaction

**Triệu chứng**: Coverage thấp (<10%)
- ✅ Giảm `reg_param=0.001` (giảm regularization)
- ✅ Tăng diversity trong loss function

**Triệu chứng**: Training chậm
- ✅ Giảm `n_iterations=10`
- ✅ Sử dụng sparse matrix operations (đã implement)

### PhoBERT Optimization

**Memory issues**:
- ✅ Giảm `BATCH_SIZE=8` (trong `phoBERT_content.py`)
- ✅ Giảm `max_length=128` nếu text ngắn

**Similarity không tốt**:
- ✅ Kiểm tra text preprocessing (xem `when_need/length_of_text.py`)
- ✅ Thử model khác: `vinai/phobert-large` (chậm hơn nhưng chính xác hơn)

### Co-occurrence Analysis

Kiểm tra xem ALS có học được patterns không:

```bash
python when_need/check_occur.py
```

Nếu "NO OVERLAP" nhiều → cần thêm dữ liệu hoặc giảm sparsity.

## 🔒 Production Deployment

### 1. Docker (recommended)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8888"]
```

```bash
docker build -t recommend-service .
docker run -p 8888:8888 -v $(pwd)/models:/app/models recommend-service
```

### 2. Environment Variables

```bash
export DB_USER=your_user
export DB_PASSWORD=your_password
export DB_HOST=localhost
export DB_NAME=ecommerce
export USE_DB=false  # true nếu dùng MySQL
```

### 3. Load Balancing

API tự động register với Eureka Server:
- Heartbeat mỗi 30s
- Timeout 90s trước khi bị remove
- Health check endpoint: `/health`

### 4. Hot-swapping Models

```bash
# Train models mới
python mf.py
python phoBERT_content.py

# API tự động reload sau khi training xong (via /train endpoint)
# Không cần restart service
```

## 🐛 Troubleshooting

### Issue: "Models not loaded"
```bash
# Kiểm tra models tồn tại
ls -lh models/
# Nếu thiếu, chạy lại training
python mf.py
python phoBERT_content.py
```

### Issue: "Product not found"
- Kiểm tra `product_id` format (phải là string)
- Kiểm tra product có trong `data/products.csv`

### Issue: PhoBERT OOM (Out of Memory)
```python
# Trong phoBERT_content.py, giảm batch size
BATCH_SIZE = 8  # hoặc 4
```






