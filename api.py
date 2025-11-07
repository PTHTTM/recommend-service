
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Header
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import pickle
import numpy as np
import logging
from datetime import datetime
import asyncio
import uuid
import socket
import aiohttp
from pathlib import Path
import json

from mf import AlternatingLeastSquares
import py_eureka_client.eureka_client as eureka_client

# ============================================
# LOGGING CONFIGURATION
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================
import os

class ServiceConfig:
    """Configuration for service dependencies and model paths"""
    # Eureka configuration
    EUREKA_SERVER = os.getenv("EUREKA_SERVER", "http://eureka-server:8761/eureka")
    SERVICE_NAME = "recommend-service"
    SERVICE_PORT = 8086

    # Service dependencies (discovered via Eureka or direct URLs)
    PRODUCT_SERVICE_URL = "http://product-service"  # Will use Eureka service name
    ADMIN_SERVICE_URL = "http://admin-service"      # Will use Eureka service name

    # API endpoints for data export
    PRODUCT_EXPORT_ENDPOINT = "/internal/v1/admin/export/products"
    INTERACTION_EXPORT_ENDPOINT = "/internal/v1/admin/export/interactions"

    # Security configuration
    INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "a-very-secret-key-for-internal-communication")
    API_KEY_HEADER = "X-API-KEY"
    USER_ID_HEADER = "X-User-Id"

    # Model paths
    MODEL_DIR = Path("models")
    DATA_DIR = Path("data")
    ALS_MODEL_PATH = MODEL_DIR / "als_model.pkl"
    PHOBERT_EMBEDDINGS_PATH = MODEL_DIR / "phobert_embeddings.pkl"


    # Training configuration
    ALS_N_FACTORS = 50
    ALS_REGULARIZATION = 0.01
    ALS_N_ITERATIONS = 15
    ALS_ALPHA = 1.0

    # HTTP retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    REQUEST_TIMEOUT = 60  # seconds

config = ServiceConfig()

# ============================================
# PYDANTIC MODELS
# ============================================
class ProductData(BaseModel):
    """Product metadata for PhoBERT training"""
    product_id: str
    name: str
    description: str
    brand: str
    category: str

class InteractionData(BaseModel):
    """User-product interaction with weight"""
    user_id: str
    product_id: str
    weight: float
    type: str

class TrainingRequest(BaseModel):
    """Request body for training trigger"""
    force_retrain_all: bool = Field(default=False, description="Force complete retraining")
    model_version_tag: Optional[str] = Field(default=None, description="Version tag for models")

class RecommendationResponse(BaseModel):
    """Standard response format for recommendations"""
    product_ids: List[str]
    strategy: str
    count: int

# ============================================
# GLOBAL STATE
# ============================================
class ModelState:
    """Manages ML models and training state with thread-safe operations"""
    def __init__(self):
        self.als_model: Optional[AlternatingLeastSquares] = None
        self.phobert_embeddings: Optional[np.ndarray] = None
        self.product_ids: Optional[List[str]] = None
        self.models_loaded: bool = False
        self.last_training_run: Dict[str, Any] = {}
        self.current_training_job: Optional[str] = None
        self._lock = asyncio.Lock()

    async def update_models(self, new_als_model: Optional[AlternatingLeastSquares],
                           new_phobert_embeddings: Optional[np.ndarray],
                           new_product_ids: Optional[List[str]]):
        """Thread-safe hot-swap of models"""
        async with self._lock:
            if new_als_model:
                self.als_model = new_als_model
                logger.info("ALS model hot-swapped")

            if new_phobert_embeddings is not None and new_product_ids:
                self.phobert_embeddings = new_phobert_embeddings
                self.product_ids = new_product_ids
                logger.info("PhoBERT embeddings hot-swapped")

            self.models_loaded = bool(self.als_model or self.phobert_embeddings is not None)

model_state = ModelState()

# ============================================
# FASTAPI APP INITIALIZATION
# ============================================
app = FastAPI(
    title="Recommend Service",
    description="Hybrid Recommendation System API (ALS + PhoBERT)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================
# HTTP CLIENT WITH RETRY LOGIC
# ============================================
class HttpClient:
    """Async HTTP client with retry logic for inter-service communication"""

    @staticmethod
    async def fetch_with_retry(url: str, max_retries: int = config.MAX_RETRIES,
                               timeout: int = config.REQUEST_TIMEOUT) -> Dict:
        """
        Fetch data from URL with exponential backoff retry

        Args:
            url: Target URL
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds

        Returns:
            JSON response as dictionary

        Raises:
            HTTPException: If all retries fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                timeout_obj = aiohttp.ClientTimeout(total=timeout)
                async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                    logger.info(f"HTTP GET {url} (attempt {attempt + 1}/{max_retries})")

                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"Successfully fetched from {url}")
                            return data
                        else:
                            error_text = await response.text()
                            logger.warning(f"HTTP {response.status} from {url}: {error_text}")
                            last_error = f"HTTP {response.status}: {error_text}"

            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout}s"
                logger.warning(f"Timeout fetching {url} (attempt {attempt + 1})")

            except aiohttp.ClientError as e:
                last_error = str(e)
                logger.warning(f"Client error fetching {url}: {e}")

            except Exception as e:
                last_error = str(e)
                logger.error(f"Unexpected error fetching {url}: {e}")

            # Exponential backoff
            if attempt < max_retries - 1:
                wait_time = config.RETRY_DELAY * (2 ** attempt)
                logger.info(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        # All retries failed
        error_msg = f"Failed to fetch {url} after {max_retries} attempts: {last_error}"
        logger.error(error_msg)
        raise HTTPException(status_code=503, detail=error_msg)

http_client = HttpClient()

# ============================================
# STARTUP & SHUTDOWN
# ============================================
@app.on_event("startup")
async def startup():
    """Initialize service: Register with Eureka and load ML models"""
    try:
        logger.info("=" * 60)
        logger.info("Starting Recommend Service...")
        logger.info("=" * 60)

        # Get host info
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)

        logger.info(f"Host: {hostname} ({ip_address})")
        logger.info(f"Port: {config.SERVICE_PORT}")
        logger.info(f"Eureka Server: {config.EUREKA_SERVER}")

        # Register with Eureka Server
        await eureka_client.init_async(
            eureka_server=config.EUREKA_SERVER,
            app_name=config.SERVICE_NAME,
            instance_port=config.SERVICE_PORT,
            instance_ip=ip_address,
            instance_host=hostname,
            health_check_url=f"http://{ip_address}:{config.SERVICE_PORT}/health",
            status_page_url=f"http://{ip_address}:{config.SERVICE_PORT}/admin/metrics",
            renewal_interval_in_secs=30,
            duration_in_secs=90,
        )

        logger.info("Successfully registered with Eureka")

        # Create directories if not exist
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Load ALS model (Collaborative Filtering)
        try:
            if config.ALS_MODEL_PATH.exists():
                model_state.als_model = AlternatingLeastSquares.load(str(config.ALS_MODEL_PATH))
                logger.info(f"ALS loaded: {len(model_state.als_model.user_map)} users, "
                           f"{len(model_state.als_model.item_map)} items, "
                           f"{model_state.als_model.n_factors} factors")
            else:
                logger.warning("⚠ ALS model not found - Collaborative filtering disabled")
        except Exception as e:
            logger.error(f"✗ Failed to load ALS model: {e}")

        # Load PhoBERT embeddings (Content-based)
        try:
            if config.PHOBERT_EMBEDDINGS_PATH.exists():
                with open(config.PHOBERT_EMBEDDINGS_PATH, 'rb') as f:
                    data = pickle.load(f)
                model_state.phobert_embeddings = data['embeddings']
                model_state.product_ids = data['product_ids']
                logger.info(f"PhoBERT loaded: {len(model_state.product_ids)} products, "
                           f"dim={model_state.phobert_embeddings.shape[1]}")
            else:
                logger.warning("⚠ PhoBERT embeddings not found - Content-based disabled")
        except Exception as e:
            logger.error(f"✗ Failed to load PhoBERT embeddings: {e}")

        model_state.models_loaded = bool(
            model_state.als_model or model_state.phobert_embeddings is not None
        )

        status = "READY" if model_state.models_loaded else "DEGRADED"
        logger.info("=" * 60)
        logger.info(f"Service {status}: Models loaded = {model_state.models_loaded}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"✗ Startup error: {e}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown():
    """Graceful shutdown - deregister from Eureka"""
    try:
        logger.info("Shutting down, deregistering from Eureka...")
        await eureka_client.stop_async()
        logger.info("Deregistered from Eureka")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# ============================================
# RECOMMENDATION HELPER FUNCTIONS
# ============================================
def get_content_recommendations(product_id: str, top_k: int = 10) -> List[str]:
    """
    Content-based recommendations using PhoBERT embeddings

    Args:
        product_id: Target product ID
        top_k: Number of recommendations

    Returns:
        List of recommended product IDs

    Raises:
        HTTPException: If PhoBERT unavailable or product not found
    """
    if model_state.phobert_embeddings is None:
        raise HTTPException(status_code=503, detail="PhoBERT model unavailable")

    if product_id not in model_state.product_ids:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found in embeddings")

    try:
        # Calculate cosine similarity
        prod_idx = model_state.product_ids.index(product_id)
        target_vec = model_state.phobert_embeddings[prod_idx]

        # Normalize vectors for cosine similarity
        target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-8)
        all_norms = model_state.phobert_embeddings / (
            np.linalg.norm(model_state.phobert_embeddings, axis=1, keepdims=True) + 1e-8
        )

        similarities = np.dot(all_norms, target_norm)

        # Get top-k similar products (exclude self)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        recommendations = [model_state.product_ids[i] for i in top_indices]

        logger.info(f"Content-based: {product_id} -> {len(recommendations)} recommendations")
        return recommendations

    except Exception as e:
        logger.error(f"Content-based error for product {product_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Content-based recommendation failed: {str(e)}")

def get_collaborative_recommendations(user_id: str, top_k: int = 10) -> Optional[List[str]]:
    """
    Personalized recommendations using ALS collaborative filtering

    Args:
        user_id: Target user ID
        top_k: Number of recommendations

    Returns:
        List of recommended product IDs or None if unavailable
    """
    if model_state.als_model is None:
        return None

    if not model_state.als_model.has_user_personalization():
        return None

    try:
        recommendations = model_state.als_model.predict_for_user(user_id, top_k=top_k)
        logger.info(f"Collaborative: User {user_id} -> {len(recommendations) if recommendations else 0} recommendations")
        return recommendations
    except Exception as e:
        logger.warning(f"Collaborative failed for user {user_id}: {e}")
        return None

def get_similar_items_collaborative(product_id: str, top_k: int = 10) -> Optional[List[str]]:
    """
    Item-item similarity using ALS item factors

    Args:
        product_id: Target product ID
        top_k: Number of similar items

    Returns:
        List of similar product IDs or None if unavailable
    """
    if model_state.als_model is None:
        return None

    try:
        similar_items = model_state.als_model.get_similar_items(product_id, top_k=top_k)
        logger.info(f"Item-similarity: {product_id} -> {len(similar_items) if similar_items else 0} items")
        return similar_items
    except Exception as e:
        logger.warning(f"Item similarity failed for {product_id}: {e}")
        return None

def hybrid_merge(content_list: List[str], collab_list: Optional[List[str]],
                 content_weight: float = 0.6, top_k: int = 10) -> List[str]:
    """
    Merge content-based and collaborative results with weighted strategy

    Strategy: 60% content-based + 40% collaborative (default)

    Args:
        content_list: Content-based recommendations
        collab_list: Collaborative recommendations
        content_weight: Weight for content-based (0-1)
        top_k: Number of final recommendations

    Returns:
        Merged list of product IDs
    """
    if not collab_list:
        logger.info(f"Hybrid: No collaborative data, using 100% content-based")
        return content_list[:top_k]

    # Calculate split: e.g., 60% content, 40% collaborative
    num_content = int(top_k * content_weight)
    num_collab = top_k - num_content
    result = []
    seen = set()

    # Add from content-based
    for pid in content_list[:num_content]:
        if pid not in seen:
            result.append(pid)
            seen.add(pid)

    # Add from collaborative
    for pid in collab_list[:num_collab]:
        if pid not in seen and len(result) < top_k:
            result.append(pid)
            seen.add(pid)

    # Fill remaining slots from content (if collaborative had fewer items)
    for pid in content_list[num_content:]:
        if pid not in seen and len(result) < top_k:
            result.append(pid)
            seen.add(pid)

    logger.info(f"Hybrid merge: {len(result)} items ({num_content} content + {num_collab} collaborative)")
    return result[:top_k]

def get_popular_fallback(top_k: int = 10) -> List[str]:
    """
    Fallback: return popular items based on ALS item factor norms

    Args:
        top_k: Number of popular items

    Returns:
        List of popular product IDs
    """
    if model_state.als_model is None:
        # Ultimate fallback: first N products from embeddings
        if model_state.product_ids:
            logger.warning("Popular fallback: Using first N products")
            return model_state.product_ids[:top_k]
        return []

    try:
        # Estimate popularity from item factor magnitudes
        item_popularity = {}
        for item_id in model_state.als_model.item_map.keys():
            item_idx = model_state.als_model.item_map[item_id]
            popularity = np.linalg.norm(model_state.als_model.item_factors[item_idx])
            item_popularity[item_id] = popularity

        sorted_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
        popular = [item_id for item_id, _ in sorted_items[:top_k]]

        logger.info(f"Popular fallback: {len(popular)} items")
        return popular

    except Exception as e:
        logger.error(f"Popular fallback error: {e}")
        return model_state.product_ids[:top_k] if model_state.product_ids else []

# ============================================
# RECOMMENDATION ENDPOINTS - INTERNAL API
# ============================================
@app.get("/api/v1/internal/recommend/homepage", response_model=RecommendationResponse)
async def recommend_homepage(
    user_id: int = Query(None, description="User id"),
    # user_id: Optional[str] = Header(None, alias="X-USER-ID", description="User ID for hybrid strategy"),
    top_k: int = Query(10, ge=1, le=50, description="Number of recommendations")
) -> Dict[str, Any]:
    """
    Homepage recommendations endpoint (called by API Gateway / Homepage Service)

    Strategy:
    - Logged-in users: Personalized via ALS collaborative filtering
    - Guest users: Popular items fallback

    Response format for Feign Client:
    {
        "product_ids": ["123", "456", ...],
        "strategy": "personalized" | "popular",
        "count": 10
    }
    """
    logger.info(f"Homepage request: user_id={user_id}, top_k={top_k}")

    if not model_state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    # Try personalized recommendations for logged-in users
    if user_id:
        try:
            recs = get_collaborative_recommendations(user_id, top_k)
            if recs and len(recs) > 0:
                return {
                    "product_ids": recs,
                    "strategy": "personalized",
                    "count": len(recs)
                }
        except Exception as e:
            logger.warning(f"Collaborative failed for user {user_id}: {e}")

    # Fallback to popular items
    popular_items = get_popular_fallback(top_k)
    logger.info(popular_items)
    return {
        "product_ids": popular_items,
        "strategy": "popular",
        "count": len(popular_items)
    }

@app.get("/api/v1/internal/recommend/product-detail/{product_id}",
         response_model=RecommendationResponse)
async def recommend_product_detail(
    product_id: str,
    user_id: int = Query(None, description="User id"),
    # user_id: Optional[str] = Header(None, alias="X-USER-ID", description="User ID for hybrid strategy"),
    top_k: int = Query(10, ge=1, le=50, description="Number of recommendations")
) -> Dict[str, Any]:
    """
    Product detail recommendations endpoint (called by Product Service)

    Strategy:
    - Guest users: 100% Content-based (PhoBERT similarity)
    - Logged-in users: Hybrid (60% Content + 40% Collaborative)

    Response format for Feign Client:
    {
        "product_ids": ["789", "101", ...],
        "strategy": "content-based" | "hybrid",
        "count": 10
    }
    """
    logger.info(f"Product detail request: product_id={product_id}, user_id={user_id}, top_k={top_k}")

    if not model_state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    # Always get content-based recommendations (required baseline)
    try:
        content_recs = get_content_recommendations(product_id, top_k=top_k)
    except HTTPException as e:
        if e.status_code == 404:
            logger.warning(f"Product {product_id} not found in embeddings")
            raise HTTPException(
                status_code=404,
                detail=f"Product {product_id} not found in recommendation system"
            )
        raise

    # Guest users: content-based only
    logger.info("content-based only")
    logger.info(content_recs)
    if not user_id:
        return {
            "product_ids": content_recs,
            "strategy": "content-based",
            "count": len(content_recs)
        }

    # Logged-in users: try hybrid strategy
    try:
        collab_recs = get_similar_items_collaborative(product_id, top_k=top_k)
        if collab_recs and len(collab_recs) > 0:
            hybrid_recs = hybrid_merge(
                content_recs, collab_recs,
                content_weight=0.6,
                top_k=top_k
            )
            return {
                "product_ids": hybrid_recs,
                "strategy": "hybrid",
                "count": len(hybrid_recs)
            }
    except Exception as e:
        logger.warning(f"Hybrid strategy failed, falling back to content-based: {e}")

    # Fallback to content-based if collaborative unavailable
    logger.info("collaborative unavailable")
    logger.info(content_recs)
    return {
        "product_ids": content_recs,
        "strategy": "content-based",
        "count": len(content_recs)
    }

# ============================================
# TRAINING PIPELINE
# ============================================
async def fetch_training_data() -> tuple[List[ProductData], List[InteractionData]]:
    """
    Fetch training data from Product Service and Admin/Order Service

    Returns:
        Tuple of (products, interactions)

    Raises:
        HTTPException: If data fetch fails
    """
    logger.info("=" * 60)
    logger.info("Fetching training data from services...")
    logger.info("=" * 60)

    # Fetch products from Product Service
    product_url = f"{config.PRODUCT_SERVICE_URL}{config.PRODUCT_EXPORT_ENDPOINT}"
    products_raw = await http_client.fetch_with_retry(product_url)

    products = [ProductData(**p) for p in products_raw]
    logger.info(f"Fetched {len(products)} products")

    # Fetch interactions from Admin/Order Service
    interaction_url = f"{config.ADMIN_SERVICE_URL}{config.INTERACTION_EXPORT_ENDPOINT}"
    interactions_raw = await http_client.fetch_with_retry(interaction_url)

    interactions = [InteractionData(**i) for i in interactions_raw]
    logger.info(f"Fetched {len(interactions)} interactions")

    return products, interactions

def train_als_model(interactions: List[InteractionData]) -> AlternatingLeastSquares:
    """
    Train ALS collaborative filtering model

    Args:
        interactions: User-product interaction data with weights

    Returns:
        Trained ALS model
    """
    logger.info("Training ALS model...")

    # Convert to format: [(user_id, product_id, weight), ...]
    interaction_tuples = [
        (i.user_id, i.product_id, i.weight)
        for i in interactions
    ]

    # Initialize and train ALS
    als = AlternatingLeastSquares(
        n_factors=config.ALS_N_FACTORS,
        regularization=config.ALS_REGULARIZATION,
        n_iterations=config.ALS_N_ITERATIONS,
        alpha=config.ALS_ALPHA
    )

    als.fit(interaction_tuples)

    logger.info(f"ALS trained: {len(als.user_map)} users, {len(als.item_map)} items")
    return als

def generate_phobert_embeddings(products: List[ProductData]) -> tuple[np.ndarray, List[str]]:
    """
    Generate PhoBERT embeddings for products (PLACEHOLDER - needs transformers library)

    Args:
        products: Product metadata

    Returns:
        Tuple of (embeddings matrix, product_ids list)
    """
    logger.info("Generating PhoBERT embeddings...")

    # TODO: Implement actual PhoBERT embedding generation
    # from transformers import AutoModel, AutoTokenizer
    # model = AutoModel.from_pretrained("vinai/phobert-base")
    # tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    # For now: simulate with random embeddings (768-dim like BERT)
    n_products = len(products)
    embedding_dim = 768

    # Concatenate text fields for each product
    product_texts = [
        f"{p.name} {p.description} {p.brand} {p.category}"
        for p in products
    ]

    # PLACEHOLDER: Replace with actual PhoBERT inference
    embeddings = np.random.randn(n_products, embedding_dim).astype(np.float32)

    # Normalize embeddings
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    product_ids = [p.product_id for p in products]

    logger.info(f"Generated embeddings: {embeddings.shape}")
    logger.warning("⚠ Using random embeddings - implement actual PhoBERT inference!")

    return embeddings, product_ids

async def run_training_pipeline(job_id: str, request: TrainingRequest):
    """
    Background task: Complete training pipeline

    1. Fetch data from Product Service and Admin/Order Service
    2. Train ALS model
    3. Generate PhoBERT embeddings
    4. Hot-swap models without downtime
    5. Save models to disk

    Args:
        job_id: Unique job identifier
        request: Training configuration
    """
    start_time = datetime.utcnow()
    logger.info("=" * 60)
    logger.info(f"Training Job {job_id} STARTED")
    logger.info("=" * 60)

    try:
        # Step 1: Fetch training data
        products, interactions = await fetch_training_data()

        if len(products) == 0:
            raise ValueError("No products fetched from Product Service")
        if len(interactions) == 0:
            raise ValueError("No interactions fetched from Admin/Order Service")

        # Step 2: Train ALS model (blocking, but run in executor to not block event loop)
        loop = asyncio.get_event_loop()
        new_als_model = await loop.run_in_executor(None, train_als_model, interactions)

        # Step 3: Generate PhoBERT embeddings
        new_embeddings, new_product_ids = await loop.run_in_executor(
            None, generate_phobert_embeddings, products
        )

        # Step 4: Save models to disk
        logger.info("Saving models to disk...")
        new_als_model.save(str(config.ALS_MODEL_PATH))

        with open(config.PHOBERT_EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump({
                'embeddings': new_embeddings,
                'product_ids': new_product_ids
            }, f)

        logger.info("Models saved to disk")

        # Step 5: Hot-swap models (no downtime)
        await model_state.update_models(
            new_als_model=new_als_model,
            new_phobert_embeddings=new_embeddings,
            new_product_ids=new_product_ids
        )

        # Calculate duration
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        # Update training status
        model_state.last_training_run.update({
            "completed_at": end_time.isoformat() + "Z",
            "status": "COMPLETED_SUCCESS",
            "duration_seconds": duration,
            "metrics": {
                "products_count": len(products),
                "interactions_count": len(interactions),
                "als_users": len(new_als_model.user_map),
                "als_items": len(new_als_model.item_map),
                "phobert_embeddings": len(new_product_ids)
            }
        })
        model_state.current_training_job = None

        logger.info("=" * 60)
        logger.info(f"Training Job {job_id} COMPLETED in {duration:.2f}s")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"✗ Training Job {job_id} FAILED")
        logger.error(f"Error: {e}", exc_info=True)
        logger.error("=" * 60)

        model_state.last_training_run.update({
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "status": "FAILED",
            "error": str(e),
            "duration_seconds": (datetime.utcnow() - start_time).total_seconds()
        })
        model_state.current_training_job = None

# ============================================
# ADMIN ENDPOINTS
# ============================================
@app.post("/api/v1/internal/recommend/train", status_code=202)
async def trigger_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Trigger model retraining (runs in background)

    This endpoint is called by API Gateway when an admin initiates retraining.
    The training process runs asynchronously to avoid blocking.

    Args:
        request: Training configuration
        background_tasks: FastAPI background tasks

    Returns:
        Job status with job_id

    Raises:
        HTTPException 409: If training already running
    """
    if model_state.current_training_job:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Training already running",
                "job_id": model_state.current_training_job,
                "started_at": model_state.last_training_run.get("started_at")
            }
        )

    job_id = str(uuid.uuid4())
    model_state.current_training_job = job_id

    model_state.last_training_run = {
        "job_id": job_id,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "status": "RUNNING",
        "force_retrain_all": request.force_retrain_all,
        "model_version_tag": request.model_version_tag
    }

    logger.info(f"Training job {job_id} queued (force_retrain={request.force_retrain_all})")
    background_tasks.add_task(run_training_pipeline, job_id, request)

    return {
        "status": "Training started",
        "job_id": job_id,
        "message": "Training process is running in background. Check /admin/metrics for progress."
    }

@app.get("/admin/metrics")
async def get_metrics():
    """
    Service metrics and model status for monitoring

    Returns comprehensive information about:
    - Service health status
    - Loaded models and their statistics
    - Last training run details
    - Current training job (if running)
    """
    return {
        "service_status": "ONLINE" if model_state.models_loaded else "DEGRADED",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "eureka_registered": True,
        "last_training_run": model_state.last_training_run or {
            "status": "NO_TRAINING_YET",
            "message": "No training has been executed since service start"
        },
        "current_training_job": model_state.current_training_job,
        "active_models": {
            "als_model": {
                "status": "loaded" if model_state.als_model else "unavailable",
                "users": len(model_state.als_model.user_map) if model_state.als_model else 0,
                "items": len(model_state.als_model.item_map) if model_state.als_model else 0,
                "latent_factors": model_state.als_model.n_factors if model_state.als_model else 0,
                "has_user_personalization": (
                    model_state.als_model.has_user_personalization()
                    if model_state.als_model else False
                )
            },
            "phobert_model": {
                "status": "loaded" if model_state.phobert_embeddings is not None else "unavailable",
                "total_items": len(model_state.product_ids) if model_state.product_ids else 0,
                "embedding_dim": (
                    model_state.phobert_embeddings.shape[1]
                    if model_state.phobert_embeddings is not None else 0
                )
            }
        },
        "configuration": {
            "service_name": config.SERVICE_NAME,
            "service_port": config.SERVICE_PORT,
            "eureka_server": config.EUREKA_SERVER,
            "als_factors": config.ALS_N_FACTORS,
            "als_iterations": config.ALS_N_ITERATIONS
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint for Eureka and Load Balancer

    Status:
    - HEALTHY: At least one model is loaded
    - UNHEALTHY: No models loaded

    Returns:
        503 status if unhealthy, 200 if healthy
    """
    if not model_state.models_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "status": "UNHEALTHY",
                "error": "No models loaded",
                "message": "Service is starting or models failed to load"
            }
        )

    models_loaded = []
    if model_state.als_model:
        models_loaded.append("als")
    if model_state.phobert_embeddings is not None:
        models_loaded.append("phobert")

    return {
        "status": "HEALTHY",
        "models_loaded": models_loaded,
        "eureka_registered": True,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.get("/")
async def root():
    """
    Service info and available endpoints

    Provides an overview of the service for developers and documentation.
    """
    return {
        "service": "Recommend Service",
        "version": "1.0.0",
        "description": "Hybrid Recommendation System (ALS + PhoBERT)",
        "status": "online" if model_state.models_loaded else "initializing",
        "eureka_registered": True,
        "models": {
            "als": "loaded" if model_state.als_model else "unavailable",
            "phobert": "loaded" if model_state.phobert_embeddings is not None else "unavailable"
        },
        "endpoints": {
            "internal_api": {
                "homepage": "/api/v1/internal/recommend/homepage",
                "product_detail": "/api/v1/internal/recommend/product-detail/{productId}"
            },
            "admin_api": {
                "trigger_training": "/api/v1/internal/recommend/train",
                "metrics": "/admin/metrics"
            },
            "monitoring": {
                "health": "/health",
                "docs": "/docs",
                "redoc": "/redoc"
            }
        },
        "documentation": {
            "swagger_ui": f"http://localhost:{config.SERVICE_PORT}/docs",
            "redoc": f"http://localhost:{config.SERVICE_PORT}/redoc"
        }
    }

# ============================================
# ERROR HANDLERS
# ============================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom handler for HTTP exceptions with detailed logging"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - Path: {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Catch-all handler for unexpected errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url.path)
        }
    )

# ============================================
# MAIN ENTRY POINT
# ============================================
if __name__ == "__main__":
    import uvicorn

    # Run the service
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.SERVICE_PORT,
        log_level="info",
        access_log=True
    )
