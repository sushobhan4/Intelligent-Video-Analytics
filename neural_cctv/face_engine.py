"""
================================================================================
Neural CCTV Analyser - Face Recognition & ChromaDB Vector Store
================================================================================
Extracts 512-dimensional latent vectors from detected faces using Inception-ResNet
and indexes/matches identities locally via ChromaDB without cloud connectivity.
Supports automatic unknown face capture and interactive user enrollment.
"""

import os
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import chromadb
from chromadb.config import Settings
from facenet_pytorch import InceptionResnetV1, MTCNN


class FaceVectorEngine:
    """
    Local face recognition engine with 512-d embeddings and ChromaDB vector store.
    """

    def __init__(
        self,
        db_path: str = "./data/face_db",
        unknown_dir: str = "./data/unknown_faces",
        distance_threshold: float = 0.38,
        device: Optional[str] = None
    ):
        self.db_path = os.path.abspath(db_path)
        self.unknown_dir = os.path.abspath(unknown_dir)
        self.distance_threshold = distance_threshold

        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs(self.unknown_dir, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Initialize MTCNN Face Detector & Aligner
        self.mtcnn = MTCNN(
            image_size=160,
            margin=14,
            min_face_size=30,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=False
        )

        # 2. Initialize Inception-ResNet-V1 512-d Embedding Network
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # 3. Initialize Persistent ChromaDB Client with Cosine Space
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="household_faces",
            metadata={"hnsw:space": "cosine"}
        )

        # Cache of recently processed unknown faces to avoid saving duplicate snapshots
        self._recently_saved_unknowns: Dict[int, float] = {}

    @torch.no_grad()
    def get_embedding(self, face_rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Takes an RGB face crop or image, runs MTCNN detection/alignment,
        or falls back to direct normalized crop embedding, generating a 512-d vector.
        """
        try:
            pil_img = Image.fromarray(face_rgb)
            face_tensor = None
            try:
                face_tensor = self.mtcnn(pil_img)
            except Exception:
                face_tensor = None

            if face_tensor is None:
                # Fallback: if already a head crop, directly resize to 160x160 and standardize
                resized = cv2.resize(face_rgb, (160, 160), interpolation=cv2.INTER_AREA)
                tensor = torch.tensor(resized).permute(2, 0, 1).float()
                face_tensor = (tensor - 127.5) / 128.0

            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0)

            face_tensor = face_tensor.to(self.device)
            embedding = self.resnet(face_tensor)
            # L2-normalize vector for robust cosine similarity
            embedding = F.normalize(embedding, p=2, dim=1)
            return embedding.cpu().squeeze(0).numpy()
        except Exception:
            return None

    def identify_face(
        self,
        face_crop_bgr: np.ndarray,
        track_id: Optional[int] = None,
        timestamp_str: Optional[str] = None
    ) -> Tuple[bool, str, float, Optional[np.ndarray]]:
        """
        Matches a face crop against the local ChromaDB database.
        
        Returns:
            is_known (bool): True if matched within distance threshold.
            identity_name (str): Recognized name (e.g. 'Father') or 'Unknown Person'.
            distance (float): Cosine distance to nearest match (0.0 = identical, 1.0 = opposite).
            embedding (np.ndarray): The 512-d embedding vector.
        """
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return False, "Unknown Person", 1.0, None

        h, w = face_crop_bgr.shape[:2]
        if h < 25 or w < 25:
            return False, "Unknown Person", 1.0, None

        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        embedding = self.get_embedding(face_rgb)

        if embedding is None:
            return False, "Unknown Person", 1.0, None

        vector_list = embedding.tolist()

        # Query ChromaDB if we have registered profiles
        count = self.collection.count()
        if count > 0:
            results = self.collection.query(
                query_embeddings=[vector_list],
                n_results=1,
                include=["metadatas", "distances"]
            )

            if results and results["distances"] and len(results["distances"][0]) > 0:
                best_distance = float(results["distances"][0][0])
                best_meta = results["metadatas"][0][0]

                if best_distance <= self.distance_threshold:
                    matched_name = best_meta.get("name", "Unknown Person")
                    return True, matched_name, best_distance, embedding

        # If not matched or database is empty, handle as unknown
        now = datetime.now()
        ts = timestamp_str or now.strftime("%Y%m%d_%H%M%S")

        # Save snapshot for user labeling if track hasn't been saved in the last 10s
        should_save = True
        if track_id is not None:
            last_saved = self._recently_saved_unknowns.get(track_id, 0.0)
            if (now.timestamp() - last_saved) < 10.0:
                should_save = False

        if should_save:
            unknown_id = f"unknown_{ts}_track{track_id or 'na'}_{uuid.uuid4().hex[:6]}"
            snapshot_path = os.path.join(self.unknown_dir, f"{unknown_id}.jpg")
            cv2.imwrite(snapshot_path, face_crop_bgr)
            if track_id is not None:
                self._recently_saved_unknowns[track_id] = now.timestamp()

        return False, "Unknown Person", 1.0, embedding

    def enroll_face(
        self,
        name: str,
        image_bgr_or_path: Any,
        relation: str = "Household"
    ) -> bool:
        """
        Enrolls a new named face into ChromaDB.
        Accepts either an image path (str) or a BGR numpy array.
        """
        if isinstance(image_bgr_or_path, str):
            if not os.path.exists(image_bgr_or_path):
                raise FileNotFoundError(f"Image file not found: {image_bgr_or_path}")
            img_bgr = cv2.imread(image_bgr_or_path)
        else:
            img_bgr = image_bgr_or_path

        face_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        embedding = self.get_embedding(face_rgb)

        if embedding is None:
            return False

        doc_id = f"face_{uuid.uuid4().hex[:8]}"
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "name": name,
                "relation": relation,
                "enrolled_at": datetime.now().isoformat()
            }]
        )
        return True

    def list_known_faces(self) -> List[Dict[str, Any]]:
        """Returns all enrolled profiles in the vector database."""
        count = self.collection.count()
        if count == 0:
            return []
        data = self.collection.get(include=["metadatas"])
        faces = []
        for doc_id, meta in zip(data["ids"], data["metadatas"]):
            faces.append({
                "id": doc_id,
                "name": meta.get("name"),
                "relation": meta.get("relation"),
                "enrolled_at": meta.get("enrolled_at")
            })
        return faces

    def list_unidentified_snapshots(self) -> List[str]:
        """Lists pending unidentified face snapshots awaiting user labeling."""
        if not os.path.exists(self.unknown_dir):
            return []
        files = [
            os.path.join(self.unknown_dir, f)
            for f in os.listdir(self.unknown_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return files
