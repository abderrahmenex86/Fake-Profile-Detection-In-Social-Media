import gc
import os

import cv2
import numpy as np
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import hog
from skimage.io import imread
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from fake_profile_detector.configs.general import BASE_DIR


class FeatureExtractor:
    def __init__(self, feature_type="raw_pixel", apply_pca=False, pca_components=10000, batch_size=500):
        self.feature_type = feature_type
        self.apply_pca = apply_pca
        self.pca_components = pca_components
        self.batch_size = batch_size
        self.pca = None
        self.scaler = StandardScaler()
        self.is_fitted = False

        if feature_type == "sift":
            self.sift = cv2.SIFT_create()
        elif feature_type == "surf":
            raise ValueError("SURF is not available in this OpenCV build. Use SIFT instead.")

    def extract_single_image(self, image_path):
        try:
            if self.feature_type == "raw_pixel":
                return self._extract_raw_pixels(image_path)
            elif self.feature_type == "hog":
                return self._extract_hog(image_path)
            elif self.feature_type == "sift":
                return self._extract_sift(image_path)
            elif self.feature_type == "surf":
                raise ValueError("SURF is not available in this OpenCV build. Use SIFT instead.")
            else:
                raise ValueError(f"Unknown feature type: {self.feature_type}")
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None

    def _resize_image_if_needed(self, img, target_size=(150, 200)):  # 150 height, 200 width
        """Resize image to consistent size and ensure 3 channels for raw pixels"""
        if self.feature_type == "raw_pixel":
            from skimage.transform import resize
            
            # Ensure 3 channels (RGB)
            if len(img.shape) == 2:  # Grayscale
                img = np.stack([img] * 3, axis=-1)
            elif len(img.shape) == 3 and img.shape[2] == 1:  # Single channel
                img = np.repeat(img, 3, axis=-1)
            elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
                img = img[:, :, :3]  # Keep only RGB channels
            
            # Resize to 150x200 (height x width)
            img = resize(img, target_size + (3,), anti_aliasing=True)
        return img

    def _extract_raw_pixels(self, image_path):
        img = imread(image_path)
        if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
            img = rgba2rgb(img)
        if len(img.shape) == 3:  # RGB - keep as is
            pass
        elif len(img.shape) == 2:  # Grayscale - will be converted to 3 channels in resize
            pass
        
        img = self._resize_image_if_needed(img)
        features = img.flatten().astype(np.float32)
        
        # Ensure exactly 90,000 features (150 * 200 * 3 = 90,000)
        expected_size = 150 * 200 * 3  # 90,000
        if len(features) != expected_size:
            # Pad or truncate to ensure consistent size
            if len(features) < expected_size:
                padded_features = np.zeros(expected_size, dtype=np.float32)
                padded_features[:len(features)] = features
                features = padded_features
            else:
                features = features[:expected_size]
        
        return features

    def _extract_hog(self, image_path):
        img = imread(image_path)
        if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
            img = rgba2rgb(img)
        if len(img.shape) == 3:  # RGB
            img = rgb2gray(img)
        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            visualize=False,
            feature_vector=True,
            channel_axis=None,
        )
        return features.astype(np.float32)

    def _extract_sift(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            # Return zeros for target feature size (let's aim for ~30k features)
            target_features = 30000
            return np.zeros(target_features, dtype=np.float32)
            
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract all keypoints and descriptors
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        
        # Target: 15k-45k features, let's aim for ~30k
        # Each SIFT descriptor is 128 dimensions
        # To get ~30k features: 30000/128 ≈ 234 descriptors
        target_descriptors = 234  # This gives us 234 * 128 = 29,952 features
        target_features = target_descriptors * 128  # 29,952 features

        if descriptors is None:
            return np.zeros(target_features, dtype=np.float32)

        if len(descriptors) >= target_descriptors:
            # Use the first target_descriptors descriptors
            features = descriptors[:target_descriptors].flatten()
        else:
            # Pad with zeros if we have fewer descriptors
            features = np.zeros(target_features, dtype=np.float32)
            actual_features = len(descriptors) * 128
            features[:actual_features] = descriptors.flatten()
        
        return features.astype(np.float32)

    def _get_pca_components(self, feature_dim):
        """Determine PCA components based on feature type and dimensions"""
        if self.pca_components is not None:
            return self.pca_components
            
        if self.feature_type == "raw_pixel":
            return 10000  # Raw pixel + PCA should have 10,000 features
        elif self.feature_type == "hog":
            return 10000  # HOG + PCA should have 10,000 features
        elif self.feature_type == "sift":
            # SIFT + PCA: 10,000 if features > 30k, else 5,000
            if feature_dim > 30000:
                return 10000
            else:
                return 5000
        else:
            return min(1000, feature_dim // 4)  # Default fallback

    def fit_transform(self, image_paths):
        print(f"Extracting {self.feature_type} features from {len(image_paths)} images using batch size {self.batch_size}...")

        # Extract raw features first without PCA
        if len(image_paths) > 1000 and self.feature_type == "raw_pixel":
            X, valid_indices = self._extract_features_incremental(image_paths)
        else:
            X, valid_indices = self._extract_features_standard(image_paths)
        
        # Only fit scaler here, not PCA
        print("Scaling features...")
        X = self.scaler.fit_transform(X)
        
        # Store original features for later train/test split
        self.is_fitted = True
        return X, valid_indices

    def fit_pca_and_transform(self, X_train, X_test=None):
        """Fit PCA only on training data and transform both train and test"""
        if not self.apply_pca:
            return X_train, X_test
            
        print("Applying PCA (fitted only on training data)...")
        n_components = self._get_pca_components(X_train.shape[1])
        
        self.pca = PCA(n_components=n_components)
        X_train_pca = self.pca.fit_transform(X_train)
        
        print(f"PCA reduced shape: {X_train_pca.shape}")
        if hasattr(self.pca, 'explained_variance_ratio_'):
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        if X_test is not None:
            X_test_pca = self.pca.transform(X_test)
            return X_train_pca, X_test_pca
        else:
            return X_train_pca, None

    def _extract_features_incremental(self, image_paths):
        """Extract features using incremental processing without PCA"""
        print("Using incremental processing for large dataset...")
        
        valid_indices = []
        all_features = []
        
        # Calculate total number of batches for progress tracking
        total_batches = (len(image_paths) + self.batch_size - 1) // self.batch_size
        
        for batch_idx, batch_start in enumerate(tqdm(range(0, len(image_paths), self.batch_size), 
                                                   desc="Processing batches", 
                                                   total=total_batches)):
            batch_end = min(batch_start + self.batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]
            
            batch_features = []
            batch_indices = []
            
            for i, image_path in enumerate(batch_paths):
                path = os.path.join(BASE_DIR, "x", image_path)
                features = self.extract_single_image(path)
                if features is not None:
                    batch_features.append(features)
                    batch_indices.append(batch_start + i)
            
            if batch_features:
                batch_X = np.array(batch_features, dtype=np.float32)
                all_features.append(batch_X)
                valid_indices.extend(batch_indices)
                
                del batch_features, batch_X
                gc.collect()
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{total_batches} batches, "
                      f"valid samples so far: {len(valid_indices)}")
        
        if not all_features:
            raise ValueError("No valid features extracted")
            
        # Concatenate all features
        print("Concatenating features from all batches...")
        X = np.vstack(all_features)
        
        print(f"Raw feature shape: {X.shape}")
        return X, valid_indices

    def _extract_features_standard(self, image_paths):
        """Extract features using standard processing without PCA"""
        all_features = []
        valid_indices = []
        
        for batch_start in range(0, len(image_paths), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//self.batch_size + 1}/{(len(image_paths) + self.batch_size - 1)//self.batch_size}")
            
            batch_features = []
            batch_valid_indices = []
            
            for i, image_path in enumerate(tqdm(batch_paths, desc=f"Batch {batch_start//self.batch_size + 1}")):
                path = os.path.join(BASE_DIR, "x", image_path)
                features = self.extract_single_image(path)
                if features is not None:
                    batch_features.append(features)
                    batch_valid_indices.append(batch_start + i)
            
            if batch_features:
                all_features.extend(batch_features)
                valid_indices.extend(batch_valid_indices)
            
            # Clear memory
            del batch_features, batch_valid_indices
            gc.collect()

        if not all_features:
            raise ValueError("No valid features extracted")

        print(f"Converting to numpy array...")
        X = np.array(all_features, dtype=np.float32)
        print(f"Raw feature shape: {X.shape}")
        
        del all_features
        gc.collect()
        
        return X, valid_indices

    def transform(self, image_paths):
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
            
        features_list = []
        valid_indices = []

        for i, image_path in enumerate(tqdm(image_paths)):
            features = self.extract_single_image(image_path)
            if features is not None:
                features_list.append(features)
                valid_indices.append(i)

        if not features_list:
            return np.array([]), []

        X = np.array(features_list, dtype=np.float32)
        X = self.scaler.transform(X)

        if self.apply_pca and self.pca is not None:
            X = self.pca.transform(X)

        return X, valid_indices
