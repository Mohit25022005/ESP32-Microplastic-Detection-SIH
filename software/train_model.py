#!/usr/bin/env python3
"""
Machine Learning Model Training for Microplastic Classification
ESP32-CAM Microplastic Detection System - SIH Hackathon

This script trains ML models to enhance particle classification accuracy
beyond the basic rule-based approach in the main detector.

Features:
- Feature extraction from particle measurements
- Multiple classification algorithms
- Model validation and optimization
- TensorFlow Lite model generation for ESP32
- Performance benchmarking

Revolutionary approach: Advanced ML at ₹4,000 cost point
"""

import numpy as np
import pandas as pd
import json
import logging
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available - deep learning models disabled")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    models_to_train: List[str] = None
    
    def __post_init__(self):
        if self.models_to_train is None:
            self.models_to_train = ['random_forest', 'gradient_boosting', 'svm', 'mlp']

class MicroplasticMLTrainer:
    """
    Machine Learning trainer for microplastic classification
    
    Enhances the basic ESP32-CAM detection with advanced ML models
    for improved accuracy while maintaining the revolutionary cost advantage
    """
    
    def __init__(self, config: TrainingConfig = None):
        """
        Initialize ML trainer
        
        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.best_model = None
        self.feature_names = [
            'area', 'perimeter', 'circularity', 'equivalent_diameter',
            'aspect_ratio', 'solidity', 'compactness', 'extent'
        ]
        
        logging.info("🤖 ML Trainer initialized for microplastic classification")
        logging.info(f"💰 Cost advantage maintained: ₹4,000 vs ₹50,00,000")
    
    def generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic training data for demonstration
        
        In real deployment, this would be replaced with actual labeled data
        from microscopy analysis or expert annotation
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic particle data
        """
        np.random.seed(self.config.random_state)
        
        data = []
        
        # Define typical characteristics for different particle types
        particle_types = {
            'spherical_microplastic': {
                'circularity': (0.8, 1.0),
                'aspect_ratio': (1.0, 1.5),
                'solidity': (0.9, 1.0),
                'size_range': (1, 500)
            },
            'fiber_microplastic': {
                'circularity': (0.1, 0.4),
                'aspect_ratio': (3.0, 10.0),
                'solidity': (0.7, 0.9),
                'size_range': (10, 1000)
            },
            'fragment_microplastic': {
                'circularity': (0.3, 0.7),
                'aspect_ratio': (1.5, 4.0),
                'solidity': (0.6, 0.9),
                'size_range': (5, 800)
            },
            'organic_debris': {
                'circularity': (0.2, 0.6),
                'aspect_ratio': (1.2, 3.0),
                'solidity': (0.4, 0.8),
                'size_range': (20, 2000)
            },
            'mineral_particle': {
                'circularity': (0.4, 0.8),
                'aspect_ratio': (1.0, 2.5),
                'solidity': (0.8, 1.0),
                'size_range': (1, 100)
            }
        }
        
        samples_per_type = n_samples // len(particle_types)
        
        for particle_type, characteristics in particle_types.items():
            for _ in range(samples_per_type):
                # Generate size (equivalent diameter)
                size_min, size_max = characteristics['size_range']
                equivalent_diameter = np.random.uniform(size_min, size_max)
                
                # Calculate area from diameter (with some noise)
                area = np.pi * (equivalent_diameter / 2) ** 2 * np.random.uniform(0.8, 1.2)
                
                # Generate circularity
                circ_min, circ_max = characteristics['circularity']
                circularity = np.random.uniform(circ_min, circ_max)
                
                # Calculate perimeter from area and circularity
                perimeter = np.sqrt(4 * np.pi * area / circularity) * np.random.uniform(0.9, 1.1)
                
                # Generate aspect ratio
                ar_min, ar_max = characteristics['aspect_ratio']
                aspect_ratio = np.random.uniform(ar_min, ar_max)
                
                # Generate solidity
                sol_min, sol_max = characteristics['solidity']
                solidity = np.random.uniform(sol_min, sol_max)
                
                # Calculate additional features
                compactness = (perimeter ** 2) / area if area > 0 else 0
                extent = area / (equivalent_diameter ** 2) * np.random.uniform(0.7, 1.0)
                
                sample = {
                    'area': area,
                    'perimeter': perimeter,
                    'circularity': circularity,
                    'equivalent_diameter': equivalent_diameter,
                    'aspect_ratio': aspect_ratio,
                    'solidity': solidity,
                    'compactness': compactness,
                    'extent': extent,
                    'particle_type': particle_type
                }
                
                data.append(sample)
        
        df = pd.DataFrame(data)
        logging.info(f"✅ Generated {len(df)} synthetic training samples")
        logging.info(f"📊 Class distribution: {df['particle_type'].value_counts().to_dict()}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for training
        
        Args:
            df: DataFrame with particle data
            
        Returns:
            Tuple of (features, labels)
        """
        # Extract features
        X = df[self.feature_names].values
        
        # Encode labels
        y = self.label_encoder.fit_transform(df['particle_type'])
        
        # Handle any missing values
        X = np.nan_to_num(X)
        
        logging.info(f"🔧 Prepared features: {X.shape}, labels: {y.shape}")
        logging.info(f"📋 Label classes: {list(self.label_encoder.classes_)}")
        
        return X, y
    
    def create_models(self) -> Dict:
        """
        Create different ML models for comparison
        
        Returns:
            Dictionary of model pipelines
        """
        models = {}
        
        if 'random_forest' in self.config.models_to_train:
            models['random_forest'] = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=self.config.random_state,
                    n_jobs=-1
                ))
            ])
        
        if 'gradient_boosting' in self.config.models_to_train:
            models['gradient_boosting'] = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=self.config.random_state
                ))
            ])
        
        if 'svm' in self.config.models_to_train:
            models['svm'] = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    random_state=self.config.random_state
                ))
            ])
        
        if 'mlp' in self.config.models_to_train:
            models['mlp'] = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    batch_size='auto',
                    learning_rate='constant',
                    learning_rate_init=0.001,
                    max_iter=1000,
                    random_state=self.config.random_state
                ))
            ])
        
        logging.info(f"🏗️ Created {len(models)} ML models for training")
        return models
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train all models and evaluate performance
        
        Args:
            X: Feature matrix
            y: Label vector
            
        Returns:
            Dictionary with model performance results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Create models
        models = self.create_models()
        results = {}
        
        logging.info("🎯 Starting model training and evaluation...")
        
        for name, model in models.items():
            logging.info(f"🔄 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.config.cv_folds,
                scoring='accuracy'
            )
            
            # Test predictions
            y_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            # Store results
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': test_accuracy,
                'classification_report': classification_report(
                    y_test, y_pred,
                    target_names=self.label_encoder.classes_,
                    output_dict=True
                )
            }
            
            logging.info(f"✅ {name}: CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}, "
                        f"Test={test_accuracy:.3f}")
        
        # Store trained models
        self.models = {name: result['model'] for name, result in results.items()}
        
        # Select best model
        best_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        self.best_model = self.models[best_name]
        
        logging.info(f"🏆 Best model: {best_name} (accuracy: {results[best_name]['test_accuracy']:.3f})")
        
        return results
    
    def create_tensorflow_model(self, input_dim: int, num_classes: int) -> Optional[tf.keras.Model]:
        """
        Create TensorFlow model for potential ESP32 deployment
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            
        Returns:
            Compiled TensorFlow model or None if TF not available
        """
        if not TF_AVAILABLE:
            logging.warning("TensorFlow not available - skipping neural network model")
            return None
        
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logging.info("🧠 Created TensorFlow neural network model")
        return model
    
    def convert_to_tflite(self, model: tf.keras.Model, save_path: str) -> bool:
        """
        Convert TensorFlow model to TensorFlow Lite for ESP32 deployment
        
        Args:
            model: Trained TensorFlow model
            save_path: Path to save TFLite model
            
        Returns:
            Success flag
        """
        if not TF_AVAILABLE:
            return False
        
        try:
            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Quantize for smaller size (important for ESP32)
            converter.representative_dataset = None
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
            tflite_model = converter.convert()
            
            # Save the model
            with open(save_path, 'wb') as f:
                f.write(tflite_model)
            
            logging.info(f"📱 TensorFlow Lite model saved: {save_path}")
            logging.info(f"💾 Model size: {len(tflite_model)} bytes")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ TFLite conversion failed: {e}")
            return False
    
    def save_models(self, save_dir: str = "models"):
        """
        Save trained models and preprocessing objects
        
        Args:
            save_dir: Directory to save models
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save best model
        if self.best_model:
            model_path = os.path.join(save_dir, "best_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            logging.info(f"💾 Best model saved: {model_path}")
        
        # Save label encoder
        encoder_path = os.path.join(save_dir, "label_encoder.pkl")
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save feature names
        features_path = os.path.join(save_dir, "feature_names.json")
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        logging.info(f"✅ All models and preprocessing objects saved to {save_dir}")
    
    def create_visualizations(self, results: Dict, save_dir: str = "plots"):
        """
        Create visualizations of model performance
        
        Args:
            results: Model training results
            save_dir: Directory to save plots
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Model comparison plot
        plt.figure(figsize=(12, 6))
        
        models = list(results.keys())
        cv_means = [results[m]['cv_mean'] for m in models]
        test_accs = [results[m]['test_accuracy'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, cv_means, width, label='CV Accuracy', alpha=0.8)
        plt.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison - ESP32 Microplastic Detection')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, "model_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"📊 Model comparison plot saved: {plot_path}")
        
        # Feature importance (for tree-based models)
        for name, result in results.items():
            if hasattr(result['model'].named_steps['classifier'], 'feature_importances_'):
                plt.figure(figsize=(10, 6))
                
                importances = result['model'].named_steps['classifier'].feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.bar(range(len(importances)), importances[indices])
                plt.xticks(range(len(importances)), 
                          [self.feature_names[i] for i in indices], 
                          rotation=45)
                plt.title(f'Feature Importance - {name}')
                plt.ylabel('Importance')
                plt.tight_layout()
                
                importance_path = os.path.join(save_dir, f"feature_importance_{name}.png")
                plt.savefig(importance_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logging.info(f"📊 Feature importance plot saved: {importance_path}")

def main():
    """
    Main training pipeline for microplastic classification models
    """
    logging.info("🚀 Starting ML training for ESP32 microplastic detection")
    logging.info("💰 Revolutionary approach: Advanced ML at ₹4,000 cost point")
    
    # Initialize trainer
    config = TrainingConfig()
    trainer = MicroplasticMLTrainer(config)
    
    # Generate training data (replace with real data in production)
    logging.info("📊 Generating synthetic training data...")
    df = trainer.generate_synthetic_data(n_samples=2000)
    
    # Prepare features
    X, y = trainer.prepare_features(df)
    
    # Train models
    results = trainer.train_models(X, y)
    
    # Print results summary
    print("\n" + "="*60)
    print("🤖 ML TRAINING RESULTS - ESP32 MICROPLASTIC DETECTION")
    print("="*60)
    print(f"💰 System Cost: ₹4,000 (100x cheaper than traditional methods)")
    print(f"📊 Training Samples: {len(df)}")
    print(f"🔧 Features: {len(trainer.feature_names)}")
    print(f"🏷️ Classes: {len(trainer.label_encoder.classes_)}")
    print("-"*60)
    
    for name, result in results.items():
        print(f"🎯 {name.upper()}:")
        print(f"   CV Accuracy: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")
        print(f"   Test Accuracy: {result['test_accuracy']:.3f}")
    
    print("="*60)
    
    # Create TensorFlow model for ESP32 deployment
    if TF_AVAILABLE:
        tf_model = trainer.create_tensorflow_model(
            input_dim=len(trainer.feature_names),
            num_classes=len(trainer.label_encoder.classes_)
        )
        
        if tf_model:
            # Train TensorFlow model
            X_scaled = trainer.models[list(trainer.models.keys())[0]].named_steps['scaler'].fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            tf_model.fit(X_train, y_train, epochs=50, verbose=0, validation_split=0.2)
            
            # Convert to TensorFlow Lite
            trainer.convert_to_tflite(tf_model, "models/microplastic_model.tflite")
    
    # Save models
    trainer.save_models()
    
    # Create visualizations
    trainer.create_visualizations(results)
    
    logging.info("✅ Training complete - Models ready for ESP32 deployment!")
    logging.info("🏆 Revolutionary achievement: Lab-grade ML at ₹4,000 cost")

if __name__ == "__main__":
    main()