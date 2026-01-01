
import os
import shutil
import tempfile
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from XREPORT.server.utils.services.training.metrics import MaskedSparseCategoricalCrossentropy, MaskedAccuracy
from XREPORT.server.utils.services.training.model import build_xreport_model
from keras.models import load_model, Model
from keras import layers
import keras

def test_simple_custom_object():
    print("Testing simple custom object save/load...")
    try:
        model_input = layers.Input(shape=(10,))
        model_output = layers.Dense(1)(model_input)
        model = Model(model_input, model_output)
        
        loss = MaskedSparseCategoricalCrossentropy()
        metric = MaskedAccuracy()
        
        model.compile(loss=loss, metrics=[metric], optimizer='adam')
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "model.keras")
            model.save(filepath)
            print(f"Model saved to {filepath}")
            
            custom_objects = {
                "MaskedSparseCategoricalCrossentropy": MaskedSparseCategoricalCrossentropy,
                "MaskedAccuracy": MaskedAccuracy
            }
            loaded_model = load_model(filepath, custom_objects=custom_objects)
            print("Simple model loaded successfully.")
    except Exception as e:
        print(f"FAILED Simple Test: {e}")
        import traceback
        traceback.print_exc()

def test_full_model():
    print("\nTesting full XREPORT model save/load...")
    try:
        metadata = {
            "vocabulary_size": 100,
            "max_report_size": 20,
        }
        config = {
            "embedding_dims": 32,
            "attention_heads": 2,
            "num_encoders": 1,
            "num_decoders": 1,
            "freeze_img_encoder": False,
            "jit_compile": False,
            "target_LR": 0.001
        }
        
        model = build_xreport_model(metadata, config)
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "full_model.keras")
            model.save(filepath)
            print(f"Full model saved to {filepath}")
            
            # Need all custom objects
            from XREPORT.server.utils.services.training.serializer import ModelSerializer
            serializer = ModelSerializer()
            # We can use the serializer's method but let's just use load_model directly to see error trace
            # Actually let's use the serializer logic to be close to reality
            
            custom_objects = {
                "MaskedSparseCategoricalCrossentropy": MaskedSparseCategoricalCrossentropy,
                "MaskedAccuracy": MaskedAccuracy,
            }
            # Add others from serializer
            from XREPORT.server.utils.services.training.layers import (
                PositionalEmbedding, AddNorm, FeedForward, SoftMaxClassifier, 
                TransformerEncoder, TransformerDecoder
            )
            from XREPORT.server.utils.services.training.encoder import BeitXRayImageEncoder
            from XREPORT.server.utils.services.training.scheduler import WarmUpLRScheduler
            
            custom_objects.update({
                "PositionalEmbedding": PositionalEmbedding,
                "AddNorm": AddNorm,
                "FeedForward": FeedForward,
                "SoftMaxClassifier": SoftMaxClassifier,
                "TransformerEncoder": TransformerEncoder,
                "TransformerDecoder": TransformerDecoder,
                "BeitXRayImageEncoder": BeitXRayImageEncoder,
                "WarmUpLRScheduler": WarmUpLRScheduler
            })
            
            loaded_model = load_model(filepath, custom_objects=custom_objects)
            print("Full model loaded successfully.")
            
    except Exception as e:
        print(f"FAILED Full Model Test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_custom_object()
    test_full_model()
