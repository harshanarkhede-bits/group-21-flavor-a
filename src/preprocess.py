"""Feature engineering and preprocessing entrypoint."""
from src.features.engineer import engineer_features_pipeline

if __name__ == "__main__":
    engineer_features_pipeline()