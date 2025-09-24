import config
import os

def main():
    print("Oil Spill Detection Application")
    print("Configuration loaded from:", os.path.abspath(config.DATA_DIR))
    print("Model directory:", os.path.abspath(config.MODEL_DIR))
    print("Debug mode:", config.DEBUG)

if __name__ == "__main__":
    main()
