import logging
import yaml

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)

def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    logger.info("config loaded: %s", config)
    logger.info("Gaze Control System starting. . .")


if __name__ == "__main__":
    main()