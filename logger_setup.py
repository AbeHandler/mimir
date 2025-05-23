# adding a lgger
import logging
import os

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logger
logging.basicConfig(
    filename="logs/mylog.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
)

logger = logging.getLogger(__name__)