
        
import os
import logging

# 设置日志记录的级别和格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def capture_event(event_name, event_properties):
    if not os.environ.get('EVENT_CAPTURE') == "disable":
        logging.info(f"Event captured: {event_name} with properties: {event_properties}")
