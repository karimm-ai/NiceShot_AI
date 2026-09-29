from .events_config import *
from .charts_config import *

supported_games = {"call of duty: black ops 6": cod_bo6_config,
         "call of duty: black ops 7": cod_bo7_config}


models_paths = {"call of duty: black ops 6": "game_models/yolov8n-cod_bo6.pt",
         "call of duty: black ops 7": "game_models/yolov11n-cod_bo7.pt"}


available_charts = {"call of duty: black ops 6": cod_bo6_chart_config,
         "call of duty: black ops 7": cod_bo7_chart_config}

