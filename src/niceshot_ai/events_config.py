from deep_sort_realtime.deepsort_tracker import DeepSort
from trackers import DeathTracker


cod_bo6_config = {
    "Kill": {"pre": 2,
             "post": 2,
             'tracker': DeepSort(max_age=3, n_init=1),
             'cls_label': 0,
             'conf_thres': 0.6,
             'confirm_event': True,
             'clip_eligible': True},

    "Medal": {'tracker': DeepSort(max_age=30, nms_max_overlap=0.01),
              'cls_label': 1,
              'conf_thres': 0.9,
              'confirm_event': True},

    "Death": {"pre": 2,
              "post": 1,
              'tracker': DeathTracker(max_age=18),
              'cls_label': 2,
              'conf_thres': 0.82,
              'clip_eligible': True}
}


cod_bo7_config = {
    "Kill": {"pre": 2,
             "post": 2,
             'tracker': DeepSort(max_age=3, n_init=1),
             'cls_label': 0,
             'conf_thres': 0.6,
             'confirm_event': True,
             'clip_eligible': True},

    "Medal": {'tracker': DeepSort(max_age=30, nms_max_overlap=0.01),
              'cls_label': 1,
              'conf_thres': 0.9,
              'confirm_event': True},

    "Death": {"pre": 2,
              "post": 1,
              'tracker': DeathTracker(max_age=18),
              'cls_label': 2,
              'conf_thres': 0.82,
              'clip_eligible': True}
}