from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from landmark_detection.model import DenseNet

# Return object detector
def load_object_detector(model_path, model_name):
    # Load objects detector model
    cfg = get_cfg()
    #cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
    cfg.merge_from_file('./object_detection/model/config.yaml')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Set threshold for this model
    cfg.MODEL.WEIGHTS = model_path+model_name # Set path model .pth
    cfg.MODEL.DEVICE = ('cpu') # Set path model .pth
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    return DefaultPredictor(cfg)
    
# Return landmarks detector    
def load_landmarks_detector(model_path, model_name):
    model = DenseNet(dense_blocks=5, dense_layers=-1, growth_rate=8, dropout_rate=0.2,
                       bottleneck=True, compression=1.0, weight_decay=1e-4, depth=40)
    model.load_weights(model_path+model_name)
    return model