from torch import optim
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from load_data import MaxProbExtractor_yolov5, MeanProbExtractor_yolov5


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        self.img_dir = "../datasets/MAR20/train/images"
        self.lab_dir = "../datasets/MAR20/train/labels"
        self.weightfile = "weights/yolov5s6.pt"  # "weights/yolo.weights"
        self.printfile = "non_printability/30values.txt"
        self.patch_size = 50  # 50

        self.start_learning_rate = 0.03

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0

        self.batch_size = 1

        self.loss_target = lambda obj, cls: obj * cls  # # self.loss_target(obj, cls) return obj * cls


class Experiment1(BaseConfig):
    """
    Model that uses a maximum total variation, tv cannot go below this point.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment1'
        self.max_tv = 0.165


class Experiment2HighRes(Experiment1):
    """
    Higher res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 400
        self.patch_name = 'Exp2HighRes'

class Experiment3LowRes(Experiment1):
    """
    Lower res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 100
        self.patch_name = "Exp3LowRes"

class Experiment4ClassOnly(Experiment1):  # 仅考虑分类损失
    """
    Only minimise class score.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment4ClassOnly'
        self.loss_target = lambda obj, cls: cls




class Experiment1Desktop(Experiment1):
    """
    """

    def __init__(self):
        """
        Change batch size.
        """
        super().__init__()

        self.batch_size = 8
        self.patch_size = 400


class ReproducePaperObj(BaseConfig):
    """
    仅考虑目标置信度损失
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 4
        self.patch_size = 300

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj
        
        # yolov5
        # 15,exp:n, 16,19:s, 9:m, 17,24:l, 18,26:x
        self.weights_yolov5 = ' '
        self.device = select_device('')
        self.data = 'data/MAR20.yaml'
        self.img_size = 640
        self.imgsz = (640, 640)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 100  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS


class yolov5s(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        # 2080
        # self.weights_yolov5 = '/home/mnt/ljw305/yolov5/runs/train/yolov5s/weights/best.pt'
        # self.data = '/home/mnt/ljw305/yolov5/data/DOTA1_0.yaml'

        # 3080
        # self.weights_yolov5 = 'weights/best_20classes_50epochs_preVisdrone.pt'
        self.weights_yolov5 = 'weights/best_singleclass_preVisdrone.pt'
        self.data = 'data/MAR20.yaml'
        
        self.batch_size = 1
        self.start_learning_rate = 0.05
        self.patch_size = 120  # 50

        self.device = select_device('0')
        self.img_size = 640
        self.imgsz = (640, 640)
        self.conf_thres = 0.3 # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 50  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS

        self.model = DetectMultiBackend(self.weights_yolov5,
                                        device=self.device,
                                        dnn=False,
                                        data=self.data).eval()
        # self.prob_extractor = MaxProbExtractor_yolov5(0, 20, self.loss_target)
        self.prob_extractor = MeanProbExtractor_yolov5(0, 20, self.loss_target, self.conf_thres, self.iou_thres, self.max_det)

patch_configs = {
    "base": BaseConfig,
    "exp1": Experiment1,
    "exp1_des": Experiment1Desktop,
    "exp2_high_res": Experiment2HighRes,
    "exp3_low_res": Experiment3LowRes,
    "exp4_class_only": Experiment4ClassOnly,
    "paper_obj": ReproducePaperObj,
    "yolov5s": yolov5s,
}
