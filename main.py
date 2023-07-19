import os
import settings
import torch
from loader.model_loader import load_model

from feature_operation import hook_feature,FeatureOperator
from visualize.report import generate_html_summary
from util.clean import clean

from torchdistill.common.main_util import init_distributed_mode
from torchdistill.common import yaml_util


fo = FeatureOperator()
distributed, device_ids = init_distributed_mode(4, 'env://')
config = yaml_util.load_yaml_file(os.path.expanduser(settings.YAML_FILE)) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_config = config['models']
student_model_config = models_config['student_model'] if 'student_model' in models_config else models_config['model']
ckpt_file_path = student_model_config['ckpt']
checkpoint = settings.MODEL_FILE

student_model = load_model(student_model_config, device, distributed, hook_feature, checkpoint)
student_model = student_model.eval()


############ STEP 1: feature extraction ###############
features, maxfeature = fo.feature_extraction(model=student_model) 

for layer_id,layer in enumerate(settings.FEATURE_NAMES):

############ STEP 2: calculating threshold ############
    thresholds = fo.quantile_threshold(features[layer_id],savepath="quantile.npy")

############ STEP 3: calculating IoU scores ###########
# if you want to visualize the activation map of concept detector, set vis=True and index of concept detectors
    tally_result = fo.tally(features[layer_id],thresholds,savepath="tally.csv", vis=False, num=0)

############ STEP 4: generating results ###############
    generate_html_summary(fo.data, layer,
                          tally_result=tally_result,
                          maxfeature=maxfeature[layer_id],
                          features=features[layer_id],
                          thresholds=thresholds)
    if settings.CLEAN:
        clean()