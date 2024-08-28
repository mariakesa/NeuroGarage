from dotenv import load_dotenv
load_dotenv()
from HuggingMouse.pipelines.pipeline_tasks import pipeline

from HuggingMouse.pipelines.single_trial_fs import MovieSingleTrialRegressionAnalysis
from transformers import ViTModel, ViTMAEModel, CLIPVisionModel
from sklearn.linear_model import Ridge

#model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

#image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

#ViTModel doesn't work
#model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
model = ViTModel.from_pretrained('facebook/dino-vitb8')

#model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

regr_model = Ridge(10)

pipe = pipeline("neural-activity-prediction",
                model=model,
                regression_model=regr_model,
                single_trial_f=MovieSingleTrialRegressionAnalysis(),
                test_set_size=0.25)
#511511001 and 646959386 are experiment container ID's.  
#pipe(565039910).dropna().scatter_movies().heatmap()
#posteriormedial visual cortex
a=[565039910, 575766605, 561463418, 574529963]
#Visl
#a=[564425775,557520762,558471484,547315012,583136565,584155531,573261513,559792042]
for i in a:
    pipe(i).dropna().scatter_movies().heatmap()
#pipe(646959386).dropna().scatter_movies().heatmap()