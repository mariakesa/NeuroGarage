from dotenv import load_dotenv
load_dotenv()
from HuggingMouse.pipelines.pipeline_tasks import pipeline

from HuggingMouse.pipelines.single_trial_fs import MovieSingleTrialRegressionAnalysis
from transformers import ViTModel
from sklearn.linear_model import Ridge

model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
regr_model = Ridge(10)

pipe = pipeline("neural-activity-prediction",
                model=model,
                regression_model=regr_model,
                single_trial_f=MovieSingleTrialRegressionAnalysis(),
                test_set_size=0.25)
#511511001 and 646959386 are experiment container ID's.  
#pipe(565039910).dropna().scatter_movies().heatmap()
a=[565039910, 575766605, 561463418, 574529963]
for i in a:
    pipe(i).dropna().scatter_movies().heatmap()
#pipe(646959386).dropna().scatter_movies().heatmap()