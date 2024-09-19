import os
import sys

MONGODB_URL_KEY = "MONGODB_URL"


from us_visa.pipeline.training_pipeline import TrainPipeline

obj = TrainPipeline()
obj.run_pipeline()