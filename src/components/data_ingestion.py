#read data from data sources 

import os
import sys
from src.exception import CustomException
import pandas as pd
from src.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_training import ModelTrainer
from src.components.model_training import ModelTrainerConfig



@dataclass
class DataIngestionConfig:
    # def __init__(self,train_data_path,test_data_path,raw_data_path):
    train_data_path : str=os.path.join('artifacts','train.csv')
    test_data_path : str=os.path.join('artifacts','test.csv')
    raw_data_path : str=os.path.join('artifacts','raw.csv')
    # def ret(self):
    #     return (
    #     self.train_data_path ,
    #     self.test_data_path,
    #     self.raw_data_path
    #     )

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info(f"Entered the data ingestion method")
        try:
            #here change if mongodb or other db used
            df = pd.read_csv('notebook\stud.csv')
            logging.info('Read dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok =True)
            df.to_csv(self.ingestion_config.raw_data_path,index =False,header=True)
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info(f"INGESTION DATA SUCCESSFUL !!!")


            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
            

if __name__=="__main__":
    obj =DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transforamtion  = DataTransformation()
    train_arr,test_arr,_ =data_transforamtion.initiate_data_transformation(train_data,test_data)
    print(train_arr,test_arr)

    logging.info("train_arr,test_arr passed into model trainer")
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))




    