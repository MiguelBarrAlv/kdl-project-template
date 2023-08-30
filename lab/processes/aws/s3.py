import boto3
import numpy as np
import io
import os

from dotenv import load_dotenv

class S3DatastoreManager:
    
    def __init__(self, bucket_name: str):          
        load_dotenv()
        self.access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region_name = os.getenv("AWS_REGION_NAME")
        self.s3_client = boto3.client('s3', aws_access_key_id=self.access_key, 
                                            aws_secret_access_key=self.secret_key, 
                                            region_name=self.region_name)
        self.bucket_name = bucket_name

    def _bucket_exists(self):
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            return True
        except:
            return False

    def upload_data(self, src_dir):
        if not self._bucket_exists():
            self._create_bucket()

        try:
            for root, _, files in os.walk(src_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    s3_path = os.path.join(filename)
                    self.s3_client.upload_file(file_path, self.bucket_name, s3_path)
            print("Upload successfully completed.")
            print(f"Data uploaded to: s3://{self.bucket_name}")
        except Exception as e:
            print("Error uploading data: ", e)

    def _create_bucket(self):
        try:
            self.s3_client.create_bucket(Bucket=self.bucket_name)
            print(f"Bucket '{self.bucket_name}' created successfully.")
        except Exception as e:
            print("Error creating bucket ", e)
            raise

    def load_data_from_s3(self, object_name):
        """Loads the data from AWS S3."""
        s3_object = self.s3_client.get_object(Bucket=self.bucket_name, Key=object_name)
        object_data = s3_object['Body'].read()
        return np.load(io.BytesIO(object_data))
