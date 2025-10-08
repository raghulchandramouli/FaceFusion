import requests
import os

class ReqService:

    def __init__(self):
        self.post_url = "https://platform.authenta.ai/api/media"
        self.__authtoken = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOiI2OGUzNzEyMDIxMTU3NDg3ZGQyYTI5MTUiLCJpYXQiOjE3NTk4MzA4OTYsImV4cCI6MTc1OTkxNzI5Nn0.B83RCtzkQ1bWQT5ugon-SuC0XIZPJkWeAHStgw4sOoo"
        self.__headers = {
            "Authorization": f"Bearer {self.__authtoken}",
            "Content-Type": 'application/json',
        }

    def get_s3_uri(self, file_size, expected_deepfake=None, name=None):
        data = {
            "name": name or "Gradio_upload",
            "contentType": "video/mp4",
            "size": file_size,
            "modelType": "DF-1"
        }
        
        if expected_deepfake is not None:
            data["expectedDeepfake"] = expected_deepfake

        response = requests.post(self.post_url, json=data, headers=self.__headers)
        if response.status_code == 201:
            return response.json().get('uploadUrl')
        print(f"Failed to get S3 URI: {response.status_code} - {response.text}")
        return None

    def upload_video(self, s3_uri, video_path):
        with open(video_path, 'rb') as f:
            response = requests.put(s3_uri, data=f, headers={'Content-Type': 'video/mp4'})
        return response.status_code == 200

    def __call__(self, video_path, expected_deepfake=None):
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return False
        
        try:
            file_size = os.path.getsize(video_path)
            # Use the file name as the display name to ensure uniqueness and traceability
            display_name = os.path.basename(video_path)
            s3_uri = self.get_s3_uri(file_size, expected_deepfake, name=display_name)
            if not s3_uri:
                return False
            return self.upload_video(s3_uri, video_path)
        except Exception as e:
            print(f"Error uploading video: {e}")
            return False

req_service = ReqService()
