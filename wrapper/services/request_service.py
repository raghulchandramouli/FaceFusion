import requests

class ReqService:

    def __init__(self):

        #first post request to get the s3 uri
        self.post_url = "https://platform.authenta.ai/api/media"
        self.__authtoken = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOiI2OGUzNzEyMDIxMTU3NDg3ZGQyYTI5MTUiLCJpYXQiOjE3NTk4MzA4OTYsImV4cCI6MTc1OTkxNzI5Nn0.B83RCtzkQ1bWQT5ugon-SuC0XIZPJkWeAHStgw4sOoo"
        self.__headers = {
            "Authorization": f"Bearer {self.__authtoken}",
            "Content-Type": 'application/json',
        }
        self.s3_uri = None

    def get_s3_uri(self):
        data = {
        "name": "Gradio_test01",
        "contentType": "video/mp4",
        "size": 10485760,
        "modelType": "DF-1"
        }

        response = requests.post(self.post_url, json=data, headers=self.__headers)
        if response.status_code == 201:
            if response.json().get('uploadUrl'):
                s3_url = response.json().get('uploadUrl')
                return s3_url
            else:
                print("No uploadUrl found in the response")
                return None
        print(f"Request did not return status code 200, it returned {response.status_code} \n {response}")
        return None

    def upload_video(self, s3_uri, video_path):
        with open(video_path, 'rb') as f:
            response  = requests.put(s3_uri, data=f, headers={'Content-Type': 'video/mp4'})
        assert response.status_code == 200, f"Upload failed with status code {response.status_code}"

    def __call__(self, video_path):
        try:
            s3_uri = self.get_s3_uri()
            self.upload_video(s3_uri, video_path)
            return True
        except Exception as e:
            print(f"Error uploading video: {e}")
            return False

req_service = ReqService()

if __name__ == "__main__":
    pass