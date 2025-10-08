FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY wrapper/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY facefusion/ ./facefusion/
COPY .assets/ ./.assets/
COPY wrapper/ ./wrapper/

RUN mkdir -p wrapper/temp_data

EXPOSE 7860

CMD ["python", "wrapper/main.py"]
