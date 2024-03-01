FROM python:3.9
WORKDIR /home
COPY . .
ARG HUGGINGFACE_API_KEY

COPY ./requirements.txt .
COPY ./.env .
RUN pip install -r requirements.txt
RUN pip install -U git+https://github.com/huggingface/transformers.git
RUN pip install -U git+https://github.com/huggingface/peft.git
RUN pip install -U git+https://github.com/huggingface/accelerate.git

EXPOSE 50051