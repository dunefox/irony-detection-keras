from tensorflow/tensorflow:latest-py3

WORKDIR /src/app

COPY . .

RUN apt-get install make \
    && apt-get install -y python3-venv \
    && pip install scikit-learn numpy Keras keras-self-attention

CMD make all
