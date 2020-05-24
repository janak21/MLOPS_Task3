FROM mlops:v1
RUN mkdir /root/ML_DIR
VOLUME /root/ML_DIR
COPY . ./root/ML_DIR/
WORKDIR /root/ML_DIR
CMD ["python3","Fashion_model.py"]
