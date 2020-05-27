FROM centos:latest
RUN yum install python36 -y
RUN python3 -m pip install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN yum install -y epel-release
RUN yum install -y python36-devel
RUN pip3 install keras 
RUN pip3 install pandas
RUN pip3 install tensorflow==1.14
RUN pip3 install matplotlib
ENTRYPOINT [ "python3" ]
CMD [ "/mlops/Fashion_model.py" ]
