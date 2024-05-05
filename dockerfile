#Deriving the latest base image
FROM python:3.9


#Labels as key value pair
LABEL Maintainer="mlops3"

# Any working directory can be chosen as per choice like '/' or '/home' etc
# i have chosen /usr/app/src
WORKDIR /usr/app/src

#to COPY the remote file at working directory in container
COPY ./ML/eda.py ./
COPY start.sh ./
COPY ./ML/model_creation.py ./
COPY ./ML/test_model.py ./
COPY requirements.txt ./
# Now the structure looks like this '/usr/app/src/test.py'

RUN pip install -r requirements.txt
#RUN eda.py
#RUN model_creation.py
#RUN test_model.py

#CMD instruction should be used to run the software
#contained by your image, along with any arguments.

CMD ["/bin/bash", "./start.sh"]

