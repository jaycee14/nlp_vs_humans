FROM continuumio/miniconda3:4.6.14-alpine

RUN /opt/conda/bin/pip install --upgrade pip

RUN mkdir /home/anaconda/app
WORKDIR /home/anaconda/app

# install some extra dependencies
COPY requirements.txt .
RUN /opt/conda/bin/pip install --no-cache-dir -r requirements.txt

# copy the superintendent code into the container
COPY ./superintendent ./superintendent

#copy the voila notebook, results extraction and results query scripts
COPY app.ipynb .
COPY extract_script.py .
COPY query_number.py .

CMD ["/opt/conda/bin/voila", "--no-browser","app.ipynb"]
