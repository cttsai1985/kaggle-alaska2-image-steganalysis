ARG BASE_IMAGE
FROM $BASE_IMAGE

LABEL maintainer "Chia-Ta Tsai <cttsai.aws@outlook.com>"

RUN ln -sf /usr/share/zoneinfo/US/Eastern /etc/localtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

########### APT
RUN apt-get update && apt-get install -y libgl1-mesa-dev

########### Anaconda
COPY requirements_conda.txt /tmp/requirements_conda.txt
RUN conda install -c conda-forge -c pytorch -c numba -c nvidia -c anaconda -y \
    --file /tmp/requirements_conda.txt

# clean up
COPY clean-layer.sh /tmp/clean-layer.sh
RUN bash /tmp/clean-layer.sh


########### pip
COPY requirements_pip.txt /tmp/requirements_pip.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /tmp/requirements_pip.txt

# clean up
COPY clean-layer.sh /tmp/clean-layer.sh
RUN bash /tmp/clean-layer.sh


########### General
EXPOSE 6006 8888

VOLUME ["/data"]
VOLUME ["/log"]
VOLUME ["/src"]


WORKDIR /root/src

# COPY jupyter_notebook_config.py /root/.jupyter/
# CMD ["sh", "-c", "jupyter notebook --port=8888 --no-browser --ip=* --allow-root"]
