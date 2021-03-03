FROM continuumio/miniconda3

COPY environment.yml /tmp/environment.yml
WORKDIR /tmp
RUN conda env update -f environment.yml && \ 
    conda clean --all --yes && \
    rm /tmp/*
 

RUN mkdir -p /deploy/app
COPY m2p /deploy/app

WORKDIR /deploy/app
ENV PYTHONPATH "${PYTHONPATH}:/deploy/app"

#ENTRYPOINT "/bin/bash"
CMD python app.py --bind 0.0.0.0:$PORT
