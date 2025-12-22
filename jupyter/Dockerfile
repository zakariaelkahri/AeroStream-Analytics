FROM jupyter/datascience-notebook:python-3.11

WORKDIR /home/jovyan/work

COPY --chown=jovyan:users requirements.txt /home/jovyan/work/requirements.txt
RUN pip install --no-cache-dir -r /home/jovyan/work/requirements.txt \
    && pip install --no-cache-dir jupyterlab
