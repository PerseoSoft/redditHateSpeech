FROM continuumio/anaconda3:2021.05

WORKDIR /opt/notebooks/clustering

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "hateSpeech", "/bin/bash", "-c"]
RUN conda init bash
RUN source activate hateSpeech

RUN python -m spacy download es_core_news_lg

# Setup for Jupyter Notebook
RUN groupadd -g 1000 jupyter && \
useradd -g jupyter -m -s /bin/bash jupyter && \
echo "jupyter:jupyter" | chpasswd && \
/opt/conda/bin/conda clean -y --all

COPY entrypoint.sh /usr/local/bin
RUN chmod +x /usr/local/bin/entrypoint.sh
EXPOSE 8888
USER jupyter

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD []