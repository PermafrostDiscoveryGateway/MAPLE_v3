

FROM ubuntu:18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh
RUN conda --version

RUN conda clean -a

RUN echo $CONDA_PREFIX

COPY environment_maple.yml .

COPY test.py .

RUN ls

RUN conda install -c conda-forge mamba

RUN conda env create -f environment_maple.yml

SHELL ["conda", "run", "-n", "maple", "/bin/bash", "-c"]

RUN python -m pip install --ignore-installed pyclowder


CMD ["conda", "run", "--no-capture-output", "-n", "maple", "python","-u", "test.py"]