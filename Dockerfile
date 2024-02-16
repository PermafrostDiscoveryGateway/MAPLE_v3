FROM ubuntu:22.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
RUN apt-get install -y sudo

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh
RUN conda --version

RUN conda clean -a

RUN echo $CONDA_PREFIX

COPY environment_maple.yml .

COPY config.py .

COPY hyp_best_train_weights_final.h5 .

COPY maple_workflow.py .

COPY mpl_clean_inference.py .

COPY mpl_divideimg_234_water_new.py .

COPY mpl_infer_tiles_GPU_new.py .

COPY mpl_process_shapefile.py .

COPY mpl_stitchshpfile_new.py .

COPY mpl_config.py .

COPY utils.py .

COPY model.py .

RUN ls

RUN conda env create -f environment_maple.yml

RUN sudo apt-get update

RUN apt-get clean

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libxrender-dev libgl1-mesa-glx -y

SHELL ["conda", "run", "-n", "maple_py310", "/bin/bash", "-c"]

CMD ["conda", "run", "--no-capture-output", "-n", "maple_v310", "python","-u", "maple_workflow_create_dir_struct.py"]
