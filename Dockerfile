FROM ubuntu:22.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
RUN apt-get install -y sudo

#RUN apt-get install -y python3.6


RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh
RUN conda --version


#RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh
#RUN bash ~/miniforge.sh -b -p ~/miniforge

#
RUN conda clean -a
#
RUN echo $CONDA_PREFIX
#
COPY environment_maple.yml .
#
COPY config.py .
#
COPY hyp_best_train_weights_final.h5 .
#
COPY maple_workflow.py .
#
COPY mpl_clean_inference.py .
#
COPY mpl_divideimg_234_water_new.py .
#
COPY mpl_infer_tiles_GPU_new.py .
#
COPY mpl_process_shapefile.py .
#
COPY mpl_stitchshpfile_new.py .
#
COPY mpl_config.py .
#
COPY utils.py .
#
COPY model.py .
#
COPY test.py .
#
RUN ls
#
RUN conda env create -f environment_maple.yml
#
RUN sudo apt-get update
#
RUN apt-get clean
#
RUN  python -m pip install opencv-python
#
#
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libxrender-dev libgl1-mesa-glx -y
#
#
#
SHELL ["conda", "run", "-n", "maple_py39", "/bin/bash", "-c"]
#
#RUN which python

#RUN python -m pip install tensorflow-deps
##
#RUN python -m pip install tensorflow-macos
##
#RUN python -m pip install tensorflow-metal




CMD ["conda", "run", "--no-capture-output", "-n", "maple_v39", "python","-u", "maple_workflow.py"]