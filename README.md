# MAPLE

## Introduction
Increasing availability of high spatial resolution remote sensing data, particularly the sub-meter imagery acquired by commercial satellite sensors, provide unprecedented opportunities to map, monitor, and document permafrost features and their change over time. High performance image analysis algorithms/workflows/pipelines centered on AI are indeed required to perform rigonal to pan-Arctic scale mappings from big image data sets that are freely available to the NSF-funded Arctic researcher community.

Mapping Application for Arctic Permafrost Land Environment (MAPLE) serves as the main container/pipeline harboring multiple mapping workflows. The MAPLE is essentially an extensible pipeline that centers on DL and high performance computing. It allows users to interlace different deep learning convolutional neural net algorithms to conveniently analyze remote sensing imagery and geospatial data.

![overview](maple_overview.png)

Mapping requires a workflow to train an AI ML model to learn how to do the mapping and another pipeline to do the mapping (inferencing). This code base is for the inferencing that will use an already trianed model to do the mapping of a large volume of avaialble un mapped images. 

In general AI-ML-DL models are transferable via [weight file](#weight-file) that is a prerequsite for this model to do the inferencing. And it should be noted that the type of training data that was used for training will have a major effect on the accuracy. This codebase was used to produce the data product at the [Permafrost Discovery Gateway](https://arcticdata.io/catalog/portals/permafrost) Considering the volume of the data a big data pipeline was used on HPCs for the mapping. Care was taken ensure that the code base work on hetrogenious big data envirenments taking advantage of the underline computing architectures. This code can be also executed on a single local machine with or without GPUs. 

# Setting up Conda Envirenment
The code is scripted using python and for execution the proper python env should be setup using conda. It is advisable to install the latest version of miniconda or varient and install the required envionment using the yml file provided. The required packages are:

  - python=3.6
  - tensorflow-gpu==1.14.0
  - keras==2.3.1
  - numpy==1.17.4
  - gdal==2.3.2
  - pyshp==2.1.0
  - scikit-image==0.16.2
  - shapely==1.6.4

Make sure conda is used rather than pip to install as the conda will let you know if there are conflict in the envirenment. Some of the above are python wrappers to libraries that may have issues in the particular envirenment you are running it. [More details about conda managment can be found](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-conda.html)


# Running MAPLE

To execute maple you need to run the workflow. Depending on the location where you are runing you may need to write some code to generate a batch script to be deployed on the compute nodes some examples can be made avaialble to get started.

## Configuration
Multiple configurations are possible based on the compute envirenment and also the input data to be processed.
It is required to maintain the model based configurations to ensure you get an output that is consistent to the training done. The most important configuration is to indicate the location of the weight file

It is recomended not to edit the main config file but to have our own config file to modify the configuration based on our requirements. You can edit the maple_config.py file with all the required parameters that include the location of the data and the locations of the output files and also the locations to store the temporary files. Since most of the HPCs do not allow us to store large files on local nodes we have to use common location for the files.

As it is wirtten you need to create a directory structure based on the config file to store the temp data. A small [pythen script](mpl_wokflow_create_dir_struct.py) is written  to generate this structure but make sure it mathches the configuration. Otherwise errors will be thrown in certain cases and may not in certain cases and fail later on the pipeline.

The following directory structure and the files must be there to run

├── data <br>
│ ├── cln_data <br>
│ ├── divided_img <br>
│ ├── final_shp <br>
│ ├── input_img_local <br>
│ │    └── test_image_01.tif <br>
│ ├── output_img <br>
│ ├── output_shp <br>
│ └── water_mask <br>
│    └── temp <br>
└── trained_weights_Dataset_251_13_24_.h5 <br>

Root directory where this structure is located should also be given in the maple_configuration file. 

For HPC execution depending on your HPC you also need to indicate the .out .err locations for it to get it executed as a batch job.

## Preparing Data

Data can be in geo tagged tiff files. Have the option of running it by pointing it to a input data directory that contains the files that you want executed or you can give a list of files to be executed. There is also an option to use a collection of overlap shapefiles related a to a batch of input files that inidicates the overlaps so that the processing of overalped input images can be avoided.

As part of the data preperation
For the basic vanila configuration you only need a list of input tiff files in a directory and for this shared code it is for one image file

<h2 id="weight-file"> Weight File </h2>

This is the trainded weight file that is required for the model to do the inferencing


