import json
import os
import posixpath

import requests


def download_input_files_clowder(host, dataset_id, key, directory):
    headers = {'Content-Type': 'application/json',
               'X-API-KEY': key}
    url = posixpath.join(host, "api/v2/datasets/%s/files" % dataset_id)

    result = requests.get(url, headers=headers)
    result.raise_for_status()

    files = json.loads(result.text)["data"]
    print(files)

    # # Loop through dataset and download all file "locally"
    for file_dict in files:
        # Use the correct key depending on the Clowder version
        extension = "." + file_dict['content_type']['content_type'].split("/")[1]
        if file_dict['name'] != 'hyp_best_train_weights_final.h5':
            _download_file(host, key, file_dict['id'], file_dict['name'], directory)


def download_model_weights_clowder (host, dataset_id, key, directory):
    headers = {'Content-Type': 'application/json',
               'X-API-KEY': key}
    url = posixpath.join(host, "api/v2/datasets/%s/files" % dataset_id)

    result = requests.get(url, headers=headers)
    result.raise_for_status()

    files = json.loads(result.text)["data"]
    print(files)

    for file_dict in files:
        # Use the correct key depending on the Clowder version
        extension = "." + file_dict['content_type']['content_type'].split("/")[1]
        if file_dict['name'] == 'hyp_best_train_weights_final.h5':
            _download_file(host, key, file_dict['id'], file_dict['name'], directory)


def upload_output_clowder(host, dataset_id, key, directory):
    headers = {'Content-Type': 'application/json',
               'X-API-KEY': key}
    url = posixpath.join(host, "api/v2/datasets/%s/filesMultiple" % dataset_id)

    files = []
    dir_list = os.listdir(directory)
    print("Output files found: ", dir_list)
    for file in dir_list:
        files.append(("files", open(file, "rb")))
    response = requests.post(
        url,
        headers=headers,
        files=files,
    )
    response.raise_for_status()


def _download_file(host, key, file_id, name, directory):
    url = posixpath.join(host, f'api/v2/files/{file_id}')
    headers = {"X-API-KEY": key}
    file_path = os.path.join(directory, name)

    r = requests.get(url, stream=True, headers=headers)

    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=16 * 1024):
            f.write(chunk)
    f.close()

