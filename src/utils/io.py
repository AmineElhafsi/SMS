from pathlib import Path
from typing import Union

import os
import shutil

import torch
import dill as pickle
import json
import yaml


def load_pkl(file_path: Union[str, Path]) -> object:
    """ Loads a pickle file and returns its contents.
    Args:
        file_path: The path to the pickle file.
    Returns:
        The contents of the pickle file.
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)
    

def load_json(file_path: Union[str, Path]) -> dict:
    """ Loads a JSON file and returns its contents as a dictionary.
    Args:
        file_path: The path to the JSON file.
    Returns:
        The dictionary containing the contents of the JSON file.
    """
    with open(file_path, "r") as file:
        return json.load(file)


def load_yaml(file_path: Union[str, Path]) -> dict:
    """ Loads a YAML file and returns its contents as a dictionary.
    Args:
        file_path: The path to the YAML file.
    Returns:
        The dictionary containing the contents of the YAML file.
    """
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def mkdir_decorator(func):
    """A decorator that creates the directory specified in the function's 'directory' keyword
       argument before calling the function.
    Args:
        func: The function to be decorated.
    Returns:
        The wrapper function.
    """
    def wrapper(*args, **kwargs):
        output_path = Path(kwargs["directory"])
        output_path.mkdir(parents=True, exist_ok=True)
        return func(*args, **kwargs)
    return wrapper


@mkdir_decorator
def save_dict_to_ckpt(dictionary: dict, file_name: str, *, directory: Union[str, Path]) -> None:
    """ Saves a dictionary to a checkpoint file in the specified directory, creating the directory 
    if it does not exist.
    Args:
        dictionary: The dictionary to be saved.
        file_name: The name of the checkpoint file.
        directory: The directory where the checkpoint file will be saved.
    """
    torch.save(dictionary, directory / file_name,
               _use_new_zipfile_serialization=False)
    

@mkdir_decorator
def save_dict_to_yaml(dictionary: dict, file_name: str, *, directory: Union[str, Path]) -> None:
    """ Saves a dictionary to a YAML file in the specified directory, creating the directory if it does not exist.
    Args:
        dictionary: The dictionary to be saved.
        file_name: The name of the YAML file.
        directory: The directory where the YAML file will be saved.
    """
    with open(directory / file_name, "w") as f:
        yaml.dump(dictionary, f)

@mkdir_decorator
def save_dict_to_pkl(dictionary: dict, file_name: str, *, directory: Union[str, Path]) -> None:
    """ Saves a dictionary to a pickle file in the specified directory, creating the directory if it does not exist.
    Args:
        dictionary: The dictionary to be saved.
        file_name: The name of the pickle file.
        directory: The directory where the pickle file will be saved.
    """
    with open(directory / file_name, "wb") as file:
        pickle.dump(dictionary, file)

# save dict to json
@mkdir_decorator
def save_dict_to_json(dictionary: dict, file_name: str, *, directory: Union[str, Path]) -> None:
    """ Saves a dictionary to a JSON file in the specified directory, creating the directory if it does not exist.
    Args:
        dictionary: The dictionary to be saved.
        file_name: The name of the JSON file.
        directory: The directory where the JSON file will be saved.
    """
    with open(directory / file_name, "w") as f:
        json.dump(dictionary, f, indent=4)


def create_directory(directory: Union[str, Path], overwrite: bool) -> None:
    """
    Create a directory if it does not exist. If the directory already exists, it will be deleted and recreated if
    the overwrite flag is set to True.

    Args:
        folder_path (str): The path to the folder to be deleted and recreated.
    """
    directory = Path(directory)

    if directory.exists():
        if overwrite:
            print(f"Output directory {directory} already exists. Overwriting.")
            # delete existing directory and create a new one
            shutil.rmtree(directory)
            directory.mkdir(parents=True)
        else:
            print(f"Output directory {directory} already exists. Not overwriting.")
            # raise ValueError(f"Output directory {directory} already exists.")
    else:
        directory.mkdir(parents=True)


def get_unique_file_path(file_path):
    """
    Generate a unique file path by appending a number if the file already exists.
    
    Args:
        filepath (str): The desired file path.
    
    Returns:
        str: A unique file path.
    """
    directory, filename = os.path.split(file_path)
    base, ext = os.path.splitext(filename)
    counter = 1
    unique_file_path = file_path

    while os.path.exists(unique_file_path):
        unique_file_path = os.path.join(directory, f"{base}_{counter}{ext}")
        counter += 1

    return unique_file_path