import json
import os
import time
from collections import Counter
from typing import Dict, List, Tuple

import requests
from flask import Flask, render_template, request
from Levenshtein import distance as lev_distance
from pydub import AudioSegment
from pydub.silence import split_on_silence

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/Zaid/whisper-large-v2-ar"
HEADERS = {"Authorization": "Bearer <Insert_your_hugging_face_API_here>"}

MAX_RETRIES = float("inf")
WAIT_TIME = 60


def query(api_url: str, filename: str) -> Dict:
    """
    Queries the Hugging Face API with the given audio file and returns the response as a dictionary.

    Args:
        api_url (str): The URL of the Hugging Face API.
        filename (str): The path to the audio file to transcribe.

    Returns:
        dict: A dictionary containing the transcription of the audio file.
    """
    with open(filename, "rb") as f:
        data = f.read()
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.post(api_url, headers=HEADERS, data=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Please wait some time, the model is currently loading...")
            retries += 1
            print(f"Retrying request ({retries}/{MAX_RETRIES}) in {WAIT_TIME} seconds...")
            time.sleep(WAIT_TIME)
    print("Max number of retries reached. Request failed.")
    return None


def load_words(filepath: str) -> Dict:
    """
    Loads a dictionary of words from a file.

    Args:
        filepath (str): The path to the file containing the words.

    Returns:
        dict: A dictionary containing the words and their correct spellings.
    """
    words_dict = {}
    with open(filepath, "r") as f:
        # load the words dict with data from the file, wrong key is the first word, correct is the second word separated by ":"
        for line in f:
            closest, exact = line.strip().split(":")
            words_dict[closest] = exact
    return words_dict


def closest_match(target: str, words: List[str]) -> str:
    """
    Finds the closest matching word to the target word from a list of words.

    Args:
        target (str): The target word to match.
        words (List[str]): A list of words to match against.

    Returns:
        str: The closest matching word from the list.
    """
    closest_word = ""
    min_distance = float("inf")
    for word in words:
        distance = lev_distance(target, word)
        if distance < min_distance:
            min_distance = distance
            closest_word = word
    return closest_word


def find_top_letter_swaps(predictions: Dict) -> List[Tuple[Tuple[str, str], int]]:
    """
    Finds the top letter swaps between the predicted words and the correct words.

    Args:
        predictions (Dict): A dictionary containing the predicted and correct words.

    Returns:
        List[Tuple[Tuple[str, str], int]]: A list of tuples containing the letter swap and the number of occurrences.
    """
    letter_swaps = Counter()
    for value in predictions.values():
        exact_word = value["exact"]
        closest_word = value["closest"]
        swaps = set()
        i = j = 0
        while i < len(exact_word) and j < len(closest_word):
            if exact_word[i] != closest_word[j]:
                if i + 1 < len(exact_word) and exact_word[i + 1] == closest_word[j]:
                    swaps.add((exact_word[i], closest_word[j]))
                    i += 1
                elif j + 1 < len(closest_word) and exact_word[i] == closest_word[j + 1]:
                    swaps.add((closest_word[j], exact_word[i]))
                    j += 1
                else:
                    swaps.add((exact_word[i], closest_word[j]))
                    swaps.add((closest_word[j], exact_word[i]))
            i += 1
            j += 1
        letter_swaps.update(swaps)

    top_swaps = letter_swaps.most_common(5)
    return top_swaps


def remove_duplicates(items: List[Tuple[Tuple[str, str], int]]) -> List[Tuple[Tuple[str, str], int]]:
    """
    Removes duplicate letter swaps from a list of tuples.

    Args:
        items (List[Tuple[Tuple[str, str], int]]): A list of tuples containing the letter swap and the number of occurrences.

    Returns:
        List[Tuple[Tuple[str, str], int]]: A list of tuples containing the unique letter swaps and the number of occurrences.
    """
    unique_items = set()
    result = []
    for item, count in items:
        swap1 = (item[0], item[1])
        swap2 = (item[1], item[0])
        if swap1 not in unique_items and swap2 not in unique_items:
            unique_items.add(swap1)
            result.append((swap1, count))
    return result


def empty_directory(directory_path: str) -> None:
    """
    Empties a directory of all files.

    Args:
        directory_path (str): The path to the directory to empty.
    """
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        os.remove(filepath)


def split_audio_file(
    input_file: str, output_folder: str, min_silence_len: int = 500, silence_thresh: int = -40, keep_silence: int = 500
) -> None:
    """
    Splits an audio file into segments based on silence.

    Args:
        input_file (str): The path to the input audio file.
        output_folder (str): The path to the output directory for the audio segments.
        min_silence_len (int): The minimum length of silence to split on, in milliseconds.
        silence_thresh (int): The threshold for silence, in decibels.
        keep_silence (int): The amount of silence to keep at the beginning and end of each segment, in milliseconds.
    """
    audio = AudioSegment.from_wav(input_file)
    segments = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence,
    )

    for i, segment in enumerate(segments):
        output_file = f"{output_folder}/segment_{i}.wav"
        segment.export(output_file, format="wav")
        print(f"Saved {output_file}")


def transcribe_directory(directory_path: str) -> Dict:
    """
    Transcribes all audio segments in a directory and returns a dictionary of the predicted and correct words.

    Args:
        directory_path (str): The path to the directory containing the audio segments.

    Returns:
        Dict: A dictionary containing the predicted and correct words for each audio segment.
    """
    predictions = {}
    words_dict = load_words("all.txt")
    for filename in os.listdir(directory_path):
        if filename.startswith("segment_"):
            filepath = os.path.join(directory_path, filename)
            output = query(API_URL, filepath)["text"]
            # output = query_hosted(filepath, "https://b9d4-34-124-186-217.ngrok-free.app/transcribe")["text"]
            output = "".join([c for c in output if c not in ["ِ", "ُ", "ٓ", "ٰ", "ْ", "ٌ", "ٍ", "ً", "ّ", "َ"]])
            output = output.replace(".", "")
            closest_word = closest_match(output, words_dict.keys())
            predictions[filename] = {
                "closest": closest_word,
                "exact": words_dict[closest_word],
            }
            print(
                f"{filename}: {output} <- {closest_word} <- {words_dict[closest_word]}"
            )
    return predictions


# function that write dict to file in json format with unique name
def write_to_file(dictionary: Dict) -> None:
    """
    Writes a dictionary to a file in JSON format with a unique name.

    Args:
        dictionary (Dict): The dictionary to write to the file.
    """
    file_name = "output" + str(time.time()) + ".json"
    with open(file_name, "w") as f:
        json.dump(dictionary, f)


@app.route("/", methods=["GET", "POST"])
def upload_file() -> str:
    """
    Handles file uploads and displays the top letter swaps.

    Returns:
        str: The HTML template to display.
    """
    if request.method == "POST":
        audio_file = request.files["audio_file"]
        if audio_file:
            directory_path = "./audio_segments"
            os.makedirs(directory_path, exist_ok=True)
            empty_directory(directory_path)
            audio_path = os.path.join(directory_path, audio_file.filename)
            audio_file.save(audio_path)
            split_audio_file(audio_path, directory_path)
            predictions = transcribe_directory(directory_path)
            write_to_file(predictions)
            top_swaps = find_top_letter_swaps(predictions)
            top_swaps = remove_duplicates(top_swaps)
            return render_template("index.html", top_swaps=top_swaps)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
