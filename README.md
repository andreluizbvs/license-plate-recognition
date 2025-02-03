# License Plate Recognition
Code repository for the a system that retrieves the information of license plates given an image or video.

## Installation

### Using pip

1. **Download the wheel file:**

    Download the latest wheel file from the [releases page](https://github.com/andreluizbvs/license-plate-recognition/releases).

2. **Install the wheel file:**

    ```sh
    pip install path/to/plate_ocr-<version>-py3-none-any.whl
    ```


### Using Poetry

1. **Clone the repository:**

    ```sh
    git clone https://github.com/andreluizbvs/license-plate-recognition.git
    cd license-plate-recognition
    ```

2. **Install Poetry:**

    Follow the instructions on the [Poetry website](https://python-poetry.org/docs/#installation) to install Poetry.

3. **Install dependencies:**

    ```sh
    poetry install
    ```

## Usage

You can use the CLI to process videos or images to extract license plate information.

### Process a Video

```sh
plate-ocr --type video --path path/to/your/video.mp4
```

### Process an Image

```sh
plate-ocr --type image --path path/to/your/image.jpg
```

#### Options

- `--approach`: Choose the approach to use: 1 (slower & more accurate) or 2 (faster & less accurate). Default is 2. **More details on the approaches in the report below.**
- `--type`: Specify the type of input (video or image). This option is required.
- `--path`: Path to the video or image file. This option is required.

# Chain of thought for developing this project

## 1. Uncertainties

- How is the scene? 
    - Urban or Rural settings? 
    - Streets, Roads, Highways?
    - Parking lots?
    - Wide or narrow shot frame (amount of vehicles in scene)?
- Image resolution? 
- Type of image (RGB, etc.)?
- How many vehicles in the scene/frame? 
    - How many plates? 
        - From which countries? 
        - Which model of license plates? 
        - Recognize only the letters and numbers or also other information from it?
- If I only have one week to do this, I shouldn't spend too much time thinking of all edge cases and possiblities or studying, I'll try to come up with a couple of approaches, raise some pros and cons and decide what to implement (maybe also discuss with some LLMs).


## 2. How can I approach this problem?


### Approach #1

- Run a vehicle detection or segmentation model for all vehicles in the frame
- Run a license plate detection model for each cropped vehicles 
- Run a OCR model to extract all information from each license plates

#### Pros

- Probably gonna yield the best results

#### Cons

- Quite expensive
    - Three different models 
        - Vehicle detection (model 1)
        - License plate detection (model 2)
        - License plate OCR model (model 3)
    - High computational cost in a scene with many vehicles
        - Lets say n is inference time for model 1, m is inference time for model 2 and k is inference time for model 3:
        - In a scene with v vehicles, the total inference time (only prediction and not considering preprocessing/postprocessing time, which are normally much faster than model prediction) in this case would be n + v * (m + k)
            - Just had a new idea: I could stitch together all the vehicles detected in one single image and run the license plate model on it only once. This would decrease the total inference time to n + m + v * k
            - Not a good idea actually, because of image resizing for model input.
        - Sometimes is hard enough to keep one inference in real-time!

### Approach #2

- Run the license plate detection model once in the whole frame looking for all the plates directly
- Run the OCR model to extract all information for each license plate

#### Pros

- Cheaper
- Faster
- Only two models to develop/maintain
    - License plate detection (model 2)
    - OCR model (model 3)

#### Cons

- Probably not so great results
- Lets say m is inference time for model 2 and k is inference time for model 3:
    - In a scene with v vehicles, the total inference time in this case would be m + v * k

## 3. Revisiting Problem statement and making comments

Build a CLI that has two functionalities:
- Video: Build a small system that receives a video and from that video you extract license plates from it `----> Track the plates in the video to just show unique plates (they would be added to a Python set/dict)`
    - Printing the plate details to terminal `-----> but is fast and better to just export a JSON file`
- Image processing: Given a frame, you should print the license plate information. You can use detection algorithms like yolo to detect cars and so on... `----> this is just a subcase of the video input. In this case there is no need to use any tracking`

it would be great if you could explain how you would approach the problem if you had more time and resources `----> definetly use Approach #1, ONNX/Torchscript/TensorRT (depends where the solution will be deployed) to optimize the inference time in each model and check if we can run everything in real-time (or near). After building an MVP with the steps I just described, I would fine-tune each model with more data to improve mAP/balanced accuracy. On second thought, is real-time really that important? Maybe we just need to run it in every x frames of a video.`

Time spent

You should spend less than a week on this problem, we are not interested in a productionized solution, we are interested in how you can tackle problems (`you may evaluate that by looking at my chain of thought, the approaches I described, the comments I added in this problem statement, and the tech deep dive`), code organization (`modularization, maybe some OOP if needed, Ruff to format the code, etc.`) and how well you figure out solutions on your own (`everything up to this point was just me, no LLMs [which will be the next part =)]`).


## 4. LLMs!

- Okay, now that I spent some time reflecting about this problem, its time to discuss the problem with a LLM (GitHub Copilot, Deepseek R1, GPT o3, etc.) and check if I'm missing something.

Prompt:

```
<Problem statement...>

Given the context above, return me two roadmaps with detailed steps and all important things I should consider while developing this project:
    - One roadmap should considering I only have one week to develop that system with limited resources
    - The other should consider I have more time and resources

Also provide some code or pseudocode that code serve as a starting point to this project.
```

Very little help here, actually. I chose to use an old project as a starting point and I will be communicating with GitHub Copilot to accelerate development.


## 5. Looking for some data to test the system

- Kaggle is a good place with free data
- The web in general
- Which dataset to use? Global or Local? Cars only?
    - Since from our talk the problem is initially set in Brazil I would focus here, but if it's not that hard to use/develop a model for global plate recog, I would use one (also check its performance on a brazilian plate dataset)

Some useful datasets/links:
- https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
- https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
- https://github.com/ramajoballester/UC3M-LP
- https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset

Found 2 sample videos for iterative testing (here are the links: [sample.mp4](https://drive.google.com/file/d/1otrBjr0GycUVr-gTBb4pAyjD85ftQ5tO/view?usp=drive_link) and [sample2.mp4](https://drive.google.com/file/d/1mk2VZE2TNbFAHUaUVZK4O4uYDrWYnWih/view?usp=drive_link))

## 6. Looking for some useful references (models, weights, data, etc.)

These are very helpful!
- https://github.com/ankandrew/fast-plate-ocr
- https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8


## 7. Coding...

## 8. Bad results
 - Thinking of ways to improve such as:
    1. Image preprocessing (traditional techniques):
        - Grayscale conversion (helped)
        - CLAHE (helped!)
        - Binarization (didn't help)
        - Morphological operations (didn't help)
            - Erosion
            - Dilation 
    2. Filter low confidence detections
    3. Minimum area strategy: 
        - Very small bounding boxes related to the input image/frame doesn't need to be analysed since they very likely won't be successfully OCR'ed (due to image resizing for model input)
            - This decreases "blinking" detections and objects too far away
    4. Plate frequency counter strategy:
        - To be really considered recognized in videos, plates need to be recognized a reasonable amount of times

## 9. Finally some good results
 - The strategies above help immensely!
 - Created the poetry project and built a release. Model weights inside it. Tested and everything worked, apparently fine.
 - In CPU, approach #1 takes 400~500ms per frame
 - In CPU, approach #2 takes 100~200ms per frame
 - No GPU tested, but it would probably be way faster
 - Need to run the results through a dataset to check the accuracy (but it would be difficult to check the accuracy due to the adopted strategies [what to compare, exactly])
 - The end (or the beginning?)