# License Plate Recognition
Code repository for the a system that given an image, returns the all license plates information (characters, origin, etc.) from it.


## Installation

Create conda environment from this 
...


## Inference

...


# Chain of thought developing of this project

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
            - Apparently the available License plate detection/OCR models to the detection + OCR in one step, that would decrease again the total inference time to n + m + k (ideal)
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


## 5. Look for some data to test the system

- Kaggle is a good place with free data
- The web in general
- Which dataset to use? Global or Local? Cars only?
    - Since from our talk the problem is initially set in Brazil I would focus here, but if it's not that hard to use/develop a model for global plate recog, I would use one (also check its performance on a brazilian plate dataset)

Some useful datasets/links:
- https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
- https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
- https://github.com/ramajoballester/UC3M-LP
- https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset

Found 2 sample videos for iterative testing (here a the links: sample.mp4 and sample2.mp4)

## References (models, weights, data, etc.)

- https://github.com/ankandrew/fast-plate-ocr
- https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8
