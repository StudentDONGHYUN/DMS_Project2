TITLE: Creating and Activating Python Virtual Environment
DESCRIPTION: This command creates a new Python virtual environment named `mp_env` and then activates it. This practice isolates project dependencies, preventing conflicts with other Python projects or system-wide installations.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python.md#_snippet_0

LANGUAGE: Bash
CODE:
```
$ python3 -m venv mp_env && source mp_env/bin/activate
```

----------------------------------------

TITLE: Cloning MediaPipe Repository (Bash)
DESCRIPTION: This command sequence clones the MediaPipe GitHub repository and navigates into its root directory, which is the first step for setting up the MediaPipe development environment.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/youtube_8m.md#_snippet_0

LANGUAGE: bash
CODE:
```
git clone https://github.com/google/mediapipe.git
cd mediapipe
```

----------------------------------------

TITLE: Adding MediaPipe Android Solution APIs to Gradle Dependencies
DESCRIPTION: This snippet demonstrates how to add MediaPipe Android Solution APIs to an Android Studio project's `build.gradle` file. It includes the core solution library and optional libraries for Face Detection, Face Mesh, and Hands, allowing developers to easily incorporate these functionalities into their Android applications. These dependencies are fetched from Google's Maven Repository.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/android_solutions.md#_snippet_0

LANGUAGE: Gradle
CODE:
```
dependencies {
    // MediaPipe solution-core is the foundation of any MediaPipe Solutions.
    implementation 'com.google.mediapipe:solution-core:latest.release'
    // Optional: MediaPipe Face Detection Solution.
    implementation 'com.google.mediapipe:facedetection:latest.release'
    // Optional: MediaPipe Face Mesh Solution.
    implementation 'com.google.mediapipe:facemesh:latest.release'
    // Optional: MediaPipe Hands Solution.
    implementation 'com.google.mediapipe:hands:latest.release'
}
```

----------------------------------------

TITLE: Cloning MediaPipe Repository and Navigating Directory
DESCRIPTION: These commands clone the MediaPipe GitHub repository with a depth of 1 (shallow clone) and then change the current directory into the newly cloned 'mediapipe' folder, preparing for further setup.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_34

LANGUAGE: bash
CODE:
```
git clone --depth 1 https://github.com/google/mediapipe.git

cd mediapipe
```

----------------------------------------

TITLE: Detecting Faces with MediaPipe Face Detector (JavaScript)
DESCRIPTION: This snippet initializes the MediaPipe Face Detector task by loading the necessary WASM files and a pre-trained model. It then performs face detection on an HTML image element, returning the detected face locations and presence. Requires the @mediapipe/tasks-vision library.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md#_snippet_0

LANGUAGE: JavaScript
CODE:
```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const faceDetector = await FaceDetector.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
);
const image = document.getElementById("image") as HTMLImageElement;
const detections = faceDetector.detect(image);
```

----------------------------------------

TITLE: Declaring Camera Permissions in AndroidManifest.xml
DESCRIPTION: This XML snippet declares the necessary permissions and features in `AndroidManifest.xml` to allow the application to access and use the device's camera. It requests `android.permission.CAMERA` and specifies the use of `android.hardware.camera`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_9

LANGUAGE: xml
CODE:
```
<!-- For using the camera -->
<uses-permission android:name="android.permission.CAMERA" />
<uses-feature android:name="android.hardware.camera" />
```

----------------------------------------

TITLE: Initializing and Using MediaPipe LLM Inference in JavaScript
DESCRIPTION: This snippet demonstrates how to initialize the MediaPipe LLM Inference task and generate a text response. It requires a FilesetResolver to load the WASM module from a CDN and a pre-trained LLM model, specified by MODEL_URL, to perform inference on the inputText.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/genai/README.md#_snippet_0

LANGUAGE: JavaScript
CODE:
```
const genai = await FilesetResolver.forGenAiTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai/wasm"
);
const llmInference = await LlmInference.createFromModelPath(genai, MODEL_URL);
const response = await llmInference.generateResponse(inputText);
```

----------------------------------------

TITLE: Cloning MediaPipe Repository (Bash)
DESCRIPTION: This command sequence clones the MediaPipe GitHub repository to the local machine and then changes the current directory into the newly cloned `mediapipe` directory. It is the initial step required to set up the development environment for MediaPipe projects.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/media_sequence.md#_snippet_0

LANGUAGE: bash
CODE:
```
git clone https://github.com/google/mediapipe.git
cd mediapipe
```

----------------------------------------

TITLE: Initializing and Processing with MediaPipe Holistic in JavaScript
DESCRIPTION: This JavaScript code initializes the MediaPipe Holistic model, configures its options, and sets up a camera to feed video frames for processing. The `onResults` function handles the output, drawing segmentation masks, landmarks for pose, face, and hands onto a canvas.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md#_snippet_3

LANGUAGE: JavaScript
CODE:
```
<script type="module">
const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');

function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.segmentationMask, 0, 0,
                      canvasElement.width, canvasElement.height);

  // Only overwrite existing pixels.
  canvasCtx.globalCompositeOperation = 'source-in';
  canvasCtx.fillStyle = '#00FF00';
  canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);

  // Only overwrite missing pixels.
  canvasCtx.globalCompositeOperation = 'destination-atop';
  canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);

  canvasCtx.globalCompositeOperation = 'source-over';
  drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS,
                 {color: '#00FF00', lineWidth: 4});
  drawLandmarks(canvasCtx, results.poseLandmarks,
                {color: '#FF0000', lineWidth: 2});
  drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_TESSELATION,
                 {color: '#C0C0C070', lineWidth: 1});
  drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS,
                 {color: '#CC0000', lineWidth: 5});
  drawLandmarks(canvasCtx, results.leftHandLandmarks,
                {color: '#00FF00', lineWidth: 2});
  drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS,
                 {color: '#00CC00', lineWidth: 5});
  drawLandmarks(canvasCtx, results.rightHandLandmarks,
                {color: '#FF0000', lineWidth: 2});
  canvasCtx.restore();
}

const holistic = new Holistic({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
}});
holistic.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  enableSegmentation: true,
  smoothSegmentation: true,
  refineFaceLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
holistic.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await holistic.send({image: videoElement});
  },
  width: 1280,
  height: 720
});
camera.start();
</script>
```

----------------------------------------

TITLE: Processing Webcam Input with MediaPipe Pose (Python)
DESCRIPTION: This snippet illustrates how to use MediaPipe Pose for real-time pose detection from a webcam feed. It initializes the `Pose` model for live video, continuously captures frames, processes them to detect pose landmarks, and displays the annotated video stream. It also includes performance optimization by marking images as not writeable.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md#_snippet_1

LANGUAGE: Python
CODE:
```
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
```

----------------------------------------

TITLE: Cloning MediaPipe Repository (Bash)
DESCRIPTION: This command clones the official MediaPipe GitHub repository to your local machine. This repository contains all the source code, examples, and necessary files for building MediaPipe applications.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/ios.md#_snippet_3

LANGUAGE: bash
CODE:
```
git clone https://github.com/google/mediapipe.git
```

----------------------------------------

TITLE: Installing MediaPipe Solution via NPM
DESCRIPTION: This snippet demonstrates how to install a MediaPipe solution package, specifically @mediapipe/holistic, using the npm package manager. This command adds the specified package to your project's node_modules directory, making it available for local development. It's a common method for integrating MediaPipe solutions into JavaScript projects.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/javascript.md#_snippet_0

LANGUAGE: Bash
CODE:
```
npm install @mediapipe/holistic.
```

----------------------------------------

TITLE: Detecting Holistic Body Landmarks with MediaPipe Holistic Landmarker (JavaScript)
DESCRIPTION: This snippet initializes the MediaPipe Holistic Landmarker task, loading the WASM files and a pre-trained model. It combines pose, face, and hand landmark detection to provide a complete set of human body landmarks from an HTML image element. Requires the @mediapipe/tasks-vision library.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md#_snippet_5

LANGUAGE: JavaScript
CODE:
```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const holisticLandmarker = await HolisticLandmarker.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/hand_landmark.task"
);
const image = document.getElementById("image") as HTMLImageElement;
const landmarks = holisticLandmarker.detect(image);
```

----------------------------------------

TITLE: Recognizing Hand Gestures with MediaPipe Gesture Recognizer (JavaScript)
DESCRIPTION: This snippet initializes the MediaPipe Gesture Recognizer task, loading the WASM files and a pre-trained model. It then recognizes hand gestures in an HTML image element, providing both the recognized gesture results and hand landmarks. Requires the @mediapipe/tasks-vision library.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md#_snippet_3

LANGUAGE: JavaScript
CODE:
```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const gestureRecognizer = await GestureRecognizer.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
);
const image = document.getElementById("image") as HTMLImageElement;
const recognitions = gestureRecognizer.recognize(image);
```

----------------------------------------

TITLE: Installing Android APK via ADB
DESCRIPTION: This command installs the compiled Android Package Kit (APK) file onto a connected device or emulator using Android Debug Bridge (ADB). The APK is located in the `bazel-bin` directory under the application's path.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_8

LANGUAGE: bash
CODE:
```
adb install bazel-bin/$APPLICATION_PATH/helloworld.apk
```

----------------------------------------

TITLE: Binding Custom C++ Class with pybind11
DESCRIPTION: This C++ snippet demonstrates how to use pybind11 to create Python bindings for a custom C++ class, 'MyType'. It shows the basic structure for defining the class within a pybind11 module, including its constructor and other potential member functions, making it accessible from Python.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_26

LANGUAGE: C++
CODE:
```
#include "path/to/my_type/header/file.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

PYBIND11_MODULE(my_type_binding, m) {
  // Write binding code or a custom type caster for MyType.
  py::class_<MyType>(m, "MyType")
      .def(py::init<>())}

```

----------------------------------------

TITLE: Building MediaPipe AAR (Bazel)
DESCRIPTION: This Bazel command compiles and links the MediaPipe AAR. It includes various optimization flags (`-c opt`, `--strip=ALWAYS`, `-Oz`), specifies target CPU architectures (`arm64-v8a,armeabi-v7a`), and manages linking options to reduce binary size and improve performance. The command targets a generic AAR build file path and name.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/android_archive_library.md#_snippet_1

LANGUAGE: bash
CODE:
```
bazel build -c opt --strip=ALWAYS \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
    --fat_apk_cpu=arm64-v8a,armeabi-v7a \
    --legacy_whole_archive=0 \
    --features=-legacy_whole_archive \
    --copt=-fvisibility=hidden \
    --copt=-ffunction-sections \
    --copt=-fdata-sections \
    --copt=-fstack-protector \
    --copt=-Oz \
    --copt=-fomit-frame-pointer \
    --copt=-DABSL_MIN_LOG_LEVEL=2 \
    --linkopt=-Wl,--gc-sections,--strip-all \
    //path/to/the/aar/build/file:aar_name.aar
```

----------------------------------------

TITLE: Initializing MediaPipe CalculatorGraph with Text Config (Python)
DESCRIPTION: This snippet demonstrates how to initialize a MediaPipe CalculatorGraph using a text-based CalculatorGraphConfig protobuf. It sets up an input and output stream and defines a PassThroughCalculator. An output stream observer is registered to collect packets into a list, showcasing how to capture results.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_31

LANGUAGE: Python
CODE:
```
import mediapipe as mp

config_text = """
  input_stream: 'in_stream'
  output_stream: 'out_stream'
  node {
    calculator: 'PassThroughCalculator'
    input_stream: 'in_stream'
    output_stream: 'out_stream'
  }
"""
graph = mp.CalculatorGraph(graph_config=config_text)
output_packets = []
graph.observe_output_stream(
    'out_stream',
    lambda stream_name, packet:
        output_packets.append(mp.packet_getter.get_str(packet)))
```

----------------------------------------

TITLE: Configuring and Running MediaPipe Face Mesh with Video Input - Android Java
DESCRIPTION: This snippet demonstrates how to initialize MediaPipe Face Mesh for video input on Android. It configures `FaceMeshOptions` for GPU usage and landmark refinement, sets up `VideoInput` to feed frames to the Face Mesh solution, and integrates `SolutionGlSurfaceView` for OpenGL rendering of results. It also shows how to listen for results, render them, and handle video selection from the device's media store using `ActivityResultLauncher`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md#_snippet_6

LANGUAGE: Java
CODE:
```
// For video input and result rendering with OpenGL.
FaceMeshOptions faceMeshOptions =
    FaceMeshOptions.builder()
        .setStaticImageMode(false)
        .setRefineLandmarks(true)
        .setMaxNumFaces(1)
        .setRunOnGpu(true).build();
FaceMesh faceMesh = new FaceMesh(this, faceMeshOptions);
faceMesh.setErrorListener(
    (message, e) -> Log.e(TAG, "MediaPipe Face Mesh error:" + message));

// Initializes a new VideoInput instance and connects it to MediaPipe Face Mesh Solution.
VideoInput videoInput = new VideoInput(this);
videoInput.setNewFrameListener(
    textureFrame -> faceMesh.send(textureFrame));

// Initializes a new GlSurfaceView with a ResultGlRenderer<FaceMeshResult> instance
// that provides the interfaces to run user-defined OpenGL rendering code.
// See mediapipe/examples/android/solutions/facemesh/src/main/java/com/google/mediapipe/examples/facemesh/FaceMeshResultGlRenderer.java
// as an example.
SolutionGlSurfaceView<FaceMeshResult> glSurfaceView =
    new SolutionGlSurfaceView<>(
        this, faceMesh.getGlContext(), faceMesh.getGlMajorVersion());
glSurfaceView.setSolutionResultRenderer(new FaceMeshResultGlRenderer());
glSurfaceView.setRenderInputImage(true);

faceMesh.setResultListener(
    faceMeshResult -> {
      NormalizedLandmark noseLandmark =
          result.multiFaceLandmarks().get(0).getLandmarkList().get(1);
      Log.i(
          TAG,
          String.format(
              "MediaPipe Face Mesh nose normalized coordinates (value range: [0, 1]): x=%f, y=%f",
              noseLandmark.getX(), noseLandmark.getY()));
      // Request GL rendering.
      glSurfaceView.setRenderData(faceMeshResult);
      glSurfaceView.requestRender();
    });

ActivityResultLauncher<Intent> videoGetter =
    registerForActivityResult(
        new ActivityResultContracts.StartActivityForResult(),
        result -> {
          Intent resultIntent = result.getData();
          if (resultIntent != null) {
            if (result.getResultCode() == RESULT_OK) {
              glSurfaceView.post(
                  () ->
                      videoInput.start(
                          this,
                          resultIntent.getData(),
                          faceMesh.getGlContext(),
                          glSurfaceView.getWidth(),
                          glSurfaceView.getHeight()));
            }
          }
        });
Intent pickVideoIntent = new Intent(Intent.ACTION_PICK);
pickVideoIntent.setDataAndType(MediaStore.Video.Media.INTERNAL_CONTENT_URI, "video/*");
videoGetter.launch(pickVideoIntent);
```

----------------------------------------

TITLE: Implementing MediaPipe Pose Tracking in JavaScript
DESCRIPTION: This JavaScript code initializes MediaPipe Pose, handles video input, processes pose detection results, and draws them onto a canvas. It configures the Pose model with options like modelComplexity, smoothLandmarks, and enableSegmentation, and uses the Camera utility to feed video frames to the model for real-time processing and visualization.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md#_snippet_3

LANGUAGE: JavaScript
CODE:
```
<script type="module">
const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
const landmarkContainer = document.getElementsByClassName('landmark-grid-container')[0];
const grid = new LandmarkGrid(landmarkContainer);

function onResults(results) {
  if (!results.poseLandmarks) {
    grid.updateLandmarks([]);
    return;
  }

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.segmentationMask, 0, 0,
                      canvasElement.width, canvasElement.height);

  // Only overwrite existing pixels.
  canvasCtx.globalCompositeOperation = 'source-in';
  canvasCtx.fillStyle = '#00FF00';
  canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);

  // Only overwrite missing pixels.
  canvasCtx.globalCompositeOperation = 'destination-atop';
  canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);

  canvasCtx.globalCompositeOperation = 'source-over';
  drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS,
                 {color: '#00FF00', lineWidth: 4});
  drawLandmarks(canvasCtx, results.poseLandmarks,
                {color: '#FF0000', lineWidth: 2});
  canvasCtx.restore();

  grid.updateLandmarks(results.poseWorldLandmarks);
}

const pose = new Pose({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
}});
pose.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  enableSegmentation: true,
  smoothSegmentation: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
pose.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await pose.send({image: videoElement});
  },
  width: 1280,
  height: 720
});
camera.start();
</script>
```

----------------------------------------

TITLE: Implementing Camera Initialization with CameraXPreviewHelper (Java)
DESCRIPTION: This Java method, `startCamera()`, initializes `CameraXPreviewHelper` and sets a listener. When the camera starts, the listener receives the `SurfaceTexture` for frames, which is then saved, and the `previewDisplayView` is made visible to show the live preview.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_19

LANGUAGE: Java
CODE:
```
public void startCamera() {
  cameraHelper = new CameraXPreviewHelper();
  cameraHelper.setOnCameraStartedListener(
    surfaceTexture -> {
      previewFrameTexture = surfaceTexture;
      // Make the display view visible to start showing the preview.
      previewDisplayView.setVisibility(View.VISIBLE);
    });
}
```

----------------------------------------

TITLE: MediaPipe Python Project Dependencies (requirements_lock.txt)
DESCRIPTION: This snippet provides the complete list of Python packages and their specific versions, including transitive dependencies, required for the MediaPipe project. It is an autogenerated lock file (`requirements_lock.txt`) created by `pip-compile` to ensure reproducible and consistent build environments across different systems.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/requirements_lock.txt#_snippet_0

LANGUAGE: Python requirements.txt
CODE:
```
#
# This file is autogenerated by pip-compile with Python 3.9
# by the following command:
#
#    pip-compile --output-file=mediapipe/opensource_only/requirements_lock.txt mediapipe/opensource_only/requirements.txt
#
absl-py==2.1.0
    # via -r mediapipe/opensource_only/requirements.txt
attrs==24.2.0
    # via -r mediapipe/opensource_only/requirements.txt
cffi==1.17.1
    # via sounddevice
contourpy==1.3.0
    # via matplotlib
cycler==0.12.1
    # via matplotlib
flatbuffers==24.3.25
    # via -r mediapipe/opensource_only/requirements.txt
fonttools==4.54.1
    # via matplotlib
importlib-metadata==8.5.0
    # via jax
importlib-resources==6.4.5
    # via matplotlib
jax==0.4.30
    # via -r mediapipe/opensource_only/requirements.txt
jaxlib==0.4.30
    # via
    #   -r mediapipe/opensource_only/requirements.txt
    #   jax
kiwisolver==1.4.7
    # via matplotlib
matplotlib==3.9.2
    # via -r mediapipe/opensource_only/requirements.txt
ml-dtypes==0.5.0
    # via
    #   jax
    #   jaxlib
numpy==1.26.4
    # via
    #   -r mediapipe/opensource_only/requirements.txt
    #   contourpy
    #   jax
    #   jaxlib
    #   matplotlib
    #   ml-dtypes
    #   opencv-contrib-python
    #   scipy
opencv-contrib-python==4.10.0.84
    # via -r mediapipe/opensource_only/requirements.txt
opt-einsum==3.4.0
    # via jax
packaging==24.1
    # via matplotlib
pillow==10.4.0
    # via matplotlib
protobuf==4.25.5
    # via -r mediapipe/opensource_only/requirements.txt
pycparser==2.22
    # via cffi
pyparsing==3.1.4
    # via matplotlib
python-dateutil==2.9.0.post0
    # via matplotlib
scipy==1.13.1
    # via
    #   jax
    #   jaxlib
sentencepiece==0.2.0
    # via -r mediapipe/opensource_only/requirements.txt
six==1.16.0
    # via python-dateutil
sounddevice==0.5.0
    # via -r mediapipe/opensource_only/requirements.txt
zipp==3.20.2
    # via
    #   importlib-metadata
    #   importlib-resources
```

----------------------------------------

TITLE: Detecting Hand Landmarks with MediaPipe Hand Landmarker (JavaScript)
DESCRIPTION: This snippet initializes the MediaPipe Hand Landmarker task, loading the WASM files and a pre-trained model. It then detects hand landmarks on an HTML image element, which can be used for localizing key points and rendering visual effects over hands. Requires the @mediapipe/tasks-vision library.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md#_snippet_4

LANGUAGE: JavaScript
CODE:
```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const handLandmarker = await HandLandmarker.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
);
const image = document.getElementById("image") as HTMLImageElement;
const landmarks = handLandmarker.detect(image);
```

----------------------------------------

TITLE: Setting Up HTML for MediaPipe Hands JavaScript
DESCRIPTION: This HTML snippet provides the basic structure for integrating MediaPipe Hands in a web application. It includes necessary script imports from MediaPipe's CDN for camera utilities, control utilities, drawing utilities, and the Hands solution itself. It also defines video and canvas elements for input and output display.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md#_snippet_2

LANGUAGE: HTML
CODE:
```
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
</head>

<body>
  <div class="container">
    <video class="input_video"></video>
    <canvas class="output_canvas" width="1280px" height="720px"></canvas>
  </div>
</body>
</html>
```

----------------------------------------

TITLE: Installing MediaPipe and Launching Python Interpreter
DESCRIPTION: Within the activated virtual environment, this sequence of commands first installs the MediaPipe Python package using pip, then launches the Python interpreter, making MediaPipe available for immediate use.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python.md#_snippet_1

LANGUAGE: Bash
CODE:
```
(mp_env)$ pip install mediapipe
(mp_env)$ python3
```

----------------------------------------

TITLE: Real-time 3D Object Detection from Webcam with MediaPipe Objectron (Python)
DESCRIPTION: This snippet illustrates how to perform real-time 3D object detection using MediaPipe Objectron with live webcam input. It initializes the `Objectron` model for video processing, continuously captures frames, processes them, and then draws the detected 2D landmarks and 3D axes before displaying the annotated video feed.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_1

LANGUAGE: Python
CODE:
```
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.99,
                            model_name='Shoe') as objectron:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = objectron.process(image)

    # Draw the box landmarks on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detected_objects:
        for detected_object in results.detected_objects:
            mp_drawing.draw_landmarks(
              image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_drawing.draw_axis(image, detected_object.rotation,
                                 detected_object.translation)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Objectron', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
```

----------------------------------------

TITLE: Defining MediaPipe Calculator Nodes in Graph Configuration
DESCRIPTION: This snippet illustrates how to define calculator nodes within a MediaPipe graph configuration. It specifies the calculator type, its input streams, and its output streams, demonstrating both simple stream naming and tag-indexed naming for multiple outputs.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/calculators.md#_snippet_2

LANGUAGE: MediaPipe Graph Configuration
CODE:
```
node {
  calculator: "SomeAudioVideoCalculator"
  input_stream: "combined_input"
  output_stream: "VIDEO:video_stream"
  output_stream: "AUDIO:0:audio_left"
  output_stream: "AUDIO:1:audio_right"
}

node {
  calculator: "SomeAudioCalculator"
  input_stream: "audio_left"
  input_stream: "audio_right"
  output_stream: "audio_energy"
}
```

----------------------------------------

TITLE: Connecting Calculator Streams by Tag in MediaPipe Graph (Proto)
DESCRIPTION: This snippet demonstrates how to connect calculator input and output streams in a MediaPipe graph configuration using named streams and tags. It shows `SomeAudioVideoCalculator`'s `VIDEO` output connecting to `SomeVideoCalculator`'s `VIDEO_IN` input via `video_stream`, illustrating basic stream routing.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/calculators.md#_snippet_1

LANGUAGE: Protocol Buffer
CODE:
```
# Graph describing calculator SomeAudioVideoCalculator
node {
  calculator: "SomeAudioVideoCalculator"
  input_stream: "INPUT:combined_input"
  output_stream: "VIDEO:video_stream"
}
node {
  calculator: "SomeVideoCalculator"
  input_stream: "VIDEO_IN:video_stream"
  output_stream: "VIDEO_OUT:processed_video"
}
```

----------------------------------------

TITLE: Real-time Face Mesh Detection with MediaPipe JavaScript
DESCRIPTION: This JavaScript code initializes MediaPipe Face Mesh for real-time face landmark detection. It sets up a video stream as input, processes frames using the `FaceMesh` class, and draws the detected landmarks on a canvas. The `onResults` function handles the output, clearing the canvas and drawing the image along with multi-face landmarks using `drawConnectors` from the MediaPipe drawing utilities.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md#_snippet_3

LANGUAGE: JavaScript
CODE:
```
const videoElement = document.getElementsByClassName('input_video')[0];\nconst canvasElement = document.getElementsByClassName('output_canvas')[0];\nconst canvasCtx = canvasElement.getContext('2d');\n\nfunction onResults(results) {\n  canvasCtx.save();\n  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);\n  canvasCtx.drawImage(\n      results.image, 0, 0, canvasElement.width, canvasElement.height);\n  if (results.multiFaceLandmarks) {\n    for (const landmarks of results.multiFaceLandmarks) {\n      drawConnectors(canvasCtx, landmarks, FACEMESH_TESSELATION,\n                     {color: '#C0C0C070', lineWidth: 1});\n      drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYE, {color: '#FF3030'});\n      drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYEBROW, {color: '#FF3030'});\n      drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_IRIS, {color: '#FF3030'});\n      drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYE, {color: '#30FF30'});\n      drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYEBROW, {color: '#30FF30'});\n      drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_IRIS, {color: '#30FF30'});\n      drawConnectors(canvasCtx, landmarks, FACEMESH_FACE_OVAL, {color: '#E0E0E0'});\n      drawConnectors(canvasCtx, landmarks, FACEMESH_LIPS, {color: '#E0E0E0'});\n    }\n  }\n  canvasCtx.restore();\n}\n\nconst faceMesh = new FaceMesh({locateFile: (file) => {\n  return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;\n}});\nfaceMesh.setOptions({\n  maxNumFaces: 1,\n  refineLandmarks: true,\n  minDetectionConfidence: 0.5,\n  minTrackingConfidence: 0.5\n});\nfaceMesh.onResults(onResults);\n\nconst camera = new Camera(videoElement, {\n  onFrame: async () => {\n    await faceMesh.send({image: videoElement});\n  },\n  width: 1280,\n  height: 720\n});\ncamera.start();
```

----------------------------------------

TITLE: Real-time Face Mesh Detection with Webcam in Python
DESCRIPTION: This Python snippet demonstrates how to perform real-time face mesh detection using MediaPipe and OpenCV. It captures video from a webcam, processes each frame to detect facial landmarks, and then draws the tessellation, contours, and iris connections on the image before displaying it. It handles frame reading, color conversion, and display.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md#_snippet_1

LANGUAGE: Python
CODE:
```
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
```

----------------------------------------

TITLE: Processing Webcam Input with MediaPipe Hands in Python
DESCRIPTION: This snippet demonstrates how to capture live video from a webcam, process each frame with MediaPipe Hands for hand landmark detection, and display the results. It initializes the MediaPipe Hands model with specified confidence levels and handles frame reading, color conversion, and drawing annotations.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md#_snippet_1

LANGUAGE: Python
CODE:
```
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
```

----------------------------------------

TITLE: Processing Static Images with MediaPipe Holistic in Python
DESCRIPTION: This snippet demonstrates how to use MediaPipe Holistic to process a list of static images. It initializes the Holistic model with segmentation and face landmark refinement enabled, then processes each image to detect pose, face, and hand landmarks, and applies segmentation. It outputs annotated images and prints nose coordinates.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md#_snippet_0

LANGUAGE: Python
CODE:
```
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    refine_face_landmarks=True) as holistic:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
      print(
          f'Nose coordinates: ('
          f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
          f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
      )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose, left and right hands, and face landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.
        get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
```

----------------------------------------

TITLE: Building and Copying MediaPipe Assets to Android Project - Bash
DESCRIPTION: These commands build the MediaPipe binary graph using Bazel and then copy both the compiled binary graph (`.binarypb`) and the associated TensorFlow Lite model (`.tflite`) into the `app/src/main/assets` directory. These assets are crucial for MediaPipe's runtime operation within the Android application.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/android_archive_library.md#_snippet_5

LANGUAGE: bash
CODE:
```
bazel build -c opt mediapipe/graphs/face_detection:face_detection_mobile_gpu_binary_graph
cp bazel-bin/mediapipe/graphs/face_detection/face_detection_mobile_gpu.binarypb /path/to/your/app/src/main/assets/
cp mediapipe/modules/face_detection/face_detection_short_range.tflite /path/to/your/app/src/main/assets/
```

----------------------------------------

TITLE: Detecting Pose Landmarks with MediaPipe Pose Landmarker (JavaScript)
DESCRIPTION: This snippet demonstrates how to initialize the MediaPipe Pose Landmarker to detect body pose landmarks in an image. It involves loading the vision tasks WASM module and creating the landmarker from a model path. The `detect` method processes an HTML image element to return landmark results.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md#_snippet_11

LANGUAGE: JavaScript
CODE:
```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const poseLandmarker = await PoseLandmarker.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
);
const image = document.getElementById("image") as HTMLImageElement;
const landmarks = poseLandmarker.detect(image);
```

----------------------------------------

TITLE: Segmenting Images with MediaPipe Image Segmenter (JavaScript)
DESCRIPTION: This snippet shows how to initialize the MediaPipe Image Segmenter and perform image segmentation. It requires loading the vision tasks WASM module and creating the segmenter from a model path. The `segment` method processes an HTML image element and provides masks, width, and height via a callback function.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md#_snippet_8

LANGUAGE: JavaScript
CODE:
```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const imageSegmenter = await ImageSegmenter.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite"
);
const image = document.getElementById("image") as HTMLImageElement;
imageSegmenter.segment(image, (masks, width, height) => {
  ...
});
```

----------------------------------------

TITLE: Including MediaPipe Libraries via CDN in HTML
DESCRIPTION: This HTML snippet shows how to include MediaPipe utility and solution libraries directly into a web page using a Content Delivery Network (CDN) like jsDelivr. By adding these <script> tags within the <head> section, the libraries become globally available in the browser, eliminating the need for local installation via npm. This method is useful for quick prototyping or when a full build system is not desired.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/javascript.md#_snippet_1

LANGUAGE: HTML
CODE:
```
<head>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.1/drawing_utils.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.1/holistic.js" crossorigin="anonymous"></script>
</head>
```

----------------------------------------

TITLE: Installing Xcode Command Line Tools (Bash)
DESCRIPTION: This command installs the Xcode Command Line Tools, which are essential for various development tasks on macOS, including compiling code and using command-line utilities required by MediaPipe.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/ios.md#_snippet_0

LANGUAGE: bash
CODE:
```
xcode-select --install
```

----------------------------------------

TITLE: Processing Image Input with MediaPipe Face Detection in Android
DESCRIPTION: This snippet demonstrates how to initialize MediaPipe Face Detection for static image mode, set up listeners for processing results and errors, and integrate with an ActivityResultLauncher to select and send images from the device gallery for detection. It shows how to draw detection results on a custom ImageView.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_detection.md#_snippet_6

LANGUAGE: Java
CODE:
```
// For reading images from gallery and drawing the output in an ImageView.
FaceDetectionOptions faceDetectionOptions =
    FaceDetectionOptions.builder()
        .setStaticImageMode(true)
        .setModelSelection(0).build();
FaceDetection faceDetection = new FaceDetection(this, faceDetectionOptions);

// Connects MediaPipe Face Detection Solution to the user-defined ImageView
// instance that allows users to have the custom drawing of the output landmarks
// on it. See mediapipe/examples/android/solutions/facedetection/src/main/java/com/google/mediapipe/examples/facedetection/FaceDetectionResultImageView.java
// as an example.
FaceDetectionResultImageView imageView = new FaceDetectionResultImageView(this);
faceDetection.setResultListener(
    faceDetectionResult -> {
      if (faceDetectionResult.multiFaceDetections().isEmpty()) {
        return;
      }
      int width = faceDetectionResult.inputBitmap().getWidth();
      int height = faceDetectionResult.inputBitmap().getHeight();
      RelativeKeypoint noseTip =
          faceDetectionResult
              .multiFaceDetections()
              .get(0)
              .getLocationData()
              .getRelativeKeypoints(FaceKeypoint.NOSE_TIP);
      Log.i(
          TAG,
          String.format(
              "MediaPipe Face Detection nose tip coordinates (pixel values): x=%f, y=%f",
              noseTip.getX() * width, noseTip.getY() * height));
      // Request canvas drawing.
      imageView.setFaceDetectionResult(faceDetectionResult);
      runOnUiThread(() -> imageView.update());
    });
faceDetection.setErrorListener(
    (message, e) -> Log.e(TAG, "MediaPipe Face Detection error:" + message));

// ActivityResultLauncher to get an image from the gallery as Bitmap.
ActivityResultLauncher<Intent> imageGetter =
    registerForActivityResult(
        new ActivityResultContracts.StartActivityForResult(),
        result -> {
          Intent resultIntent = result.getData();
          if (resultIntent != null && result.getResultCode() == RESULT_OK) {
            Bitmap bitmap = null;
            try {
              bitmap =
                  MediaStore.Images.Media.getBitmap(
                      this.getContentResolver(), resultIntent.getData());
              // Please also rotate the Bitmap based on its orientation.
            } catch (IOException e) {
              Log.e(TAG, "Bitmap reading error:" + e);
            }
            if (bitmap != null) {
              faceDetection.send(bitmap);
            }
          }
        });
Intent pickImageIntent = new Intent(Intent.ACTION_PICK);
pickImageIntent.setDataAndType(MediaStore.Images.Media.INTERNAL_CONTENT_URI, "image/*");
imageGetter.launch(pickImageIntent);
```

----------------------------------------

TITLE: Real-time Webcam Processing with MediaPipe Holistic in Python
DESCRIPTION: This snippet illustrates how to use MediaPipe Holistic for real-time processing of webcam input. It initializes the Holistic model with confidence thresholds, captures video frames, processes them to detect face and pose landmarks, and displays the annotated frames in a live window. The image is flipped for a selfie-view display.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md#_snippet_1

LANGUAGE: Python
CODE:
```
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
```

----------------------------------------

TITLE: Defining a Simple MediaPipe Graph in Proto
DESCRIPTION: This snippet defines a basic MediaPipe CalculatorGraphConfig in Protocol Buffer format. It specifies input_stream and input_side_packet, an output_stream, and a single InferenceCalculator node. The node is configured to use a GPU delegate via InferenceCalculatorOptions.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_0

LANGUAGE: Protocol Buffer
CODE:
```
# Graph inputs.
input_stream: "input_tensors"
input_side_packet: "model"

# Graph outputs.
output_stream: "output_tensors"

node {
  calculator: "InferenceCalculator"
  input_stream: "TENSORS:input_tensors"
  input_side_packet: "MODEL:model"
  output_stream: "TENSORS:output_tensors"
  node_options: {
    [type.googleapis.com/mediapipe.InferenceCalculatorOptions] {
      # Requesting GPU delegate.
      delegate { gpu {} }
    }
  }
}
```

----------------------------------------

TITLE: Initializing MediaPipe Face Detection for Video Input (Java)
DESCRIPTION: This Java snippet demonstrates the complete setup for MediaPipe Face Detection on Android, enabling video input processing and OpenGL rendering. It initializes the FaceDetection model for video mode, sets up a VideoInput to feed frames, configures a SolutionGlSurfaceView for result visualization, and includes an ActivityResultLauncher to select and process a video from the device's media store.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_detection.md#_snippet_7

LANGUAGE: Java
CODE:
```
// For video input and result rendering with OpenGL.
FaceDetectionOptions faceDetectionOptions =
    FaceDetectionOptions.builder()
        .setStaticImageMode(false)
        .setModelSelection(0).build();
FaceDetection faceDetection = new FaceDetection(this, faceDetectionOptions);
faceDetection.setErrorListener(
    (message, e) -> Log.e(TAG, "MediaPipe Face Detection error:" + message));

// Initializes a new VideoInput instance and connects it to MediaPipe Face Detection Solution.
VideoInput videoInput = new VideoInput(this);
videoInput.setNewFrameListener(
    textureFrame -> faceDetection.send(textureFrame));

// Initializes a new GlSurfaceView with a ResultGlRenderer<FaceDetectionResult> instance
// that provides the interfaces to run user-defined OpenGL rendering code.
// See mediapipe/examples/android/solutions/facedetection/src/main/java/com/google/mediapipe/examples/facedetection/FaceDetectionResultGlRenderer.java
// as an example.
SolutionGlSurfaceView<FaceDetectionResult> glSurfaceView =
    new SolutionGlSurfaceView<>(
        this, faceDetection.getGlContext(), faceDetection.getGlMajorVersion());
glSurfaceView.setSolutionResultRenderer(new FaceDetectionResultGlRenderer());
glSurfaceView.setRenderInputImage(true);

faceDetection.setResultListener(
    faceDetectionResult -> {
      if (faceDetectionResult.multiFaceDetections().isEmpty()) {
        return;
      }
      RelativeKeypoint noseTip =
          faceDetectionResult
              .multiFaceDetections()
              .get(0)
              .getLocationData()
              .getRelativeKeypoints(FaceKeypoint.NOSE_TIP);
      Log.i(
          TAG,
          String.format(
              "MediaPipe Face Detection nose tip normalized coordinates (value range: [0, 1]): x=%f, y=%f",
              noseTip.getX(), noseTip.getY()));
      // Request GL rendering.
      glSurfaceView.setRenderData(faceDetectionResult);
      glSurfaceView.requestRender();
    });

ActivityResultLauncher<Intent> videoGetter =
    registerForActivityResult(
        new ActivityResultContracts.StartActivityForResult(),
        result -> {
          Intent resultIntent = result.getData();
          if (resultIntent != null) {
            if (result.getResultCode() == RESULT_OK) {
              glSurfaceView.post(
                  () ->
                      videoInput.start(
                          this,
                          resultIntent.getData(),
                          faceDetection.getGlContext(),
                          glSurfaceView.getWidth(),
                          glSurfaceView.getHeight()));
            }
          }
        });
Intent pickVideoIntent = new Intent(Intent.ACTION_PICK);
pickVideoIntent.setDataAndType(MediaStore.Video.Media.INTERNAL_CONTENT_URI, "video/*");
videoGetter.launch(pickVideoIntent);
```

----------------------------------------

TITLE: Configuring MediaPipe Face Detection with Camera Input and OpenGL Rendering in Java
DESCRIPTION: This snippet demonstrates the complete setup for real-time face detection using MediaPipe on Android. It initializes FaceDetection with specific options, configures CameraInput to feed frames, and sets up a SolutionGlSurfaceView for OpenGL rendering of the detection results, including logging nose tip coordinates.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_detection.md#_snippet_5

LANGUAGE: Java
CODE:
```
// For camera input and result rendering with OpenGL.
FaceDetectionOptions faceDetectionOptions =
    FaceDetectionOptions.builder()
        .setStaticImageMode(false)
        .setModelSelection(0).build();
FaceDetection faceDetection = new FaceDetection(this, faceDetectionOptions);
faceDetection.setErrorListener(
    (message, e) -> Log.e(TAG, "MediaPipe Face Detection error:" + message));

// Initializes a new CameraInput instance and connects it to MediaPipe Face Detection Solution.
CameraInput cameraInput = new CameraInput(this);
cameraInput.setNewFrameListener(
    textureFrame -> faceDetection.send(textureFrame));

// Initializes a new GlSurfaceView with a ResultGlRenderer<FaceDetectionResult> instance
// that provides the interfaces to run user-defined OpenGL rendering code.
// See mediapipe/examples/android/solutions/facedetection/src/main/java/com/google/mediapipe/examples/facedetection/FaceDetectionResultGlRenderer.java
// as an example.
SolutionGlSurfaceView<FaceDetectionResult> glSurfaceView =
    new SolutionGlSurfaceView<>(
        this, faceDetection.getGlContext(), faceDetection.getGlMajorVersion());
glSurfaceView.setSolutionResultRenderer(new FaceDetectionResultGlRenderer());
glSurfaceView.setRenderInputImage(true);
faceDetection.setResultListener(
    faceDetectionResult -> {
      if (faceDetectionResult.multiFaceDetections().isEmpty()) {
        return;
      }
      RelativeKeypoint noseTip =
          faceDetectionResult
              .multiFaceDetections()
              .get(0)
              .getLocationData()
              .getRelativeKeypoints(FaceKeypoint.NOSE_TIP);
      Log.i(
          TAG,
          String.format(
              "MediaPipe Face Detection nose tip normalized coordinates (value range: [0, 1]): x=%f, y=%f",
              noseTip.getX(), noseTip.getY()));
      // Request GL rendering.
      glSurfaceView.setRenderData(faceDetectionResult);
      glSurfaceView.requestRender();
    });

// The runnable to start camera after the GLSurfaceView is attached.
glSurfaceView.post(
    () ->
        cameraInput.start(
            this,
            faceDetection.getGlContext(),
            CameraInput.CameraFacing.FRONT,
            glSurfaceView.getWidth(),
            glSurfaceView.getHeight()));
```

----------------------------------------

TITLE: Defining MediaPipe AAR Target for Face Detection (Bazel)
DESCRIPTION: This snippet defines a `mediapipe_aar` target in a Bazel BUILD file. It specifies the name of the AAR and lists the required calculator dependencies, in this case, `mobile_calculators` for MediaPipe Face Detection. This custom target is essential for generating an AAR that includes only the necessary MediaPipe components for a specific project.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/android_archive_library.md#_snippet_0

LANGUAGE: Bazel
CODE:
```
load("//mediapipe/java/com/google/mediapipe:mediapipe_aar.bzl", "mediapipe_aar")

mediapipe_aar(
    name = "mediapipe_face_detection",
    calculators = ["//mediapipe/graphs/face_detection:mobile_calculators"],
)
```

----------------------------------------

TITLE: Building a MediaPipe Graph Programmatically in C++
DESCRIPTION: This C++ function, BuildGraph, demonstrates how to construct the equivalent CalculatorGraphConfig shown in the proto example. It uses the Graph builder API to define graph inputs (input_tensors, model), add an InferenceCalculator node, configure its options for GPU delegation, and connect streams to define graph outputs.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_1

LANGUAGE: C++
CODE:
```
CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Graph inputs.
  Stream<std::vector<Tensor>> input_tensors =
      graph.In(0).SetName("input_tensors").Cast<std::vector<Tensor>>();
  SidePacket<TfLiteModelPtr> model =
      graph.SideIn(0).SetName("model").Cast<TfLiteModelPtr>();

  auto& inference_node = graph.AddNode("InferenceCalculator");
  auto& inference_opts =
      inference_node.GetOptions<InferenceCalculatorOptions>();
  // Requesting GPU delegate.
  inference_opts.mutable_delegate()->mutable_gpu();
  input_tensors.ConnectTo(inference_node.In("TENSORS"));
  model.ConnectTo(inference_node.SideIn("MODEL"));
  Stream<std::vector<Tensor>> output_tensors =
      inference_node.Out("TENSORS").Cast<std::vector<Tensor>>();

  // Graph outputs.
  output_tensors.SetName("output_tensors").ConnectTo(graph.Out(0));

  // Get `CalculatorGraphConfig` to pass it into `CalculatorGraph`
  return graph.GetConfig();
}
```

----------------------------------------

TITLE: Real-time Face Detection in Browser with MediaPipe Face Detection (JavaScript)
DESCRIPTION: This JavaScript module implements real-time face detection using MediaPipe in a web browser. It sets up a `FaceDetection` instance, configures options like model and confidence, and defines an `onResults` callback to draw detections on a canvas. A `Camera` utility streams video frames to the detection model, enabling live processing and visualization. It relies on the HTML structure provided previously.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_detection.md#_snippet_4

LANGUAGE: JavaScript
CODE:
```
<script type="module">
const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
const drawingUtils = window;

function onResults(results) {
  // Draw the overlays.
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);
  if (results.detections.length > 0) {
    drawingUtils.drawRectangle(
        canvasCtx, results.detections[0].boundingBox,
        {color: 'blue', lineWidth: 4, fillColor: '#00000000'});
    drawingUtils.drawLandmarks(canvasCtx, results.detections[0].landmarks, {
      color: 'red',
      radius: 5,
    });
  }
  canvasCtx.restore();
}

const faceDetection = new FaceDetection({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection@0.0/${file}`;
}});
faceDetection.setOptions({
  model: 'short',
  minDetectionConfidence: 0.5
});
faceDetection.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await faceDetection.send({image: videoElement});
  },
  width: 1280,
  height: 720
});
camera.start();
</script>
```

----------------------------------------

TITLE: Detecting Objects with MediaPipe Object Detector (JavaScript)
DESCRIPTION: This snippet illustrates the initialization and usage of the MediaPipe Object Detector to identify objects within an image. It requires loading the vision tasks WASM module and creating the detector from a model path. The `detect` method processes an HTML image element to return detection results.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md#_snippet_10

LANGUAGE: JavaScript
CODE:
```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const objectDetector = await ObjectDetector.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
);
const image = document.getElementById("image") as HTMLImageElement;
const detections = objectDetector.detect(image);
```

----------------------------------------

TITLE: Applying Selfie Segmentation to Webcam Input (Python)
DESCRIPTION: This snippet demonstrates real-time selfie segmentation using a webcam feed. It initializes the `SelfieSegmentation` model with `model_selection=1` for a more lightweight model suitable for live processing. It continuously captures frames, processes them, and displays the segmented output, allowing for a customizable background.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/selfie_segmentation.md#_snippet_2

LANGUAGE: Python
CODE:
```
BG_COLOR = (192, 192, 192) # gray
cap = cv2.VideoCapture(0)
with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=1) as selfie_segmentation:
  bg_image = None
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = selfie_segmentation.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack(
      (results.segmentation_mask,) * 3, axis=-1) > 0.1
    # The background can be customized.
    #   a) Load an image (with the same width and height of the input image) to
    #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
    #   b) Blur the input image by applying image filtering, e.g.,
    #      bg_image = cv2.GaussianBlur(image,(55,55),0)
    if bg_image is None:
      bg_image = np.zeros(image.shape, dtype=np.uint8)
      bg_image[:] = BG_COLOR
    output_image = np.where(condition, image, bg_image)

    cv2.imshow('MediaPipe Selfie Segmentation', output_image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
```

----------------------------------------

TITLE: Real-time Face Detection from Webcam with MediaPipe Face Detection (Python)
DESCRIPTION: This Python snippet shows how to perform real-time face detection from a webcam feed using MediaPipe. It captures video frames, processes them with the `FaceDetection` model, draws annotations, and displays the output. It handles frame reading, color conversion, and performance optimization by marking images as non-writeable. It requires OpenCV (`cv2`) and MediaPipe.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_detection.md#_snippet_2

LANGUAGE: Python
CODE:
```
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
```

----------------------------------------

TITLE: Initializing FrameProcessor in Android onCreate
DESCRIPTION: Initializes the `FrameProcessor` instance within the `onCreate(Bundle)` method of the Android Activity. It takes the application context, the native EGL context (from `eglManager`), and string names for the binary graph, input video stream, and output video stream, typically retrieved from the application's metadata. This setup configures the processor to work with a specific MediaPipe graph and its I/O streams.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_38

LANGUAGE: Java
CODE:
```
processor =
    new FrameProcessor(
        this,
        eglManager.getNativeContext(),
        applicationInfo.metaData.getString("binaryGraphName"),
        applicationInfo.metaData.getString("inputVideoStreamName"),
        applicationInfo.metaData.getString("outputVideoStreamName"));
```

----------------------------------------

TITLE: Defining CalculatorBase Core Methods in C++
DESCRIPTION: This C++ snippet illustrates the `CalculatorBase` class, showcasing the essential virtual methods (`GetContract`, `Open`, `Process`, `Close`) that custom MediaPipe calculators must implement or override. These methods define the calculator's expected input/output types, initialization logic, data processing, and final cleanup, respectively, governing its lifecycle within a MediaPipe graph.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/calculators.md#_snippet_0

LANGUAGE: C++
CODE:
```
class CalculatorBase {
 public:
  ...

  // The subclasses of CalculatorBase must implement GetContract.
  // ...
  static absl::Status GetContract(CalculatorContract* cc);

  // Open is called before any Process() calls, on a freshly constructed
  // calculator.  Subclasses may override this method to perform necessary
  // setup, and possibly output Packets and/or set output streams' headers.
  // ...
  virtual absl::Status Open(CalculatorContext* cc) {
    return absl::OkStatus();
  }

  // Processes the incoming inputs. May call the methods on cc to access
  // inputs and produce outputs.
  // ...
  virtual absl::Status Process(CalculatorContext* cc) = 0;

  // Is called if Open() was called and succeeded.  Is called either
  // immediately after processing is complete or after a graph run has ended
  // (if an error occurred in the graph).  ...
  virtual absl::Status Close(CalculatorContext* cc) {
    return absl::OkStatus();
  }

  ...
};
```

----------------------------------------

TITLE: Detecting Face Landmarks with MediaPipe Face Landmarker (JavaScript)
DESCRIPTION: This snippet initializes the MediaPipe Face Landmarker task, loading the WASM files and a pre-trained model. It then detects facial landmarks on an HTML image element, which can be used for localizing key points and rendering visual effects. Requires the @mediapipe/tasks-vision library.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md#_snippet_1

LANGUAGE: JavaScript
CODE:
```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const faceLandmarker = await FaceLandmarker.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
);
const image = document.getElementById("image") as HTMLImageElement;
const landmarks = faceLandmarker.detect(image);
```

----------------------------------------

TITLE: Processing Static Images with MediaPipe Hands in Python
DESCRIPTION: This Python code demonstrates how to use MediaPipe's `hands` solution to detect and track hands in static images. It initializes the `Hands` model with `static_image_mode=True`, processes images, prints handedness and landmark data, and draws landmarks on the images using `drawing_utils`. It also shows how to plot world landmarks.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md#_snippet_0

LANGUAGE: Python
CODE:
```
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
```

----------------------------------------

TITLE: Extracting Text Embeddings with MediaPipe Text Embedder (JavaScript)
DESCRIPTION: This snippet shows how to initialize and use the MediaPipe Text Embedder to generate numerical embeddings from text data. It begins by resolving the WASM files, then creates a TextEmbedder instance from a Universal Sentence Encoder model, and finally calls the embed method on the textData.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/text/README.md#_snippet_2

LANGUAGE: JavaScript
CODE:
```
const text = await FilesetResolver.forTextTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text/wasm"
);
const textEmbedder = await TextEmbedder.createFromModelPath(text,
    "https://storage.googleapis.com/mediapipe-models/text_embedder/universal_sentence_encoder/float32/1/universal_sentence_encoder.tflite"
);
const embeddings = textEmbedder.embed(textData);
```

----------------------------------------

TITLE: Building MediaPipe Graph with Utility Functions (C++)
DESCRIPTION: This C++ snippet demonstrates the most readable approach to constructing a MediaPipe CalculatorGraphConfig by encapsulating calculator definitions within separate utility functions (e.g., RunCalculator1). This modularity significantly improves code organization, reusability, and clarity, as each function represents a distinct processing step. The snippet shows the main BuildGraph function calling these helpers and connecting the final outputs.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_19

LANGUAGE: C++
CODE:
```
CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Inputs.
  Stream<A> a = graph.In(0).Cast<A>();

  Stream<B> b = RunCalculator1(a, graph);
  Stream<C> c = RunCalculator2(b, graph);
  Stream<D> d = RunCalculator3(b, c, graph);
  Stream<E> e = RunCalculator4(b, c, d, graph);

  // Outputs.
  b.SetName("b").ConnectTo(graph.Out(0));
  c.SetName("c").ConnectTo(graph.Out(1));
  d.SetName("d").ConnectTo(graph.Out(2));
  e.SetName("e").ConnectTo(graph.Out(3));

  return graph.GetConfig();
}
```

----------------------------------------

TITLE: Detecting Faces in Static Images with MediaPipe Face Detection (Python)
DESCRIPTION: This Python snippet demonstrates how to perform face detection on a list of static image files using MediaPipe. It initializes the `FaceDetection` model, processes each image, converts color formats, draws detected faces, and saves the annotated images. It requires OpenCV (`cv2`) and MediaPipe (`mp_face_detection`, `mp_drawing`).
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_detection.md#_snippet_1

LANGUAGE: Python
CODE:
```
IMAGE_FILES = []
with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face detections of each face.
    if not results.detections:
      continue
    annotated_image = image.copy()
    for detection in results.detections:
      print('Nose tip:')
      print(mp_face_detection.get_key_point(
          detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
      mp_drawing.draw_detection(annotated_image, detection)
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
```

----------------------------------------

TITLE: Processing Static Image Input with MediaPipe Hands (Java)
DESCRIPTION: This snippet illustrates how to use MediaPipe Hands for processing static images from the device gallery. It configures HandsOptions for static image mode, connects the solution to a HandsResultImageView for custom drawing, and uses ActivityResultLauncher to select and send an image Bitmap to the Hands solution for analysis, logging wrist landmark coordinates.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md#_snippet_5

LANGUAGE: java
CODE:
```
// For reading images from gallery and drawing the output in an ImageView.
HandsOptions handsOptions =
    HandsOptions.builder()
        .setStaticImageMode(true)
        .setMaxNumHands(2)
        .setRunOnGpu(true).build();
Hands hands = new Hands(this, handsOptions);

// Connects MediaPipe Hands Solution to the user-defined ImageView instance that
// allows users to have the custom drawing of the output landmarks on it.
// See mediapipe/examples/android/solutions/hands/src/main/java/com/google/mediapipe/examples/hands/HandsResultImageView.java
// as an example.
HandsResultImageView imageView = new HandsResultImageView(this);
hands.setResultListener(
    handsResult -> {
      if (result.multiHandLandmarks().isEmpty()) {
        return;
      }
      int width = handsResult.inputBitmap().getWidth();
      int height = handsResult.inputBitmap().getHeight();
      NormalizedLandmark wristLandmark =
          handsResult.multiHandLandmarks().get(0).getLandmarkList().get(HandLandmark.WRIST);
      Log.i(
          TAG,
          String.format(
              "MediaPipe Hand wrist coordinates (pixel values): x=%f, y=%f",
              wristLandmark.getX() * width, wristLandmark.getY() * height));
      // Request canvas drawing.
      imageView.setHandsResult(handsResult);
      runOnUiThread(() -> imageView.update());
    });
hands.setErrorListener(
    (message, e) -> Log.e(TAG, "MediaPipe Hands error:" + message));

// ActivityResultLauncher to get an image from the gallery as Bitmap.
ActivityResultLauncher<Intent> imageGetter =
    registerForActivityResult(
        new ActivityResultContracts.StartActivityForResult(),
        result -> {
          Intent resultIntent = result.getData();
          if (resultIntent != null && result.getResultCode() == RESULT_OK) {
            Bitmap bitmap = null;
            try {
              bitmap =
                  MediaStore.Images.Media.getBitmap(
                      this.getContentResolver(), resultIntent.getData());
              // Please also rotate the Bitmap based on its orientation.
            } catch (IOException e) {
              Log.e(TAG, "Bitmap reading error:" + e);
            }
            if (bitmap != null) {
              hands.send(bitmap);
            }
          }
        });
Intent pickImageIntent = new Intent(Intent.ACTION_PICK);
pickImageIntent.setDataAndType(MediaStore.Images.Media.INTERNAL_CONTENT_URI, "image/*");
imageGetter.launch(pickImageIntent);
```

----------------------------------------

TITLE: Configuring MediaPipe Face Mesh for Camera Input (Java)
DESCRIPTION: This snippet demonstrates how to set up MediaPipe Face Mesh for real-time camera input on Android. It configures `FaceMeshOptions` for GPU processing and real-time mode, initializes `CameraInput` to feed frames to the solution, and sets up a `SolutionGlSurfaceView` for OpenGL rendering of results. It also includes error and result listeners for logging and rendering updates.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md#_snippet_4

LANGUAGE: Java
CODE:
```
// For camera input and result rendering with OpenGL.
FaceMeshOptions faceMeshOptions =
    FaceMeshOptions.builder()
        .setStaticImageMode(false)
        .setRefineLandmarks(true)
        .setMaxNumFaces(1)
        .setRunOnGpu(true).build();
FaceMesh faceMesh = new FaceMesh(this, faceMeshOptions);
faceMesh.setErrorListener(
    (message, e) -> Log.e(TAG, "MediaPipe Face Mesh error:" + message));

// Initializes a new CameraInput instance and connects it to MediaPipe Face Mesh Solution.
CameraInput cameraInput = new CameraInput(this);
cameraInput.setNewFrameListener(
    textureFrame -> faceMesh.send(textureFrame));

// Initializes a new GlSurfaceView with a ResultGlRenderer<FaceMeshResult> instance
// that provides the interfaces to run user-defined OpenGL rendering code.
// See mediapipe/examples/android/solutions/facemesh/src/main/java/com/google/mediapipe/examples/facemesh/FaceMeshResultGlRenderer.java
// as an example.
SolutionGlSurfaceView<FaceMeshResult> glSurfaceView =
    new SolutionGlSurfaceView<>(
        this, faceMesh.getGlContext(), faceMesh.getGlMajorVersion());
glSurfaceView.setSolutionResultRenderer(new FaceMeshResultGlRenderer());
glSurfaceView.setRenderInputImage(true);

faceMesh.setResultListener(
    faceMeshResult -> {
      NormalizedLandmark noseLandmark =
          result.multiFaceLandmarks().get(0).getLandmarkList().get(1);
      Log.i(
          TAG,
          String.format(
              "MediaPipe Face Mesh nose normalized coordinates (value range: [0, 1]): x=%f, y=%f",
              noseLandmark.getX(), noseLandmark.getY()));
      // Request GL rendering.
      glSurfaceView.setRenderData(faceMeshResult);
      glSurfaceView.requestRender();
    });

// The runnable to start camera after the GLSurfaceView is attached.
glSurfaceView.post(
    () ->
        cameraInput.start(
            this,
            faceMesh.getGlContext(),
            CameraInput.CameraFacing.FRONT,
            glSurfaceView.getWidth(),
            glSurfaceView.getHeight()));
```

----------------------------------------

TITLE: Interactive Image Segmentation with MediaPipe Interactive Segmenter (JavaScript)
DESCRIPTION: This snippet demonstrates how to initialize the MediaPipe Interactive Segmenter for region-of-interest based image segmentation. It involves loading the vision tasks WASM module and creating the segmenter from a model path. The `segment` method processes an HTML image element with a specified keypoint and provides masks, width, and height via a callback.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md#_snippet_9

LANGUAGE: JavaScript
CODE:
```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const interactiveSegmenter = await InteractiveSegmenter.createFromModelPath(
    vision,
    "https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite"
);
const image = document.getElementById("image") as HTMLImageElement;
interactiveSegmenter.segment(image, { keypoint: { x: 0.1, y: 0.2 } },
    (masks, width, height) => { ... }
);
```

----------------------------------------

TITLE: Configuring MediaPipe Hands for Android Video Input
DESCRIPTION: This Java snippet demonstrates the setup for MediaPipe Hands to process video streams on Android. It initializes the `Hands` solution with specific options (e.g., GPU usage, max hands), configures a `VideoInput` to feed frames to the solution, and sets up a `SolutionGlSurfaceView` for OpenGL rendering of the results. It also includes an `ActivityResultLauncher` to allow users to pick a video from their device and start processing it.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md#_snippet_6

LANGUAGE: java
CODE:
```
// For video input and result rendering with OpenGL.
HandsOptions handsOptions =
    HandsOptions.builder()
        .setStaticImageMode(false)
        .setMaxNumHands(2)
        .setRunOnGpu(true).build();
Hands hands = new Hands(this, handsOptions);
hands.setErrorListener(
    (message, e) -> Log.e(TAG, "MediaPipe Hands error:" + message));

// Initializes a new VideoInput instance and connects it to MediaPipe Hands Solution.
VideoInput videoInput = new VideoInput(this);
videoInput.setNewFrameListener(
    textureFrame -> hands.send(textureFrame));

// Initializes a new GlSurfaceView with a ResultGlRenderer<HandsResult> instance
// that provides the interfaces to run user-defined OpenGL rendering code.
// See mediapipe/examples/android/solutions/hands/src/main/java/com/google/mediapipe/examples/hands/HandsResultGlRenderer.java
// as an example.
SolutionGlSurfaceView<HandsResult> glSurfaceView =
    new SolutionGlSurfaceView<>(
        this, hands.getGlContext(), hands.getGlMajorVersion());
glSurfaceView.setSolutionResultRenderer(new HandsResultGlRenderer());
glSurfaceView.setRenderInputImage(true);

hands.setResultListener(
    handsResult -> {
      if (result.multiHandLandmarks().isEmpty()) {
        return;
      }
      NormalizedLandmark wristLandmark =
          handsResult.multiHandLandmarks().get(0).getLandmarkList().get(HandLandmark.WRIST);
      Log.i(
          TAG,
          String.format(
              "MediaPipe Hand wrist normalized coordinates (value range: [0, 1]): x=%f, y=%f",
              wristLandmark.getX(), wristLandmark.getY()));
      // Request GL rendering.
      glSurfaceView.setRenderData(handsResult);
      glSurfaceView.requestRender();
    });

ActivityResultLauncher<Intent> videoGetter =
    registerForActivityResult(
        new ActivityResultContracts.StartActivityForResult(),
        result -> {
          Intent resultIntent = result.getData();
          if (resultIntent != null) {
            if (result.getResultCode() == RESULT_OK) {
              glSurfaceView.post(
                  () ->
                      videoInput.start(
                          this,
                          resultIntent.getData(),
                          hands.getGlContext(),
                          glSurfaceView.getWidth(),
                          glSurfaceView.getHeight()));
            }
          }
        });
Intent pickVideoIntent = new Intent(Intent.ACTION_PICK);
pickVideoIntent.setDataAndType(MediaStore.Video.Media.INTERNAL_CONTENT_URI, "video/*");
videoGetter.launch(pickVideoIntent);
```

----------------------------------------

TITLE: Defining GPU Sobel Edge Detection Graph for MediaPipe iOS
DESCRIPTION: This MediaPipe graph (`edge_detection_mobile_gpu.pbtxt`) defines a pipeline for real-time GPU Sobel edge detection on a live video stream. It takes `input_video`, converts it to luminance using `LuminanceCalculator`, and then applies the Sobel filter via `SobelEdgesCalculator`, outputting to `output_video`. This graph is used in both Android and iOS 'helloworld' examples.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_0

LANGUAGE: MediaPipe Graph Definition
CODE:
```
# MediaPipe graph that performs GPU Sobel edge detection on a live video stream.
# Used in the examples
# mediapipe/examples/android/src/java/com/google/mediapipe/apps/basic:helloworld
# and mediapipe/examples/ios/helloworld.

# Images coming into and out of the graph.
input_stream: "input_video"
output_stream: "output_video"

# Converts RGB images into luminance images, still stored in RGB format.
node: {
  calculator: "LuminanceCalculator"
  input_stream: "input_video"
  output_stream: "luma_video"
}

# Applies the Sobel filter to luminance images stored in RGB format.
node: {
  calculator: "SobelEdgesCalculator"
  input_stream: "luma_video"
  output_stream: "output_video"
}
```

----------------------------------------

TITLE: Installing Bazelisk via Homebrew (Bash)
DESCRIPTION: This command uses Homebrew, a package manager for macOS, to install Bazelisk. Bazelisk is a wrapper for Bazel that automatically downloads the correct Bazel version for your project, ensuring compatibility.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/ios.md#_snippet_1

LANGUAGE: bash
CODE:
```
brew install bazelisk
```

----------------------------------------

TITLE: Implementing MediaPipe Hands for Webcams in JavaScript
DESCRIPTION: This JavaScript code initializes the MediaPipe Hands solution, sets its configuration options like `maxNumHands` and `minDetectionConfidence`, and processes video frames from a webcam. It defines an `onResults` callback to draw detected hand landmarks and connections on a canvas, and uses `Camera` utility to continuously send frames to the MediaPipe model.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md#_snippet_3

LANGUAGE: JavaScript
CODE:
```
<script type="module">
const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');

function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);
  if (results.multiHandLandmarks) {
    for (const landmarks of results.multiHandLandmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,
                     {color: '#00FF00', lineWidth: 5});
      drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 2});
    }
  }
  canvasCtx.restore();
}

const hands = new Hands({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
}});
hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
hands.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({image: videoElement});
  },
  width: 1280,
  height: 720
});
camera.start();
</script>
```

----------------------------------------

TITLE: Sending Video Frames to MediaPipe Graph in Objective-C
DESCRIPTION: Modifies the `processVideoFrame:timestamp:fromSource:` method to send incoming `CVPixelBufferRef` frames from the camera source to the `mediapipeGraph`. Frames are sent into the `kInputStream` as `MPPPacketTypePixelBuffer` for processing.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_29

LANGUAGE: Objective-C
CODE:
```
- (void)processVideoFrame:(CVPixelBufferRef)imageBuffer
                timestamp:(CMTime)timestamp
               fromSource:(MPPInputSource*)source {
  if (source != _cameraSource) {
    NSLog(@"Unknown source: %@", source);
    return;
  }
  [self.mediapipeGraph sendPixelBuffer:imageBuffer
                            intoStream:kInputStream
                            packetType:MPPPacketTypePixelBuffer];
}
```

----------------------------------------

TITLE: Decoupled MediaPipe Graph Construction (Good Practice) - C++
DESCRIPTION: This C++ code illustrates a best practice for constructing MediaPipe graphs by explicitly defining streams for node outputs and connecting nodes via these streams. This approach decouples nodes, making the graph more maintainable, reusable, and easier to refactor, aligning with the design principles of proto representations.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_14

LANGUAGE: C++
CODE:
```
CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Inputs.
  Stream<A> a = graph.In(0).Cast<A>();

  // `node1` usage is limited to 3 lines below.
  auto& node1 = graph.AddNode("Calculator1");
  a.ConnectTo(node1.In("INPUT"));
  Stream<B> b = node1.Out("OUTPUT").Cast<B>();

  // `node2` usage is limited to 3 lines below.
  auto& node2 = graph.AddNode("Calculator2");
  b.ConnectTo(node2.In("INPUT"));
  Stream<C> c = node2.Out("OUTPUT").Cast<C>();

  // `node3` usage is limited to 4 lines below.
  auto& node3 = graph.AddNode("Calculator3");
  b.ConnectTo(node3.In("INPUT_B"));
  c.ConnectTo(node3.In("INPUT_C"));
  Stream<D> d = node3.Out("OUTPUT").Cast<D>();

  // `node4` usage is limited to 5 lines below.
  auto& node4 = graph.AddNode("Calculator4");
  b.ConnectTo(node4.In("INPUT_B"));
  c.ConnectTo(node4.In("INPUT_C"));
  d.ConnectTo(node4.In("INPUT_D"));
  Stream<E> e = node4.Out("OUTPUT").Cast<E>();

  // Outputs.
  b.SetName("b").ConnectTo(graph.Out(0));
  c.SetName("c").ConnectTo(graph.Out(1));
  d.SetName("d").ConnectTo(graph.Out(2));
  e.SetName("e").ConnectTo(graph.Out(3));

  return graph.GetConfig();
}
```

----------------------------------------

TITLE: Reusable MediaPipe Calculator Utility Functions - C++
DESCRIPTION: This C++ code demonstrates how to encapsulate MediaPipe calculator logic into reusable utility functions. Each function takes input streams and the graph object, adds a specific calculator node, connects inputs, and returns the output stream, promoting modularity and simplifying graph construction by abstracting node creation and connection details.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_16

LANGUAGE: C++
CODE:
```
Stream<B> RunCalculator1(Stream<A> a, Graph& graph) {
  auto& node = graph.AddNode("Calculator1");
  a.ConnectTo(node.In("INPUT"));
  return node.Out("OUTPUT").Cast<B>();
}

Stream<C> RunCalculator2(Stream<B> b, Graph& graph) {
  auto& node = graph.AddNode("Calculator2");
  b.ConnectTo(node.In("INPUT"));
  return node.Out("OUTPUT").Cast<C>();
}

Stream<D> RunCalculator3(Stream<B> b, Stream<C> c, Graph& graph) {
  auto& node = graph.AddNode("Calculator3");
  b.ConnectTo(node.In("INPUT_B"));
  c.ConnectTo(node.In("INPUT_C"));
  return node.Out("OUTPUT").Cast<D>();
}

Stream<E> RunCalculator4(Stream<B> b, Stream<C> c, Stream<D> d, Graph& graph) {
  auto& node = graph.AddNode("Calculator4");
  b.ConnectTo(node.In("INPUT_B"));
  c.ConnectTo(node.In("INPUT_C"));
  d.ConnectTo(node.In("INPUT_D"));
  return node.Out("OUTPUT").Cast<E>();
}

CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Inputs.
  Stream<A> a = graph.In(0).Cast<A>();

  Stream<B> b = RunCalculator1(a, graph);
```

----------------------------------------

TITLE: Detecting 3D Objects in Static Images with MediaPipe Objectron (Python)
DESCRIPTION: This snippet demonstrates how to use MediaPipe Objectron to detect 3D bounding boxes on a collection of static image files. It initializes the `Objectron` model in `static_image_mode`, processes each image, converts color formats, and then draws the detected 2D landmarks and 3D axes onto the image before saving.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_0

LANGUAGE: Python
CODE:
```
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# For static images:
IMAGE_FILES = []
with mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            model_name='Shoe') as objectron:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB and process it with MediaPipe Objectron.
    results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw box landmarks.
    if not results.detected_objects:
      print(f'No box landmarks detected on {file}')
      continue
    print(f'Box landmarks of {file}:')
    annotated_image = image.copy()
    for detected_object in results.detected_objects:
      mp_drawing.draw_landmarks(
          annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
      mp_drawing.draw_axis(annotated_image, detected_object.rotation,
                           detected_object.translation)
      cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
```

----------------------------------------

TITLE: Cloning MediaPipe Repository - Bash
DESCRIPTION: This snippet demonstrates how to clone the MediaPipe GitHub repository into the user's home directory and then navigate into the newly created `mediapipe` directory. This is a prerequisite step for building and using MediaPipe.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_1

LANGUAGE: Bash
CODE:
```
$ cd $HOME
$ git clone --depth 1 https://github.com/google/mediapipe.git

# Change directory into MediaPipe root directory
$ cd mediapipe
```

----------------------------------------

TITLE: Detecting Language with MediaPipe Language Detector (JavaScript)
DESCRIPTION: This snippet demonstrates how to initialize and use the MediaPipe Language Detector to predict the language of an input text. It first resolves the necessary WASM files, then creates a LanguageDetector instance from a specified TFLite model path, and finally calls the detect method on the provided textData.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/text/README.md#_snippet_0

LANGUAGE: JavaScript
CODE:
```
const text = await FilesetResolver.forTextTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text/wasm"
);
const languageDetector = await LanguageDetector.createFromModelPath(text,
    "https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/1/language_detector.tflite"
);
const result = languageDetector.detect(textData);
```

----------------------------------------

TITLE: Cloning MediaPipe Repository (Bash)
DESCRIPTION: This snippet clones the MediaPipe GitHub repository and navigates into its root directory, which is the first step to set up MediaPipe for further operations.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/youtube8m/README.md#_snippet_0

LANGUAGE: Bash
CODE:
```
git clone https://github.com/google/mediapipe.git
cd mediapipe
```

----------------------------------------

TITLE: Cloning MediaPipe Repository and Changing Directory on Windows
DESCRIPTION: These commands are used to clone the MediaPipe GitHub repository into a local directory and then navigate into the newly created `mediapipe` directory. The `--depth 1` option performs a shallow clone, downloading only the latest commit to save time and disk space. This is the first step to obtaining the MediaPipe source code.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_30

LANGUAGE: Shell
CODE:
```
C:\Users\Username\mediapipe_repo> git clone --depth 1 https://github.com/google/mediapipe.git

# Change directory into MediaPipe root directory
C:\Users\Username\mediapipe_repo> cd mediapipe
```

----------------------------------------

TITLE: Adding MediaPipe and AndroidX Dependencies to Gradle - Gradle
DESCRIPTION: This Gradle `dependencies` block configures the required libraries for an Android project using MediaPipe. It includes local AARs/JARs from the `libs` directory, standard AndroidX components, MediaPipe-specific dependencies like Flogger, Guava, and Protobuf, and CameraX and AutoValue libraries, ensuring all necessary components are available for the build.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/android_archive_library.md#_snippet_6

LANGUAGE: gradle
CODE:
```
dependencies {
    implementation fileTree(dir: 'libs', include: ['*.jar', '*.aar'])
    implementation 'androidx.appcompat:appcompat:1.0.2'
    implementation 'androidx.constraintlayout:constraintlayout:1.1.3'
    testImplementation 'junit:junit:4.12'
    androidTestImplementation 'androidx.test.ext:junit:1.1.0'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.1.1'
    // MediaPipe deps
    implementation 'com.google.flogger:flogger:latest.release'
    implementation 'com.google.flogger:flogger-system-backend:latest.release'
    implementation 'com.google.code.findbugs:jsr305:latest.release'
    implementation 'com.google.guava:guava:27.0.1-android'
    implementation 'com.google.protobuf:protobuf-javalite:3.19.1'
    // CameraX core library
    def camerax_version = "1.0.0-beta10"
    implementation "androidx.camera:camera-core:$camerax_version"
    implementation "androidx.camera:camera-camera2:$camerax_version"
    implementation "androidx.camera:camera-lifecycle:$camerax_version"
    // AutoValue
    def auto_value_version = "1.8.1"
    implementation "com.google.auto.value:auto-value-annotations:$auto_value_version"
    annotationProcessor "com.google.auto.value:auto-value:$auto_value_version"
}
```

----------------------------------------

TITLE: Adding Packets to MediaPipe Input Stream (C++)
DESCRIPTION: This C++ snippet shows how to create and add 10 string packets, each containing "Hello World!" with increasing timestamps, to the "in" input stream of the MediaPipe graph. After adding all packets, the input stream is closed to signal the end of input for the graph.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_cpp.md#_snippet_4

LANGUAGE: c++
CODE:
```
for (int i = 0; i < 10; ++i) {
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream("in",
                     MakePacket<std::string>("Hello World!").At(Timestamp(i))));
}
MP_RETURN_IF_ERROR(graph.CloseInputStream("in"));
```

----------------------------------------

TITLE: Enabling Graph Runtime Monitoring in MediaPipe
DESCRIPTION: This MediaPipe graph configuration snippet demonstrates how to enable the background capturing of graph runtime information by setting `enable_graph_runtime_info` to `true` within the `runtime_info` block. This flag directs the runtime monitoring output to LOG(INFO) for debugging purposes, providing insights into graph execution.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/troubleshooting.md#_snippet_9

LANGUAGE: MediaPipe Graph Config
CODE:
```
graph {
  runtime_info {
    enable_graph_runtime_info: true
  }
  ...
}
```

----------------------------------------

TITLE: Defining MediaPipe Inference Utility Functions in C++
DESCRIPTION: This snippet defines `RunInference` and `BuildGraph` functions. `RunInference` encapsulates the logic for adding an `InferenceCalculator` node to a MediaPipe graph, connecting input tensors and a model, and configuring the delegate. `BuildGraph` demonstrates how to use `RunInference` to construct a complete MediaPipe graph, setting up graph inputs, outputs, and configuring a GPU delegate for inference.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_2

LANGUAGE: C++
CODE:
```
// Updates graph to run inference.
Stream<std::vector<Tensor>> RunInference(
    Stream<std::vector<Tensor>> tensors, SidePacket<TfLiteModelPtr> model,
    const InferenceCalculatorOptions::Delegate& delegate, Graph& graph) {
  auto& inference_node = graph.AddNode("InferenceCalculator");
  auto& inference_opts =
      inference_node.GetOptions<InferenceCalculatorOptions>();
  *inference_opts.mutable_delegate() = delegate;
  tensors.ConnectTo(inference_node.In("TENSORS"));
  model.ConnectTo(inference_node.SideIn("MODEL"));
  return inference_node.Out("TENSORS").Cast<std::vector<Tensor>>();
}

CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Graph inputs.
  Stream<std::vector<Tensor>> input_tensors =
      graph.In(0).SetName("input_tensors").Cast<std::vector<Tensor>>();
  SidePacket<TfLiteModelPtr> model =
      graph.SideIn(0).SetName("model").Cast<TfLiteModelPtr>();

  InferenceCalculatorOptions::Delegate delegate;
  delegate.mutable_gpu();
  Stream<std::vector<Tensor>> output_tensors =
      RunInference(input_tensors, model, delegate, graph);

  // Graph outputs.
  output_tensors.SetName("output_tensors").ConnectTo(graph.Out(0));

  return graph.GetConfig();
}
```

----------------------------------------

TITLE: Processing Static Images with MediaPipe Face Mesh (Java)
DESCRIPTION: This snippet illustrates how to use MediaPipe Face Mesh for processing static images from the device gallery. It configures `FaceMeshOptions` for static image mode, connects the solution to a `FaceMeshResultImageView` for custom drawing, and uses an `ActivityResultLauncher` to select an image from the gallery, convert it to a Bitmap, and send it for processing. It also includes result and error listeners.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md#_snippet_5

LANGUAGE: Java
CODE:
```
// For reading images from gallery and drawing the output in an ImageView.
FaceMeshOptions faceMeshOptions =
    FaceMeshOptions.builder()
        .setStaticImageMode(true)
        .setRefineLandmarks(true)
        .setMaxNumFaces(1)
        .setRunOnGpu(true).build();
FaceMesh faceMesh = new FaceMesh(this, faceMeshOptions);

// Connects MediaPipe Face Mesh Solution to the user-defined ImageView instance
// that allows users to have the custom drawing of the output landmarks on it.
// See mediapipe/examples/android/solutions/facemesh/src/main/java/com/google/mediapipe/examples/facemesh/FaceMeshResultImageView.java
// as an example.
FaceMeshResultImageView imageView = new FaceMeshResultImageView(this);
faceMesh.setResultListener(
    faceMeshResult -> {
      int width = faceMeshResult.inputBitmap().getWidth();
      int height = faceMeshResult.inputBitmap().getHeight();
      NormalizedLandmark noseLandmark =
          result.multiFaceLandmarks().get(0).getLandmarkList().get(1);
      Log.i(
          TAG,
          String.format(
              "MediaPipe Face Mesh nose coordinates (pixel values): x=%f, y=%f",
              noseLandmark.getX() * width, noseLandmark.getY() * height));
      // Request canvas drawing.
      imageView.setFaceMeshResult(faceMeshResult);
      runOnUiThread(() -> imageView.update());
    });
faceMesh.setErrorListener(
    (message, e) -> Log.e(TAG, "MediaPipe Face Mesh error:" + message));

// ActivityResultLauncher to get an image from the gallery as Bitmap.
ActivityResultLauncher<Intent> imageGetter =
    registerForActivityResult(
        new ActivityResultContracts.StartActivityForResult(),
        result -> {
          Intent resultIntent = result.getData();
          if (resultIntent != null && result.getResultCode() == RESULT_OK) {
            Bitmap bitmap = null;
            try {
              bitmap =
                  MediaStore.Images.Media.getBitmap(
                      this.getContentResolver(), resultIntent.getData());
              // Please also rotate the Bitmap based on its orientation.
            } catch (IOException e) {
              Log.e(TAG, "Bitmap reading error:" + e);
            }
            if (bitmap != null) {
              faceMesh.send(bitmap);
            }
          }
        });
Intent pickImageIntent = new Intent(Intent.ACTION_PICK);
pickImageIntent.setDataAndType(MediaStore.Images.Media.INTERNAL_CONTENT_URI, "image/*");
imageGetter.launch(pickImageIntent);
```

----------------------------------------

TITLE: Requesting Camera Permissions with PermissionHelper (Java)
DESCRIPTION: This Java line, intended for the `onCreate` function of `MainActivity`, uses MediaPipe's `PermissionHelper` to check for and request camera permissions from the user. It triggers a dialog prompt for permission access.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_11

LANGUAGE: java
CODE:
```
PermissionHelper.checkAndRequestCameraPermissions(this);
```

----------------------------------------

TITLE: Building MediaPipe with Explicit Python Path (Bazel)
DESCRIPTION: This Bazel command builds a MediaPipe example, specifically 'hello_world', while explicitly specifying the path to the Python 3 binary using '--action_env PYTHON_BIN_PATH=$(which python3)'. It also disables GPU support with '--define MEDIAPIPE_DISABLE_GPU=1' and optimizes the build with '-c opt'. This addresses issues where Bazel fails to locate the Python binary.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/troubleshooting.md#_snippet_0

LANGUAGE: Bazel
CODE:
```
bazel build -c opt \
  --define MEDIAPIPE_DISABLE_GPU=1 \
  --action_env PYTHON_BIN_PATH=$(which python3) \
  mediapipe/examples/desktop/hello_world
```

----------------------------------------

TITLE: Building Android Application with Bazel
DESCRIPTION: This command compiles the Android application using Bazel, optimizing it for `android_arm64` architecture. It targets the `helloworld` rule within the specified `$APPLICATION_PATH`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_7

LANGUAGE: bash
CODE:
```
bazel build -c opt --config=android_arm64 $APPLICATION_PATH:helloworld
```

----------------------------------------

TITLE: Initializing and Embedding Audio with MediaPipe Audio Embedder (JavaScript)
DESCRIPTION: This snippet shows how to initialize the MediaPipe Audio Embedder and extract embeddings from audio data. It uses `FilesetResolver` to load the necessary WASM module and `AudioEmbedder` to create an instance from a specified model path. The `embed` method processes `audioData` to generate audio embeddings.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/audio/README.md#_snippet_1

LANGUAGE: JavaScript
CODE:
```
const audio = await FilesetResolver.forAudioTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio/wasm"
);
const audioEmbedder = await AudioEmbedder.createFromModelPath(audio,
    "https://storage.googleapis.com/mediapipe-assets/yamnet_embedding_metadata.tflite?generation=1668295071595506"
);
const embeddings = audioEmbedder.embed(audioData);
```

----------------------------------------

TITLE: Saving Generated MediaPipe AAR (Bash)
DESCRIPTION: This optional bash command copies the successfully built MediaPipe AAR from the Bazel output directory (`bazel-bin`) to a user-specified preferred location. This step allows developers to easily access and integrate the generated AAR into their Android projects.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/android_archive_library.md#_snippet_3

LANGUAGE: bash
CODE:
```
cp bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/aar_example/mediapipe_face_detection.aar
/absolute/path/to/your/preferred/location
```

----------------------------------------

TITLE: Detecting and Drawing Face Landmarks on Static Images using MediaPipe Face Mesh (Python)
DESCRIPTION: This Python example demonstrates how to use MediaPipe Face Mesh to detect and draw face landmarks on static images. It initializes the `FaceMesh` model with `static_image_mode=True` and processes a list of images, converting them to RGB, then drawing the face tessellation, contours, and irises on the annotated image. It requires OpenCV (`cv2`) and MediaPipe (`mediapipe`) as dependencies.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md#_snippet_0

LANGUAGE: Python
CODE:
```
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For static images:
IMAGE_FILES = []
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
      print('face_landmarks:', face_landmarks)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_iris_connections_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
```

----------------------------------------

TITLE: Running MediaPipe Hand Tracking on GPU (Bash)
DESCRIPTION: This command runs the GPU-enabled MediaPipe Hand Tracking application. It enables logging to stderr and loads the `hand_tracking_desktop_live_gpu.pbtxt` configuration file, which is optimized for processing live input using GPU acceleration.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/cpp.md#_snippet_3

LANGUAGE: Bash
CODE:
```
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_gpu \
  --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live_gpu.pbtxt
```

----------------------------------------

TITLE: Overriding onPause to Close ExternalTextureConverter (Java)
DESCRIPTION: This overridden `onPause` method ensures that the `converter` object is properly closed when the `MainActivity` enters a paused state. This is crucial for releasing OpenGL resources and preventing memory leaks.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_29

LANGUAGE: Java
CODE:
```
@Override
protected void onPause() {
  super.onPause();
  converter.close();
}
```

----------------------------------------

TITLE: HTML Setup for MediaPipe Face Mesh JavaScript
DESCRIPTION: This HTML snippet sets up the basic page structure for a MediaPipe Face Mesh application. It includes necessary script imports from the MediaPipe CDN for camera utilities, control utilities, drawing utilities, and the Face Mesh solution itself. It also defines a video element for input and a canvas element for output, which are essential prerequisites for the JavaScript logic.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md#_snippet_2

LANGUAGE: HTML
CODE:
```
<!DOCTYPE html>\n<html>\n<head>\n  <meta charset=\"utf-8\">\n  <script src=\"https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js\" crossorigin=\"anonymous\"></script>\n  <script src=\"https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js\" crossorigin=\"anonymous\"></script>\n  <script src=\"https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js\" crossorigin=\"anonymous\"></script>\n  <script src=\"https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js\" crossorigin=\"anonymous\"></script>\n</head>\n\n<body>\n  <div class=\"container\">\n    <video class=\"input_video\"></video>\n    <canvas class=\"output_canvas\" width=\"1280px\" height=\"720px\"></canvas>\n  </div>\n</body>\n</html>
```

----------------------------------------

TITLE: Requesting Camera Access and Starting Camera (Objective-C)
DESCRIPTION: This code requests camera access using `MPPCameraInputSource`. If permission is granted, it dispatches a block asynchronously to a video queue to start the camera source, ensuring camera operations are performed on the correct thread.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_17

LANGUAGE: Objective-C
CODE:
```
[_cameraSource requestCameraAccessWithCompletionHandler:^void(BOOL granted) {
  if (granted) {
    dispatch_async(_videoQueue, ^{
      [_cameraSource start];
    });
  }
}];
```

----------------------------------------

TITLE: Handling Optional MediaPipe Graph Inputs with std::optional (C++)
DESCRIPTION: This snippet shows how to use `std::optional` for MediaPipe graph input streams or side packets that are not always present. By conditionally initializing the stream, it allows for flexible graph configurations while maintaining clarity about potential inputs.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_10

LANGUAGE: C++
CODE:
```
std::optional<Stream<A>> a;
if (needs_a) {
  a = graph.In(0).SetName(a).Cast<A>();
}
```

----------------------------------------

TITLE: Generic Cross-Compilation with BAZEL_CPU in Docker
DESCRIPTION: This generic Bazel command, executed within the Docker environment, utilizes the `BAZEL_CPU` variable to automatically select the correct CPU architecture for cross-compilation. It simplifies the build process by abstracting the specific ARM variant, while still enabling Coral USB Edge TPU support.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/coral/README.md#_snippet_5

LANGUAGE: bash
CODE:
```
bazel build \
    --crosstool_top=@crosstool//:toolchains \
    --compiler=gcc \
    --cpu=${BAZEL_CPU} \
    --define darwinn_portable=1 \
    --define MEDIAPIPE_DISABLE_GPU=1 \
    --define MEDIAPIPE_EDGE_TPU=usb \
    --linkopt=-l:libusb-1.0.so \
    mediapipe/examples/coral:face_detection_tpu build
```

----------------------------------------

TITLE: Cloning MediaPipe Repository (macOS)
DESCRIPTION: This command sequence clones the MediaPipe GitHub repository with a depth of 1 (shallow clone) and then navigates into the newly created 'mediapipe' directory. This is the initial step to obtain the source code.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_20

LANGUAGE: bash
CODE:
```
$ git clone --depth 1 https://github.com/google/mediapipe.git

$ cd mediapipe
```

----------------------------------------

TITLE: Processing Camera Input with MediaPipe Hands (Java)
DESCRIPTION: This snippet demonstrates how to set up MediaPipe Hands for real-time camera input. It initializes HandsOptions for GPU processing, configures CameraInput to send frames to the Hands solution, and sets up a SolutionGlSurfaceView for OpenGL rendering of the results, including logging wrist landmark coordinates.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md#_snippet_4

LANGUAGE: java
CODE:
```
// For camera input and result rendering with OpenGL.
HandsOptions handsOptions =
    HandsOptions.builder()
        .setStaticImageMode(false)
        .setMaxNumHands(2)
        .setRunOnGpu(true).build();
Hands hands = new Hands(this, handsOptions);
hands.setErrorListener(
    (message, e) -> Log.e(TAG, "MediaPipe Hands error:" + message));

// Initializes a new CameraInput instance and connects it to MediaPipe Hands Solution.
CameraInput cameraInput = new CameraInput(this);
cameraInput.setNewFrameListener(
    textureFrame -> hands.send(textureFrame));

// Initializes a new GlSurfaceView with a ResultGlRenderer<HandsResult> instance
// that provides the interfaces to run user-defined OpenGL rendering code.
// See mediapipe/examples/android/solutions/hands/src/main/java/com/google/mediapipe/examples/hands/HandsResultGlRenderer.java
// as an example.
SolutionGlSurfaceView<HandsResult> glSurfaceView =
    new SolutionGlSurfaceView<>(
        this, hands.getGlContext(), hands.getGlMajorVersion());
glSurfaceView.setSolutionResultRenderer(new HandsResultGlRenderer());
glSurfaceView.setRenderInputImage(true);

hands.setResultListener(
    handsResult -> {
      if (result.multiHandLandmarks().isEmpty()) {
        return;
      }
      NormalizedLandmark wristLandmark =
          handsResult.multiHandLandmarks().get(0).getLandmarkList().get(HandLandmark.WRIST);
      Log.i(
          TAG,
          String.format(
              "MediaPipe Hand wrist normalized coordinates (value range: [0, 1]): x=%f, y=%f",
              wristLandmark.getX(), wristLandmark.getY()));
      // Request GL rendering.
      glSurfaceView.setRenderData(handsResult);
      glSurfaceView.requestRender();
    });

// The runnable to start camera after the GLSurfaceView is attached.
glSurfaceView.post(
    () ->
        cameraInput.start(
            this,
            hands.getGlContext(),
            CameraInput.CameraFacing.FRONT,
            glSurfaceView.getWidth(),
            glSurfaceView.getHeight()));
```

----------------------------------------

TITLE: Running the MediaPipe Command Line Profiler (Basic) - Bazel/Shell
DESCRIPTION: This command executes the `print_profile` tool, which extracts information from MediaPipe trace files. It's the basic way to run the profiler without any specific options, displaying all default columns.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/profiler/reporter/README.md#_snippet_0

LANGUAGE: Bazel
CODE:
```
bazel run :print_profile
```

----------------------------------------

TITLE: Defining GPU Sobel Edge Detection Graph for MediaPipe
DESCRIPTION: This MediaPipe graph (.pbtxt) defines a pipeline for real-time GPU Sobel edge detection on a live video stream. It takes "input_video" as input, converts it to luminance using "LuminanceCalculator", and then applies the Sobel filter via "SobelEdgesCalculator", outputting the result to "output_video". This graph is designed for use in Android and iOS MediaPipe applications.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_0

LANGUAGE: MediaPipe Graph
CODE:
```
# MediaPipe graph that performs GPU Sobel edge detection on a live video stream.
# Used in the examples in
# mediapipe/examples/android/src/java/com/mediapipe/apps/basic and
# mediapipe/examples/ios/edgedetectiongpu.

# Images coming into and out of the graph.
input_stream: "input_video"
output_stream: "output_video"

# Converts RGB images into luminance images, still stored in RGB format.
node: {
  calculator: "LuminanceCalculator"
  input_stream: "input_video"
  output_stream: "luma_video"
}

# Applies the Sobel filter to luminance images stored in RGB format.
node: {
  calculator: "SobelEdgesCalculator"
  input_stream: "luma_video"
  output_stream: "output_video"
}
```

----------------------------------------

TITLE: Detailed MediaPipe Calculator State Monitoring
DESCRIPTION: This detailed log output provides a comprehensive overview of a specific MediaPipe calculator's state, including its idle time, timestamp bound, and statistics for both input and output streams. It shows queue sizes, total packets added, and timestamp bounds for each stream, aiding in deep debugging of data flow and performance.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/troubleshooting.md#_snippet_8

LANGUAGE: Log Output
CODE:
```
PreviousLoopbackCalculator: (idle for 8.17s, ts bound : 0)
Input streams:
 * LOOP:0:segmentation_finished - queue size: 0, total added: 0, ts bound: 569604400011
 * MAIN:0:input_frames_gpu - queue size: 0, total added: 2, ts bound: 569604400011
Output streams:
 * PREV_LOOP:0:prev_segmentation_finished, total added: 0, ts bound: 569604400011
```

----------------------------------------

TITLE: Processing Static Images with MediaPipe Pose (Python)
DESCRIPTION: This snippet demonstrates how to use MediaPipe Pose to process a list of static images. It initializes the `Pose` model in `static_image_mode`, enables segmentation, and then iterates through images to detect pose landmarks and generate segmentation masks. It also shows how to visualize the results by drawing landmarks and applying the segmentation mask to the image.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md#_snippet_0

LANGUAGE: Python
CODE:
```
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
```

----------------------------------------

TITLE: Setting Up HTML for MediaPipe Selfie Segmentation
DESCRIPTION: This HTML snippet defines the basic page structure required for a MediaPipe Selfie Segmentation web application. It includes meta tags, imports necessary MediaPipe utility scripts (camera_utils, control_utils, drawing_utils, selfie_segmentation), and sets up `video` and `canvas` elements for input and output respectively.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/selfie_segmentation.md#_snippet_3

LANGUAGE: HTML
CODE:
```
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/selfie_segmentation.js" crossorigin="anonymous"></script>
</head>

<body>
  <div class="container">
    <video class="input_video"></video>
    <canvas class="output_canvas" width="1280px" height="720px"></canvas>
  </div>
</body>
</html>
```

----------------------------------------

TITLE: Good Practice: Defining MediaPipe Graph Outputs at End (C++)
DESCRIPTION: This snippet illustrates the recommended practice of defining all graph outputs explicitly at the very end of the `BuildGraph` function. This improves readability, centralizes output declarations, and enhances the reusability of helper functions by returning streams that can then be connected to graph outputs.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_12

LANGUAGE: C++
CODE:
```
Stream<F> RunSomething(Stream<Input> input, Graph& graph) {
  // ...
  return node.Out("OUTPUT_F").Cast<F>();
}

CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // 10/100/N lines of code.
  Stream<D> d = node.Out("OUTPUT_D").Cast<D>();
  // 10/100/N lines of code.
  Stream<E> e = node.Out("OUTPUT_E").Cast<E>();
  // 10/100/N lines of code.
  Stream<F> f = RunSomething(input, graph);
  // ...

  // Outputs.
  d.SetName("output_d").ConnectTo(graph.Out(0));
  e.SetName("output_e").ConnectTo(graph.Out(1));
  f.SetName("output_f").ConnectTo(graph.Out(2));

  return graph.GetConfig();
}
```

----------------------------------------

TITLE: Configuring MediaPipe Graph for Tracing and Profiling
DESCRIPTION: This configuration snippet, placed within the `CalculatorGraphConfig` of a MediaPipe graph, enables the framework's built-in tracing and profiling features. It sets `trace_enabled` for packet-level information, `enable_profiler` to activate logging, `trace_log_interval_count` to manage log file rotation, and `trace_log_path` to specify the output directory for binary protobuf trace logs.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/tools/tracing_and_profiling.md#_snippet_0

LANGUAGE: MediaPipe Config
CODE:
```
profiler_config {
  trace_enabled: true
  enable_profiler: true
  trace_log_interval_count: 200
  trace_log_path: "/sdcard/Download/"
}
```

----------------------------------------

TITLE: Monitoring Packet Flow and Waiting Calculators in MediaPipe
DESCRIPTION: This log output from MediaPipe's runtime monitoring provides insights into the graph's data flow. It shows the total number of packets currently buffered in input queues and identifies specific calculators that are waiting on packets from certain input streams before they can proceed with processing, aiding in bottleneck identification.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/troubleshooting.md#_snippet_7

LANGUAGE: Log Output
CODE:
```
Running calculators: PacketClonerCalculator
Num packets in input queues: 4
GateCalculator_2 waiting on stream(s): :1:norm_start_rect
MergeCalculator waiting on stream(s): :0:output_frames_gpu_ao, :1:segmentation_preview_gpu
```

----------------------------------------

TITLE: Retrieving and Printing Packets from MediaPipe Output Stream (C++)
DESCRIPTION: This C++ snippet demonstrates how to retrieve processed packets from the MediaPipe graph's output stream using an `OutputStreamPoller`. It iterates through the available packets, extracts the string content from each, and prints it to the log, completing the data flow.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_cpp.md#_snippet_5

LANGUAGE: c++
CODE:
```
mediapipe::Packet packet;
while (poller.Next(&packet)) {
  ABSL_LOG(INFO) << packet.Get<string>();
}
```

----------------------------------------

TITLE: Logging Tensor Contents for Debugging (C++)
DESCRIPTION: This snippet demonstrates how to print the contents of a MediaPipe `Tensor` to the command line terminal using the `debug::LogTensor` function. It is useful for visualizing image data or other tensor contents without a graphical interface, even over SSH.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/troubleshooting.md#_snippet_10

LANGUAGE: C++
CODE:
```
debug::LogTensor(tensor)
```

----------------------------------------

TITLE: Bad Practice: Defining MediaPipe Graph Outputs Mid-Graph (C++)
DESCRIPTION: This snippet demonstrates an anti-pattern where graph outputs are defined dynamically within the graph builder or helper functions. This approach makes it challenging to determine the total number of outputs, is prone to indexing errors, and restricts the reusability of helper functions like `RunSomething`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_11

LANGUAGE: C++
CODE:
```
void RunSomething(Stream<Input> input, Graph& graph) {
  // ...
  node.Out("OUTPUT_F")
      .SetName("output_f").ConnectTo(graph.Out(2));  // Bad.
}

CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // 10/100/N lines of code.
  node.Out("OUTPUT_D")
      .SetName("output_d").ConnectTo(graph.Out(0));  // Bad.
  // 10/100/N lines of code.
  node.Out("OUTPUT_E")
      .SetName("output_e").ConnectTo(graph.Out(1));  // Bad.
  // 10/100/N lines of code.
  RunSomething(input, graph);
  // ...

  return graph.GetConfig();
}
```

----------------------------------------

TITLE: Building MediaPipe with OpenGL ES 3.1+ Support
DESCRIPTION: This Bazel command builds MediaPipe targets when the Linux desktop GPU supports OpenGL ES 3.1 or greater. The `--copt` flags address potential issues with X11 headers, enabling TFLite inference on GPU.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/gpu_support.md#_snippet_3

LANGUAGE: bash
CODE:
```
$ bazel build --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 <my-target>
```

----------------------------------------

TITLE: Running AutoFlip for Video Cropping (Linux/macOS)
DESCRIPTION: This command executes the compiled AutoFlip binary, specifying the MediaPipe graph configuration file and input side packets. It sets the input video path, output video path, and desired output aspect ratio (e.g., 1:1) for the cropping operation.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/autoflip.md#_snippet_1

LANGUAGE: bash
CODE:
```
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/autoflip/run_autoflip \
  --calculator_graph_config_file=mediapipe/examples/desktop/autoflip/autoflip_graph.pbtxt \
  --input_side_packets=input_video_path=/absolute/path/to/the/local/video/file,output_video_path=/absolute/path/to/save/the/output/video/file,aspect_ratio=1:1
```

----------------------------------------

TITLE: Initializing and Starting MediaPipe Graph (C++)
DESCRIPTION: This C++ snippet demonstrates the initialization and execution of a MediaPipe `CalculatorGraph`. It initializes the graph with a given configuration, adds an `OutputStreamPoller` to retrieve data from the "out" stream, and then starts the graph run. This sets up the graph for processing packets.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_cpp.md#_snippet_3

LANGUAGE: c++
CODE:
```
CalculatorGraph graph;
MP_RETURN_IF_ERROR(graph.Initialize(config));
MP_ASSIGN_OR_RETURN(OutputStreamPoller poller,
                    graph.AddOutputStreamPoller("out"));
MP_RETURN_IF_ERROR(graph.StartRun({}));
```

----------------------------------------

TITLE: Running MediaPipe Hello World Example (CPU)
DESCRIPTION: This Bazel command executes the MediaPipe 'Hello World' C++ example specifically configured for CPU-only operation on a Linux desktop. The `--define MEDIAPIPE_DISABLE_GPU=1` flag ensures that any GPU acceleration features are explicitly disabled during the build and execution process.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_14

LANGUAGE: bash
CODE:
```
$ bazel run --define MEDIAPIPE_DISABLE_GPU=1 \
        mediapipe/examples/desktop/hello_world:hello_world
```

----------------------------------------

TITLE: Implementing MediaPipe Selfie Segmentation in JavaScript
DESCRIPTION: This JavaScript module initializes the MediaPipe Selfie Segmentation model, configures its options (e.g., `modelSelection`), and defines a callback function (`onResults`) to process the segmentation output. It also sets up a `Camera` utility to continuously send video frames to the model for real-time processing and rendering of the segmentation mask on a canvas.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/selfie_segmentation.md#_snippet_4

LANGUAGE: JavaScript
CODE:
```
<script type="module">
const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');

function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.segmentationMask, 0, 0,
                      canvasElement.width, canvasElement.height);

  // Only overwrite existing pixels.
  canvasCtx.globalCompositeOperation = 'source-in';
  canvasCtx.fillStyle = '#00FF00';
  canvasCtx.fillRect(0, 0, canvasElement.width, canvasElement.height);

  // Only overwrite missing pixels.
  canvasCtx.globalCompositeOperation = 'destination-atop';
  canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);

  canvasCtx.restore();
}

const selfieSegmentation = new SelfieSegmentation({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`;
}});
selfieSegmentation.setOptions({
  modelSelection: 1,
});
selfieSegmentation.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await selfieSegmentation.send({image: videoElement});
  },
  width: 1280,
  height: 720
});
camera.start();
</script>
```

----------------------------------------

TITLE: Handling Camera Permission Results and Resuming (Java)
DESCRIPTION: This Java code block in `MainActivity` overrides `onRequestPermissionsResult` to process the user's permission response via `PermissionHelper`. The `onResume` method checks if camera permissions are granted and, if so, calls `startCamera()` to initialize camera functionality.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_12

LANGUAGE: java
CODE:
```
@Override
public void onRequestPermissionsResult(
    int requestCode, String[] permissions, int[] grantResults) {
  super.onRequestPermissionsResult(requestCode, permissions, grantResults);
  PermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults);
}

@Override
protected void onResume() {
  super.onResume();
  if (PermissionHelper.cameraPermissionsGranted(this)) {
    startCamera();
  }
}

public void startCamera() {}
```

----------------------------------------

TITLE: Decoupled MediaPipe Graph Definition - Proto
DESCRIPTION: This Protocol Buffer (Proto) definition showcases a naturally decoupled MediaPipe graph structure. Each node explicitly declares its input and output streams by name, rather than directly referencing other nodes, which inherently promotes modularity and simplifies graph modifications and understanding.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_15

LANGUAGE: Proto
CODE:
```
input_stream: "a"

node {
  calculator: "Calculator1"
  input_stream: "INPUT:a"
  output_stream: "OUTPUT:b"
}

node {
  calculator: "Calculator2"
  input_stream: "INPUT:b"
  output_stream: "OUTPUT:C"
}

node {
  calculator: "Calculator3"
  input_stream: "INPUT_B:b"
  input_stream: "INPUT_C:c"
  output_stream: "OUTPUT:d"
}

node {
  calculator: "Calculator4"
  input_stream: "INPUT_B:b"
  input_stream: "INPUT_C:c"
  input_stream: "INPUT_D:d"
  output_stream: "OUTPUT:e"
}

output_stream: "b"
output_stream: "c"
output_stream: "d"
output_stream: "e"
```

----------------------------------------

TITLE: Retrieving an Integer from a MediaPipe Packet (int_t) in Python
DESCRIPTION: Retrieves the standard integer payload (mapped to C++ `int_t`) from a MediaPipe packet using `mp.packet_getter.get_int`. This getter is versatile for various integer types.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_3

LANGUAGE: Python
CODE:
```
get_int(packet)
```

----------------------------------------

TITLE: Configuring Android Layout for Camera Preview (XML)
DESCRIPTION: This XML snippet replaces the default TextView in `activity_main.xml` with a FrameLayout containing a TextView. The FrameLayout (`preview_display_layout`) will house the camera preview, while the nested TextView (`no_camera_access_view`) serves as a placeholder message when camera permissions are not granted.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_13

LANGUAGE: XML
CODE:
```
<FrameLayout
    android:id="@+id/preview_display_layout"
    android:layout_width="fill_parent"
    android:layout_height="fill_parent"
    android:layout_weight="1">
    <TextView
        android:id="@+id/no_camera_access_view"
        android:layout_height="fill_parent"
        android:layout_width="fill_parent"
        android:gravity="center"
        android:text="@string/no_camera_access" />
</FrameLayout>
```

----------------------------------------

TITLE: Logging cv::Mat and ImageFrame Contents (C++)
DESCRIPTION: This snippet shows how to visualize the contents of `cv::Mat` and `ImageFrame` objects in the terminal using `debug::LogMat` and `debug::LogImage` functions, respectively. These tools provide text-based visualization, supporting truecolor or ASCII output based on terminal capabilities.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/troubleshooting.md#_snippet_11

LANGUAGE: C++
CODE:
```
debug::LogMat(mat);
debug::LogImage(image_frame);
```

----------------------------------------

TITLE: Declaring Optional Input Ports (New Node API, C++)
DESCRIPTION: This code demonstrates the new, more type-safe and declarative way to define an optional input port using `Input<T>::Optional`. It replaces the verbose procedural checks with a single static constexpr declaration.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/api2/README.md#_snippet_1

LANGUAGE: C++
CODE:
```
static constexpr Input<int>::Optional kSelect{"SELECT"};
```

----------------------------------------

TITLE: HTML Setup for MediaPipe Face Detection Web Application
DESCRIPTION: This HTML snippet provides the basic structure for a web application using MediaPipe Face Detection. It includes necessary script imports for MediaPipe utilities (camera, control, drawing, face detection) from CDN, and defines `video` and `canvas` elements for input and output display. These elements are crucial for real-time video processing and rendering.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_detection.md#_snippet_3

LANGUAGE: HTML
CODE:
```
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/face_detection.js" crossorigin="anonymous"></script>
</head>

<body>
  <div class="container">
    <video class="input_video"></video>
    <canvas class="output_canvas" width="1280px" height="720px"></canvas>
  </div>
</body>
</html>
```

----------------------------------------

TITLE: Setting Next Timestamp Bound in MediaPipe Calculator (C++)
DESCRIPTION: This C++ snippet demonstrates how a MediaPipe calculator can signal that no packet will be produced for the current timestamp on a specific output stream. It sets the next timestamp bound to the successive timestamp, allowing downstream calculators to proceed without waiting indefinitely for a packet that won't arrive. This is crucial for maintaining real-time processing flow.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/realtime_streams.md#_snippet_1

LANGUAGE: C++
CODE:
```
cc->Outputs().Tag("output_frame").SetNextTimestampBound(
  cc->InputTimestamp().NextAllowedInStream());
```

----------------------------------------

TITLE: Connecting FrameProcessor to Converter in Android
DESCRIPTION: Connects the `FrameProcessor` as a consumer to an `ExternalTextureConverter` instance. This step ensures that the `processor` receives the converted camera frames from the `converter` for further processing by the MediaPipe graph. This typically occurs in the `onResume()` method after the converter has been initialized.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_39

LANGUAGE: Java
CODE:
```
converter.setConsumer(processor);
```

----------------------------------------

TITLE: Reusing MediaPipe Inference Function in C++
DESCRIPTION: This example demonstrates the reusability of the `RunInference` utility function. It shows how to chain multiple inference operations by using the output of a previous inference as the input for a subsequent one, highlighting the benefit of abstracting common graph construction patterns.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_3

LANGUAGE: C++
CODE:
```
  // Run first inference.
  Stream<std::vector<Tensor>> output_tensors =
      RunInference(input_tensors, model, delegate, graph);
  // Run second inference on the output of the first one.
  Stream<std::vector<Tensor>> extra_output_tensors =
      RunInference(output_tensors, extra_model, delegate, graph);
```

----------------------------------------

TITLE: Applying Selfie Segmentation to Static Images (Python)
DESCRIPTION: This snippet demonstrates how to perform selfie segmentation on a list of static image files. It initializes the `SelfieSegmentation` model with `model_selection=0` for general segmentation, processes each image, and saves the segmented output to a temporary file. It requires `IMAGE_FILES` to be populated with paths to input images.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/selfie_segmentation.md#_snippet_1

LANGUAGE: Python
CODE:
```
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white
with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=0) as selfie_segmentation:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    # Generate solid color images for showing the output selfie segmentation mask.
    fg_image = np.zeros(image.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    output_image = np.where(condition, fg_image, bg_image)
    cv2.imwrite('/tmp/selfie_segmentation_output' + str(idx) + '.png', output_image)
```

----------------------------------------

TITLE: Creating a Single-Precision Float MediaPipe Packet in Python
DESCRIPTION: Creates a MediaPipe packet with a single-precision float payload (mapped to C++ `float`) using `mp.packet_creator.create_float`. This is suitable for standard floating-point numbers.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_20

LANGUAGE: Python
CODE:
```
create_float(1.1)
```

----------------------------------------

TITLE: Creating an Integer MediaPipe Packet (int_t) in Python
DESCRIPTION: Creates a MediaPipe packet with a standard integer payload (mapped to C++ `int_t`) using `mp.packet_creator.create_int`. This method is suitable for general integer values.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_2

LANGUAGE: Python
CODE:
```
create_int(1)
```

----------------------------------------

TITLE: Populating Calculator Options from Graph Options in MediaPipe
DESCRIPTION: This example illustrates how a graph can accept graph_options (e.g., FaceDetectionOptions) and use them to dynamically populate fields in calculator options (e.g., ImageToTensorCalculatorOptions) and subgraph options (e.g., InferenceCalculatorOptions) using the option_value syntax. This allows for centralized configuration.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/graphs.md#_snippet_6

LANGUAGE: protobuf
CODE:
```
graph_options: {
  [type.googleapis.com/mediapipe.FaceDetectionOptions] {}
}

node: {
  calculator: "ImageToTensorCalculator"
  input_stream: "IMAGE:image"
  node_options: {
    [type.googleapis.com/mediapipe.ImageToTensorCalculatorOptions] {
        keep_aspect_ratio: true
        border_mode: BORDER_ZERO
    }
  }
  option_value: "output_tensor_width:options/tensor_width"
  option_value: "output_tensor_height:options/tensor_height"
}

node {
  calculator: "InferenceCalculator"
  node_options: {
    [type.googleapis.com/mediapipe.InferenceCalculatorOptions] {}
  }
  option_value: "delegate:options/delegate"
  option_value: "model_path:options/model_path"
}
```

----------------------------------------

TITLE: Configuring Camera Facing and Starting Camera (Java)
DESCRIPTION: This code determines the camera facing direction (front or back) based on application metadata and then initializes the camera using `cameraHelper.startCamera()`. It sets up the camera for preview without directly displaying the output to a `SurfaceTexture`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_25

LANGUAGE: Java
CODE:
```
CameraHelper.CameraFacing cameraFacing =
    applicationInfo.metaData.getBoolean("cameraFacingFront", false)
        ? CameraHelper.CameraFacing.FRONT
        : CameraHelper.CameraFacing.BACK;
cameraHelper.startCamera(this, cameraFacing, /*unusedSurfaceTexture=*/ null);
```

----------------------------------------

TITLE: Configuring Custom OpenCV and FFmpeg Paths in Bazel (Bazel BUILD)
DESCRIPTION: This configuration snippet for Bazel BUILD files (`WORKSPACE`, `opencv_linux.BUILD`, `ffmpeg_linux.BUILD`) updates the `new_local_repository` rules for 'linux_opencv' and 'linux_ffmpeg' to point to custom installation paths (e.g., '/usr/local'). It also defines `cc_library` rules for 'opencv' and 'libffmpeg', specifying their source files, headers, include paths, and linking options for a custom build.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_18

LANGUAGE: bazel
CODE:
```
new_local_repository(
    name = "linux_opencv",
    build_file = "@//third_party:opencv_linux.BUILD",
    path = "/usr/local",
)

new_local_repository(
    name = "linux_ffmpeg",
    build_file = "@//third_party:ffmpeg_linux.BUILD",
    path = "/usr/local",
)

cc_library(
    name = "opencv",
    srcs = glob(
        [
            "lib/libopencv_core.so",
            "lib/libopencv_highgui.so",
            "lib/libopencv_imgcodecs.so",
            "lib/libopencv_imgproc.so",
            "lib/libopencv_video.so",
            "lib/libopencv_videoio.so",
        ],
    ),
    hdrs = glob([
        # For OpenCV 3.x
        "include/opencv2/**/*.h*",
        # For OpenCV 4.x
        # "include/opencv4/opencv2/**/*.h*",
    ]),
    includes = [
        # For OpenCV 3.x
        "include/",
        # For OpenCV 4.x
        # "include/opencv4/",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libffmpeg",
    srcs = glob(
        [
            "lib/libav*.so",
        ],
    ),
    hdrs = glob(["include/libav*/*.h"]),
    includes = ["include"],
    linkopts = [
        "-lavcodec",
        "-lavformat",
        "-lavutil",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
```

----------------------------------------

TITLE: Installing MediaPipe Build Dependencies (Debian/Ubuntu)
DESCRIPTION: These commands install essential development packages required for building MediaPipe from source on Debian or Ubuntu systems. Dependencies include Python development headers, virtual environment tools, the Protobuf compiler, and CMake for building OpenCV from source.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python.md#_snippet_3

LANGUAGE: Bash
CODE:
```
$ sudo apt install python3-dev
$ sudo apt install python3-venv
$ sudo apt install -y protobuf-compiler

# If you need to build opencv from source.
$ sudo apt install cmake
```

----------------------------------------

TITLE: Configuring MediaPipe Calculator Options (Proto3 Syntax)
DESCRIPTION: This MediaPipe graph configuration snippet shows how to pass processing parameters to a calculator using the `node_options` field with Proto3 syntax. It specifies a `TfLiteInferenceCalculator` and sets its `model_path` option.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/calculators.md#_snippet_5

LANGUAGE: MediaPipe Graph Configuration
CODE:
```
  node {
    calculator: "TfLiteInferenceCalculator"
    input_stream: "TENSORS:main_model_input"
    output_stream: "TENSORS:main_model_output"
    node_options: {
      [type.googleapis.com/mediapipe.TfLiteInferenceCalculatorOptions] {
        model_path: "mediapipe/models/detection_model.tflite"
      }
    }
  }
```

----------------------------------------

TITLE: Using TwoPassThroughSubgraph in Main MediaPipe Graph
DESCRIPTION: This snippet demonstrates how to integrate the previously defined and registered `TwoPassThroughSubgraph` into a larger MediaPipe graph. The subgraph is instantiated as a node, connecting its input and output streams to other calculators within the main graph, showcasing its reusability as a modular component.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/graphs.md#_snippet_4

LANGUAGE: MediaPipe Graph Config
CODE:
```
# This main graph is defined in main_pass_throughcals.pbtxt
# using subgraph called "TwoPassThroughSubgraph"

input_stream: "in"
node {
    calculator: "PassThroughCalculator"
    input_stream: "in"
    output_stream: "out1"
}
node {
    calculator: "TwoPassThroughSubgraph"
    input_stream: "out1"
    output_stream: "out3"
}
node {
    calculator: "PassThroughCalculator"
    input_stream: "out3"
    output_stream: "out4"
}
```

----------------------------------------

TITLE: Defining Bazel Build Rules for iOS Application
DESCRIPTION: This Bazel BUILD file defines two rules: ios_application and objc_library. The objc_library rule compiles Objective-C source files and headers, linking against the UIKit SDK, while the ios_application rule packages this library into a runnable iOS app, specifying bundle ID, supported device families, minimum OS version, and a provisioning profile.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_1

LANGUAGE: Bazel
CODE:
```
MIN_IOS_VERSION = "12.0"

load(
    "@build_bazel_rules_apple//apple:ios.bzl",
    "ios_application",
)

ios_application(
    name = "HelloWorldApp",
    bundle_id = "com.google.mediapipe.HelloWorld",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = ["Info.plist"],
    minimum_os_version = MIN_IOS_VERSION,
    provisioning_profile = "//mediapipe/examples/ios:developer_provisioning_profile",
    deps = [":HelloWorldAppLibrary"],
)

objc_library(
    name = "HelloWorldAppLibrary",
    srcs = [
        "AppDelegate.m",
        "ViewController.m",
        "main.m",
    ],
    hdrs = [
        "AppDelegate.h",
        "ViewController.h",
    ],
    data = [
        "Base.lproj/LaunchScreen.storyboard",
        "Base.lproj/Main.storyboard",
    ],
    sdk_frameworks = [
        "UIKit",
    ],
    deps = [],
)
```

----------------------------------------

TITLE: Stylizing Faces with MediaPipe Face Stylizer (JavaScript)
DESCRIPTION: This snippet initializes the MediaPipe Face Stylizer task by loading the necessary WASM files and a pre-trained model. It then applies a stylization effect to an HTML image element, transforming the appearance of faces within the image. Requires the @mediapipe/tasks-vision library.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md#_snippet_2

LANGUAGE: JavaScript
CODE:
```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const faceStylizer = await FaceStylizer.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/face_stylizer/blaze_face_stylizer/float32/1/blaze_face_stylizer.task"
);
const image = document.getElementById("image") as HTMLImageElement;
const stylizedImage = faceStylizer.stylize(image);
```

----------------------------------------

TITLE: Configuring maxFramesInFlight for MediaPipe Graph in Objective-C
DESCRIPTION: Sets the `maxFramesInFlight` property of the `mediapipeGraph` to `2`. This limits the number of frames processed concurrently, preventing memory contention and ensuring real-time performance for live video feeds.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_27

LANGUAGE: Objective-C
CODE:
```
// Set maxFramesInFlight to a small value to avoid memory contention for real-time processing.
self.mediapipeGraph.maxFramesInFlight = 2;
```

----------------------------------------

TITLE: Setting Up MediaPipe Pose Tracking HTML Structure
DESCRIPTION: This HTML snippet defines the basic page structure for a MediaPipe Pose tracking application. It includes necessary script imports for MediaPipe utilities (camera, control, drawing, pose) from CDN and sets up video and canvas elements for input and output, along with a container for landmark visualization.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md#_snippet_2

LANGUAGE: HTML
CODE:
```
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/control_utils_3d/control_utils_3d.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js" crossorigin="anonymous"></script>
</head>

<body>
  <div class="container">
    <video class="input_video"></video>
    <canvas class="output_canvas" width="1280px" height="720px"></canvas>
    <div class="landmark-grid-container"></div>
  </div>
</body>
</html>
```

----------------------------------------

TITLE: Declaring CameraXPreviewHelper in MainActivity (Java)
DESCRIPTION: This Java declaration adds a `CameraXPreviewHelper` member variable named `cameraHelper` to `MainActivity`. This utility class from MediaPipe simplifies the integration of CameraX for camera frame acquisition.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_18

LANGUAGE: Java
CODE:
```
private CameraXPreviewHelper cameraHelper;
```

----------------------------------------

TITLE: AutoFlip MediaPipe Graph Configuration
DESCRIPTION: This MediaPipe graph defines the complete AutoFlip pipeline, from video decoding and scaling to feature extraction (border, shot boundary, face detection). It outlines the interconnected calculators and their options, processing video frames to prepare for saliency-aware cropping.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/autoflip.md#_snippet_2

LANGUAGE: protobuf
CODE:
```
# Autoflip graph that only renders the final cropped video. For use with
# end user applications.
max_queue_size: -1

# VIDEO_PREP: Decodes an input video file into images and a video header.
node {
  calculator: "OpenCvVideoDecoderCalculator"
  input_side_packet: "INPUT_FILE_PATH:input_video_path"
  output_stream: "VIDEO:video_raw"
  output_stream: "VIDEO_PRESTREAM:video_header"
  output_side_packet: "SAVED_AUDIO_PATH:audio_path"
}

# VIDEO_PREP: Scale the input video before feature extraction.
node {
  calculator: "ScaleImageCalculator"
  input_stream: "FRAMES:video_raw"
  input_stream: "VIDEO_HEADER:video_header"
  output_stream: "FRAMES:video_frames_scaled"
  node_options: {
    [type.googleapis.com/mediapipe.ScaleImageCalculatorOptions]: {
      preserve_aspect_ratio: true
      output_format: SRGB
      target_width: 480
      algorithm: DEFAULT_WITHOUT_UPSCALE
    }
  }
}

# VIDEO_PREP: Create a low frame rate stream for feature extraction.
node {
  calculator: "PacketThinnerCalculator"
  input_stream: "video_frames_scaled"
  output_stream: "video_frames_scaled_downsampled"
  node_options: {
    [type.googleapis.com/mediapipe.PacketThinnerCalculatorOptions]: {
      thinner_type: ASYNC
      period: 200000
    }
  }
}

# DETECTION: find borders around the video and major background color.
node {
  calculator: "BorderDetectionCalculator"
  input_stream: "VIDEO:video_raw"
  output_stream: "DETECTED_BORDERS:borders"
}

# DETECTION: find shot/scene boundaries on the full frame rate stream.
node {
  calculator: "ShotBoundaryCalculator"
  input_stream: "VIDEO:video_frames_scaled"
  output_stream: "IS_SHOT_CHANGE:shot_change"
  options {
    [type.googleapis.com/mediapipe.autoflip.ShotBoundaryCalculatorOptions] {
      min_shot_span: 0.2
      min_motion: 0.3
      window_size: 15
      min_shot_measure: 10
      min_motion_with_shot_measure: 0.05
    }
  }
}

# DETECTION: find faces on the down sampled stream
node {
  calculator: "AutoFlipFaceDetectionSubgraph"
  input_stream: "VIDEO:video_frames_scaled_downsampled"
  output_stream: "DETECTIONS:face_detections"
}
node {
  calculator: "FaceToRegionCalculator"
  input_stream: "VIDEO:video_frames_scaled_downsampled"
  input_stream: "FACES:face_detections"
  output_stream: "REGIONS:face_regions"
}
```

----------------------------------------

TITLE: Building a PassThrough MediaPipe Graph in C++
DESCRIPTION: This C++ function `BuildGraphConfig()` programmatically constructs a MediaPipe graph equivalent to the proto example. It demonstrates how to define graph inputs and outputs, add nodes using `graph.AddNode()`, and connect streams using a functional approach, providing a flexible way to build complex pipelines.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/graphs.md#_snippet_1

LANGUAGE: cpp
CODE:
```
CalculatorGraphConfig BuildGraphConfig() {
  Graph graph;

  // Graph inputs
  Stream<AnyType> in = graph.In(0).SetName("in");

  auto pass_through_fn = [](Stream<AnyType> in,
                            Graph& graph) -> Stream<AnyType> {
    auto& node = graph.AddNode("PassThroughCalculator");
    in.ConnectTo(node.In(0));
    return node.Out(0);
  };

  Stream<AnyType> out1 = pass_through_fn(in, graph);
  Stream<AnyType> out2 = pass_through_fn(out1, graph);
  Stream<AnyType> out3 = pass_through_fn(out2, graph);
  Stream<AnyType> out4 = pass_through_fn(out3, graph);

  // Graph outputs
  out4.SetName("out").ConnectTo(graph.Out(0));

  return graph.GetConfig();
}
```

----------------------------------------

TITLE: Building MediaPipe Hand Tracking for CPU (Bash)
DESCRIPTION: This Bazel command compiles the MediaPipe Hand Tracking example application for CPU execution. It explicitly disables GPU support using `--define MEDIAPIPE_DISABLE_GPU=1` and targets the `hand_tracking_cpu` executable.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/cpp.md#_snippet_0

LANGUAGE: Bash
CODE:
```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu
```

----------------------------------------

TITLE: Importing MediaPipe Selfie Segmentation and Dependencies (Python)
DESCRIPTION: This snippet imports necessary libraries for using MediaPipe Selfie Segmentation in Python. It includes `cv2` for OpenCV functionalities, `mediapipe` for the core MediaPipe library, `numpy` for numerical operations, and specifically imports `drawing_utils` and `selfie_segmentation` modules from MediaPipe solutions. These imports are prerequisites for setting up and running selfie segmentation tasks.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/selfie_segmentation.md#_snippet_0

LANGUAGE: Python
CODE:
```
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
```

----------------------------------------

TITLE: Importing MediaPipe Face Detection and Drawing Utilities in Python
DESCRIPTION: This snippet imports the essential libraries for utilizing MediaPipe's Face Detection solution in Python. It includes `cv2` for image processing, the main `mediapipe` library, and specific modules for face detection and drawing utilities to facilitate visualization of detection results.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_detection.md#_snippet_0

LANGUAGE: Python
CODE:
```
import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
```

----------------------------------------

TITLE: Linking MediaPipe Calculators with Bazel `alwayslink`
DESCRIPTION: This Bazel 'cc_library' rule demonstrates how to correctly link a new MediaPipe calculator. Setting 'alwayslink = True' ensures that the calculator's registration, performed via the 'REGISTER_CALCULATOR' macro, is not removed by the linker, preventing 'No registered calculator found' runtime errors.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/troubleshooting.md#_snippet_2

LANGUAGE: bazel
CODE:
```
cc_library(
    name = "our_new_calculator",
    srcs = ["our_new_calculator.cc"],
    deps = [ ... ],
    alwayslink = True,
)
```

----------------------------------------

TITLE: Fusing Detection Signals with SignalFusingCalculator (MediaPipe Graph Configuration)
DESCRIPTION: This node combines various detection signals (shot changes, face regions, object regions) into `salient_regions` using the `SignalFusingCalculator`. It applies specific minimum and maximum score thresholds for different signal types like faces, humans, pets, and cars, indicating their importance in the fusion process.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/autoflip.md#_snippet_5

LANGUAGE: MediaPipe Graph Configuration
CODE:
```
node {
  calculator: "SignalFusingCalculator"
  input_stream: "shot_change"
  input_stream: "face_regions"
  input_stream: "object_regions"
  output_stream: "salient_regions"
  options {
    [type.googleapis.com/mediapipe.autoflip.SignalFusingCalculatorOptions] {
      signal_settings {
        type { standard: FACE_CORE_LANDMARKS }
        min_score: 0.85
        max_score: 0.9
        is_required: false
      }
      signal_settings {
        type { standard: FACE_ALL_LANDMARKS }
        min_score: 0.8
        max_score: 0.85
        is_required: false
      }
      signal_settings {
        type { standard: FACE_FULL }
        min_score: 0.8
        max_score: 0.85
        is_required: false
      }
      signal_settings {
        type: { standard: HUMAN }
        min_score: 0.75
        max_score: 0.8
        is_required: false
      }
      signal_settings {
        type: { standard: PET }
        min_score: 0.7
        max_score: 0.75
        is_required: false
      }
      signal_settings {
        type: { standard: CAR }
        min_score: 0.7
        max_score: 0.75
        is_required: false
      }
      signal_settings {
        type: { standard: OBJECT }
        min_score: 0.1
        max_score: 0.2
        is_required: false
      }
    }
  }
}
```

----------------------------------------

TITLE: Initializing and Running MediaPipe Objectron for Object Detection
DESCRIPTION: This JavaScript code initializes MediaPipe Objectron, configures its options like model name and max objects, and sets up a camera feed. It defines an onResults callback to process and draw detected objects, including bounding boxes and centroids, onto a canvas element.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_3

LANGUAGE: JavaScript
CODE:
```
<script type="module">
const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');

function onResults(results) {
  canvasCtx.save();
  canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);
  if (!!results.objectDetections) {
    for (const detectedObject of results.objectDetections) {
      // Reformat keypoint information as landmarks, for easy drawing.
      const landmarks: mpObjectron.Point2D[] =
          detectedObject.keypoints.map(x => x.point2d);
      // Draw bounding box.
      drawingUtils.drawConnectors(canvasCtx, landmarks,
          mpObjectron.BOX_CONNECTIONS, {color: '#FF0000'});
      // Draw centroid.
      drawingUtils.drawLandmarks(canvasCtx, [landmarks[0]], {color: '#FFFFFF'});
    }
  }
  canvasCtx.restore();
}

const objectron = new Objectron({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/objectron/${file}`;
}});
objectron.setOptions({
  modelName: 'Chair',
  maxNumObjects: 3,
});
objectron.onResults(onResults);

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await objectron.send({image: videoElement});
  },
  width: 1280,
  height: 720
});
camera.start();
</script>
```

----------------------------------------

TITLE: Initializing MediaPipe CalculatorGraph with Binary Protobuf (Python)
DESCRIPTION: This snippet shows how to initialize a MediaPipe CalculatorGraph by loading a pre-compiled binary protobuf file. It demonstrates observing an output stream and printing the received packets, which is useful for deploying graphs defined externally.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_32

LANGUAGE: Python
CODE:
```
import mediapipe as mp
# resources dependency

graph = mp.CalculatorGraph(
    binary_graph=os.path.join(
        resources.GetRunfilesDir(), 'path/to/your/graph.binarypb'))
graph.observe_output_stream(
    'out_stream',
    lambda stream_name, packet: print(f'Get {packet} from {stream_name}'))
```

----------------------------------------

TITLE: Initializing Preview Display View in onCreate (Java)
DESCRIPTION: This Java code, placed within the `onCreate` method of `MainActivity`, initializes the `previewDisplayView` as a new `SurfaceView` and then calls `setupPreviewDisplayView()` to configure its layout. This setup occurs before requesting camera permissions.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_16

LANGUAGE: Java
CODE:
```
previewDisplayView = new SurfaceView(this);
setupPreviewDisplayView();
```

----------------------------------------

TITLE: Running MediaPipe Hello World Example (GPU)
DESCRIPTION: This Bazel command runs the MediaPipe 'Hello World' C++ example with GPU acceleration enabled on a Linux desktop, assuming Mesa drivers are installed. The `--copt` flags pass specific compiler options (`-DMESA_EGL_NO_X11_HEADERS`, `-DEGL_NO_X11`) to leverage the GPU for processing.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_15

LANGUAGE: bash
CODE:
```
$ bazel run --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 \
        mediapipe/examples/desktop/hello_world:hello_world
```

----------------------------------------

TITLE: Creating a 64-bit Unsigned Integer MediaPipe Packet in Python
DESCRIPTION: Creates a MediaPipe packet with a 64-bit unsigned integer payload (mapped to C++ `uint64_t`) using `mp.packet_creator.create_uint64`. This is suitable for very large unsigned integer values.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_18

LANGUAGE: Python
CODE:
```
create_uint64(2**64-1)
```

----------------------------------------

TITLE: Configuring Bazel for CUDA Support in MediaPipe
DESCRIPTION: These Bazel build configurations define flags for enabling CUDA support in MediaPipe. The `using_cuda` config ensures CUDA is available, while the `cuda` config specifically enables building CUDA op kernels using `nvcc`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/gpu_support.md#_snippet_8

LANGUAGE: Bazel
CODE:
```
# This config refers to building with CUDA available. It does not necessarily
# mean that we build CUDA op kernels.
build:using_cuda --define=using_cuda=true
build:using_cuda --action_env TF_NEED_CUDA=1
build:using_cuda --crosstool_top=@local_config_cuda//crosstool:toolchain

# This config refers to building CUDA op kernels with nvcc.
build:cuda --config=using_cuda
build:cuda --define=using_cuda_nvcc=true
```

----------------------------------------

TITLE: Closing MediaPipe CalculatorGraph (Python)
DESCRIPTION: This snippet demonstrates how to properly close a MediaPipe CalculatorGraph after its processing is complete. Closing the graph releases resources and allows for a potential restart for subsequent runs, ensuring efficient resource management.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_34

LANGUAGE: Python
CODE:
```
graph.close()
```

----------------------------------------

TITLE: Installing Pre-compiled OpenCV and FFmpeg Libraries via APT
DESCRIPTION: This command installs pre-compiled OpenCV development libraries and FFmpeg (via libopencv-video-dev) using the apt package manager. This is the first option for setting up OpenCV dependencies for MediaPipe.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_35

LANGUAGE: bash
CODE:
```
sudo apt-get install libopencv-core-dev libopencv-highgui-dev \
                           libopencv-calib3d-dev libopencv-features2d-dev \
                           libopencv-imgproc-dev libopencv-video-dev
```

----------------------------------------

TITLE: Starting MediaPipe Graph Run and Feeding Packets (Python)
DESCRIPTION: This snippet illustrates how to start a MediaPipe graph run and feed various types of packets into its input stream. It provides examples of adding a string packet and an image frame packet, demonstrating how to provide data to the graph for processing.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_33

LANGUAGE: Python
CODE:
```
graph.start_run()

graph.add_packet_to_input_stream(
    'in_stream', mp.packet_creator.create_string('abc').at(0))

rgb_img = cv2.cvtColor(cv2.imread('/path/to/your/image.png'), cv2.COLOR_BGR2RGB)
graph.add_packet_to_input_stream(
    'in_stream',
    mp.packet_creator.create_image_frame(image_format=mp.ImageFormat.SRGB,
                                         data=rgb_img).at(1))
```

----------------------------------------

TITLE: Configuring OpenCV C++ Library in Bazel
DESCRIPTION: This Bazel configuration snippet defines the include paths and linker options for integrating OpenCV 4.x into a C++ project. It specifies the header search paths and links against various OpenCV shared libraries required for core image processing, calibration, features, GUI, codecs, and video functionalities.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_8

LANGUAGE: Bazel
CODE:
```
        "include/opencv4/opencv2/**/*.h*",
      ]),
      includes = [
        "include/opencv4/",
      ],
      linkopts = [
        "-L/usr/local/lib",
        "-l:libopencv_core.so",
        "-l:libopencv_calib3d.so",
        "-l:libopencv_features2d.so",
        "-l:libopencv_highgui.so",
        "-l:libopencv_imgcodecs.so",
        "-l:libopencv_imgproc.so",
        "-l:libopencv_video.so",
        "-l:libopencv_videoio.so",
      ]
    }
```

----------------------------------------

TITLE: Creating and Retrieving Custom Type MediaPipe Packets in C++
DESCRIPTION: This C++ code defines functions for creating and retrieving MediaPipe packets containing custom C++ types. The 'create_my_type' function converts a 'MyType' object into a MediaPipe Packet, while 'get_my_type' validates and extracts a 'MyType' object from a given Packet, ensuring type safety.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_27

LANGUAGE: C++
CODE:
```
#include "path/to/my_type/header/file.h"
#include "mediapipe/framework/packet.h"
#include "pybind11/pybind11.h"

namespace mediapipe {
namespace py = pybind11;

PYBIND11_MODULE(my_packet_methods, m) {
  m.def(
      "create_my_type",
      [](const MyType& my_type) { return MakePacket<MyType>(my_type); });

  m.def(
      "get_my_type",
      [](const Packet& packet) {
        if(!packet.ValidateAsType<MyType>().ok()) {

```

----------------------------------------

TITLE: Installing OpenCV and FFmpeg via APT - Bash
DESCRIPTION: This command installs pre-compiled OpenCV and FFmpeg libraries on Debian/Ubuntu systems using the `apt-get` package manager. It includes core, highgui, calib3d, features2d, imgproc, and video development libraries, with FFmpeg being a dependency of `libopencv-video-dev`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_2

LANGUAGE: Bash
CODE:
```
$ sudo apt-get install -y \
    libopencv-core-dev \
    libopencv-highgui-dev \
    libopencv-calib3d-dev \
    libopencv-features2d-dev \
    libopencv-imgproc-dev \
    libopencv-video-dev
```

----------------------------------------

TITLE: Building MediaPipe Iris CPU Video Input Application (Bash)
DESCRIPTION: This command builds the MediaPipe Iris application for CPU-based video input. It uses Bazel to compile the C++ example, disabling GPU support for CPU optimization. This is a prerequisite for running the application with video files.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/iris.md#_snippet_0

LANGUAGE: bash
CODE:
```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/iris_tracking:iris_tracking_cpu_video_input
```

----------------------------------------

TITLE: Accessing Input Packet (New Node API, C++)
DESCRIPTION: This code illustrates the more concise method of accessing an input packet using the new Node API. It leverages the previously declared typed constant (`kSelect`) and the `CalculatorContext` to retrieve the packet.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/api2/README.md#_snippet_4

LANGUAGE: C++
CODE:
```
int select = kSelect(cc).Get();  // alternative: *kSelect(cc)
```

----------------------------------------

TITLE: Listing Connected Android Devices (ADB Bash)
DESCRIPTION: This `adb` command lists all Android devices currently connected to the development machine. It's used to verify that the device is properly recognized before proceeding with app installation or file operations.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/tools/tracing_and_profiling.md#_snippet_2

LANGUAGE: bash
CODE:
```
adb devices
```

----------------------------------------

TITLE: Creating a 64-bit Signed Integer MediaPipe Packet in Python
DESCRIPTION: Creates a MediaPipe packet with a 64-bit signed integer payload (mapped to C++ `int64_t`) using `mp.packet_creator.create_int64`. This is suitable for large integer values.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_10

LANGUAGE: Python
CODE:
```
create_int64(2**63-1)
```

----------------------------------------

TITLE: Converting Points from NDC to Pixel Space - Mathematical
DESCRIPTION: These formulas illustrate the conversion of coordinates from Normalized Device Coordinates (NDC), where x and y are in [-1, 1], to pixel space. The conversion scales and shifts the NDC values based on the image's width and height to map them to pixel coordinates.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_16

LANGUAGE: mathematical
CODE:
```
x_pixel = (1 + x_ndc) / 2.0 * image_width
y_pixel = (1 - y_ndc) / 2.0 * image_height
```

----------------------------------------

TITLE: Running MediaPipe C++ Hello World Example in Docker (Bash)
DESCRIPTION: This snippet runs the previously built 'mediapipe' Docker image in interactive mode, then executes the MediaPipe C++ 'Hello World' example within the container. It disables GPU usage for the example. This demonstrates basic functionality and confirms the Docker setup.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_39

LANGUAGE: bash
CODE:
```
$ docker run -it --name mediapipe mediapipe:latest

root@bca08b91ff63:/mediapipe# GLOG_logtostderr=1 bazel run --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hello_world
```

----------------------------------------

TITLE: Loading MediaPipe and OpenCV Native Libraries in Android MainActivity
DESCRIPTION: This Java static initializer block in `MainActivity` loads the required native libraries for MediaPipe (`mediapipe_jni`) and OpenCV (`opencv_java4`). This ensures that the underlying C++ and JNI components of the MediaPipe framework and its OpenCV dependency are loaded into the application's memory before any MediaPipe operations are performed.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_35

LANGUAGE: Java
CODE:
```
static {
  // Load all native libraries needed by the app.
  System.loadLibrary("mediapipe_jni");
  System.loadLibrary("opencv_java4");
}
```

----------------------------------------

TITLE: Running MediaPipe Hand Tracking on CPU (Bash)
DESCRIPTION: This command executes the previously built MediaPipe Hand Tracking application on the CPU. It sets the `GLOG_logtostderr=1` environment variable for logging and specifies the `hand_tracking_desktop_live.pbtxt` configuration file, typically used for live webcam input.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/cpp.md#_snippet_1

LANGUAGE: Bash
CODE:
```
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu \
  --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt
```

----------------------------------------

TITLE: Initializing MediaPipe Native Asset Manager in Android
DESCRIPTION: Initializes the MediaPipe native asset manager. This is crucial for MediaPipe's native libraries to access application assets, such as compiled binary graphs (.binarypb files), ensuring they can be loaded and used by the framework. This should be called early in the application lifecycle, typically within the `onCreate` method of an Activity.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_36

LANGUAGE: Java
CODE:
```
// Initialize asset manager so that MediaPipe native libraries can access the app assets, e.g.,
// binary graphs.
AndroidAssetUtil.initializeNativeAssetManager(this);
```

----------------------------------------

TITLE: Defining TwoPassThroughSubgraph in MediaPipe Graph Config
DESCRIPTION: This snippet demonstrates how to define a MediaPipe subgraph, named 'TwoPassThroughSubgraph', in a `.pbtxt` file. It specifies the subgraph's type, its public input and output streams, and the internal calculator nodes that constitute its functionality, effectively encapsulating a sequence of operations.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/graphs.md#_snippet_2

LANGUAGE: MediaPipe Graph Config
CODE:
```
# This subgraph is defined in two_pass_through_subgraph.pbtxt
# and is registered as "TwoPassThroughSubgraph"

type: "TwoPassThroughSubgraph"
input_stream: "out1"
output_stream: "out3"

node {
    calculator: "PassThroughCalculator"
    input_stream: "out1"
    output_stream: "out2"
}
node {
    calculator: "PassThroughCalculator"
    input_stream: "out2"
    output_stream: "out3"
}
```

----------------------------------------

TITLE: Configuring MediaPipe for Custom OpenCV Installation (Bazel)
DESCRIPTION: This Bazel configuration snippet shows how to modify the `WORKSPACE` and `opencv_linux.BUILD` files to point MediaPipe to a manually built OpenCV installation, typically located in `/usr/local`. It defines a `new_local_repository` for OpenCV and a `cc_library` rule to link against its shared libraries and headers.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_36

LANGUAGE: bazel
CODE:
```
new_local_repository(
    name = "linux_opencv",
    build_file = "@//third_party:opencv_linux.BUILD",
    path = "/usr/local",
)

cc_library(
    name = "opencv",
    srcs = glob(
        [
            "lib/libopencv_core.so",
            "lib/libopencv_highgui.so",
            "lib/libopencv_imgcodecs.so",
            "lib/libopencv_imgproc.so",
            "lib/libopencv_video.so",
            "lib/libopencv_videoio.so"
        ]
    ),
    hdrs = glob(["include/opencv4/**/*.h*"]),
    includes = ["include/opencv4/"],
    linkstatic = 1,
    visibility = ["//visibility:public"]
)
```

----------------------------------------

TITLE: Installing OpenCV and FFmpeg via Homebrew (macOS)
DESCRIPTION: These commands use Homebrew to install OpenCV 3, which also pulls in FFmpeg as a dependency. The second command addresses a known issue by uninstalling the `glog` dependency to prevent build conflicts.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_21

LANGUAGE: bash
CODE:
```
$ brew install opencv@3

# There is a known issue caused by the glog dependency. Uninstall glog.
$ brew uninstall --ignore-dependencies glog
```

----------------------------------------

TITLE: Retrieving a String from a MediaPipe Packet (UTF-8) in Python
DESCRIPTION: Retrieves the UTF-8 string payload (mapped to C++ `std::string`) from a MediaPipe packet using `mp.packet_getter.get_str`. This allows access to the string content of the packet.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_25

LANGUAGE: Python
CODE:
```
get_str(packet)
```

----------------------------------------

TITLE: Retrieving a Boolean from a MediaPipe Packet in Python
DESCRIPTION: Retrieves the boolean payload from a MediaPipe packet using `mp.packet_getter.get_bool`. The retrieved content is a copy, meaning modifications to it will not affect the original packet's immutable payload.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_1

LANGUAGE: Python
CODE:
```
get_bool(packet)
```

----------------------------------------

TITLE: Retrieving an 8-bit Signed Integer from a MediaPipe Packet in Python
DESCRIPTION: Retrieves the 8-bit signed integer payload (mapped to C++ `int8_t`) from a MediaPipe packet using `mp.packet_getter.get_int`. The `get_int` method handles various signed integer sizes.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_5

LANGUAGE: Python
CODE:
```
get_int(packet)
```

----------------------------------------

TITLE: Loading MediaPipe Graph from Resource in Objective-C
DESCRIPTION: A static helper method that loads a MediaPipe graph configuration from a `.binarypb` resource file, parses it into a `mediapipe::CalculatorGraphConfig` proto, and initializes an `MPPGraph` object. It also adds the specified output stream to the graph, enabling packet reception.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_24

LANGUAGE: Objective-C
CODE:
```
+ (MPPGraph*)loadGraphFromResource:(NSString*)resource {
  // Load the graph config resource.
  NSError* configLoadError = nil;
  NSBundle* bundle = [NSBundle bundleForClass:[self class]];
  if (!resource || resource.length == 0) {
    return nil;
  }
  NSURL* graphURL = [bundle URLForResource:resource withExtension:@"binarypb"];
  NSData* data = [NSData dataWithContentsOfURL:graphURL options:0 error:&configLoadError];
  if (!data) {
    NSLog(@"Failed to load MediaPipe graph config: %@", configLoadError);
    return nil;
  }

  // Parse the graph config resource into mediapipe::CalculatorGraphConfig proto object.
  mediapipe::CalculatorGraphConfig config;
  config.ParseFromArray(data.bytes, data.length);

  // Create MediaPipe graph with mediapipe::CalculatorGraphConfig proto object.
  MPPGraph* newGraph = [[MPPGraph alloc] initWithGraphConfig:config];
  [newGraph addFrameOutputStream:kOutputStream outputPacketType:MPPPacketTypePixelBuffer];
  return newGraph;
}
```

----------------------------------------

TITLE: Configuring MediaPipe Graph Properties in Android Manifest Values
DESCRIPTION: This Bazel `BUILD` rule snippet defines custom `manifest_values` that embed MediaPipe-specific properties directly into the Android application's manifest. These values, such as `binaryGraphName`, `inputVideoStreamName`, and `outputVideoStreamName`, are crucial for `MainActivity` to correctly initialize and interact with the MediaPipe graph at runtime.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_34

LANGUAGE: Bazel
CODE:
```
manifest_values = {
    "applicationId": "com.google.mediapipe.apps.basic",
    "appName": "Hello World",
    "mainActivity": ".MainActivity",
    "cameraFacingFront": "False",
    "binaryGraphName": "mobile_gpu.binarypb",
    "inputVideoStreamName": "input_video",
    "outputVideoStreamName": "output_video",
},
```

----------------------------------------

TITLE: Bazel Compilation Flags for MediaPipe GPU Support
DESCRIPTION: These Bazel compilation options (`--copt`) are used to enable GPU support when building MediaPipe examples. They define preprocessor macros (`-DMESA_EGL_NO_X11_HEADERS`, `-DEGL_NO_X11`) that instruct the compiler to use Mesa EGL drivers without X11 dependencies, crucial for GPU-accelerated builds on Linux.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_12

LANGUAGE: bash
CODE:
```
--copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11
```

----------------------------------------

TITLE: Configuring a Simple PassThrough MediaPipe Graph in Proto
DESCRIPTION: This `CalculatorGraphConfig` proto defines a basic MediaPipe graph consisting of four `PassThroughCalculator` nodes. It demonstrates how to declare input and output streams at the graph level and connect sequential nodes using intermediate streams, illustrating a simple data flow through the graph.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/graphs.md#_snippet_0

LANGUAGE: proto
CODE:
```
# This graph named main_pass_throughcals_nosubgraph.pbtxt contains 4
# passthrough calculators.
input_stream: "in"
output_stream: "out"
node {
    calculator: "PassThroughCalculator"
    input_stream: "in"
    output_stream: "out1"
}
node {
    calculator: "PassThroughCalculator"
    input_stream: "out1"
    output_stream: "out2"
}
node {
    calculator: "PassThroughCalculator"
    input_stream: "out2"
    output_stream: "out3"
}
node {
    calculator: "PassThroughCalculator"
    input_stream: "out3"
    output_stream: "out"
}
```

----------------------------------------

TITLE: Setting MediaPipe Graph Delegate in Objective-C
DESCRIPTION: Assigns the current `ViewController` instance as the delegate for the `mediapipeGraph` object. This enables the `ViewController` to receive callbacks for output packets from the graph, such as processed video frames.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_26

LANGUAGE: Objective-C
CODE:
```
self.mediapipeGraph.delegate = self;
```

----------------------------------------

TITLE: Installing Mesa GPU Driver Development Libraries
DESCRIPTION: This `apt-get` command installs essential Mesa development packages on Linux. These libraries (`mesa-common-dev`, `libegl1-mesa-dev`, `libgles2-mesa-dev`) provide EGL and GLES2 support, which are prerequisites for compiling and running MediaPipe examples with GPU acceleration.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_11

LANGUAGE: bash
CODE:
```
sudo apt-get install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev
```

----------------------------------------

TITLE: Specifying Calculator and Subgraph Options in MediaPipe Graph
DESCRIPTION: This snippet demonstrates how to directly specify options for a FlowLimiterCalculator and a FaceDetectionSubgraph within a CalculatorGraphConfig using the node_options field. It shows how to pass literal configuration values to individual nodes.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/graphs.md#_snippet_5

LANGUAGE: protobuf
CODE:
```
node {
  calculator: "FlowLimiterCalculator"
  input_stream: "image"
  output_stream: "throttled_image"
  node_options: {
    [type.googleapis.com/mediapipe.FlowLimiterCalculatorOptions] {
      max_in_flight: 1
    }
  }
}

node {
  calculator: "FaceDetectionSubgraph"
  input_stream: "IMAGE:throttled_image"
  node_options: {
    [type.googleapis.com/mediapipe.FaceDetectionOptions] {
      tensor_width: 192
      tensor_height: 192
    }
  }
}
```

----------------------------------------

TITLE: Configuring Bazel for OpenCV 4 (Package Manager) - Bazel
DESCRIPTION: This Bazel `WORKSPACE` and `cc_library` configuration is for MediaPipe to link against OpenCV 4 libraries installed via a Debian package manager. It includes `hdrs` and `includes` to account for multiarch paths and specifies the necessary OpenCV shared libraries for linking, requiring uncommenting based on the system's architecture.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_5

LANGUAGE: Bazel
CODE:
```
# WORKSPACE
new_local_repository(
  name = "linux_opencv",
  build_file = "@//third_party:opencv_linux.BUILD",
  path = "/usr",
)

# opencv_linux.BUILD for OpenCV 4 installed from Debian package
cc_library(
  name = "opencv",
  hdrs = glob([
    # Uncomment according to your multiarch value (gcc -print-multiarch):
    #  "include/aarch64-linux-gnu/opencv4/opencv2/cvconfig.h",
    #  "include/arm-linux-gnueabihf/opencv4/opencv2/cvconfig.h",
    #  "include/x86_64-linux-gnu/opencv4/opencv2/cvconfig.h",
    "include/opencv4/opencv2/**/*.h*"
  ]),
  includes = [
    # Uncomment according to your multiarch value (gcc -print-multiarch):
    #  "include/aarch64-linux-gnu/opencv4/",
    #  "include/arm-linux-gnueabihf/opencv4/",
    #  "include/x86_64-linux-gnu/opencv4/",
    "include/opencv4/"
  ],
  linkopts = [
    "-l:libopencv_core.so",
    "-l:libopencv_calib3d.so",
    "-l:libopencv_features2d.so",
    "-l:libopencv_highgui.so",
    "-l:libopencv_imgcodecs.so",
    "-l:libopencv_imgproc.so",
    "-l:libopencv_video.so",
    "-l:libopencv_videoio.so"
  ]
)
```

----------------------------------------

TITLE: Defining Calculator Contract in C++ for MediaPipe
DESCRIPTION: This C++ snippet shows the `GetContract` method for a MediaPipe calculator. It defines the expected input and output stream types and their identification methods (by index, tag, or tag and index), ensuring type compatibility within the graph.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/calculators.md#_snippet_3

LANGUAGE: C++
CODE:
```
// c++ Code snippet describing the SomeAudioVideoCalculator GetContract() method
class SomeAudioVideoCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    // SetAny() is used to specify that whatever the type of the
    // stream is, it's acceptable.  This does not mean that any
    // packet is acceptable.  Packets in the stream still have a
    // particular type.  SetAny() has the same effect as explicitly
    // setting the type to be the stream's type.
    cc->Outputs().Tag("VIDEO").Set<ImageFrame>();
    cc->Outputs().Get("AUDIO", 0).Set<Matrix>();
    cc->Outputs().Get("AUDIO", 1).Set<Matrix>();
    return absl::OkStatus();
  }
```

----------------------------------------

TITLE: Running MediaPipe Object Detection with TFLite Model (Desktop)
DESCRIPTION: This command executes the built MediaPipe object detection application, processing a video file using the TFLite model. It directs GLog output to stderr and specifies the calculator graph configuration file. Users must replace placeholders for input and output video paths.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/object_detection.md#_snippet_1

LANGUAGE: bash
CODE:
```
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_tflite \
  --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_tflite_graph.pbtxt \
  --input_side_packets=input_video_path=<input video path>,output_video_path=<output video path>
```

----------------------------------------

TITLE: Setting GLOG Logging to Standard Error
DESCRIPTION: This `export` command sets the `GLOG_logtostderr` environment variable to `1`. This configuration directs all Google Logging (GLOG) output, typically used by MediaPipe, to the standard error stream, which is beneficial for real-time debugging and monitoring application behavior during execution.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_13

LANGUAGE: bash
CODE:
```
$ export GLOG_logtostderr=1
```

----------------------------------------

TITLE: Declaring Camera Preview Variables in MainActivity (Java)
DESCRIPTION: These Java declarations add `SurfaceTexture` and `SurfaceView` member variables to `MainActivity`. `previewFrameTexture` will hold the camera frames, while `previewDisplayView` is used to display these frames on the UI.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_15

LANGUAGE: Java
CODE:
```
private SurfaceTexture previewFrameTexture;
private SurfaceView previewDisplayView;
```

----------------------------------------

TITLE: Defining a MediaPipe Protobuf Library Build Rule
DESCRIPTION: This snippet defines a `mediapipe_proto_library` build rule, typically used in Bazel, to compile a protobuf schema for MediaPipe calculator options. It specifies the proto source file, visibility, and dependencies on core MediaPipe protobuf definitions.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/calculators.md#_snippet_6

LANGUAGE: Bazel Build Configuration
CODE:
```
  mediapipe_proto_library(
      name = "packet_cloner_calculator_proto",
      srcs = ["packet_cloner_calculator.proto"],
      visibility = ["//visibility:public"],
      deps = [
          "//mediapipe/framework:calculator_options_proto",
          "//mediapipe/framework:calculator_proto",
      ],
  )
```

----------------------------------------

TITLE: Configuring Bazel Build Rules for Android App (Bazel)
DESCRIPTION: This Bazel BUILD file defines rules for compiling the Android application. It includes an `android_library` rule for the source and resources, and an `android_binary` rule to build the final APK, replacing manifest placeholders and specifying dependencies.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_6

LANGUAGE: Bazel
CODE:
```
android_library(
    name = "basic_lib",
    srcs = glob(["*.java"]),
    manifest = "AndroidManifest.xml",
    resource_files = glob(["res/**"]),
    deps = [
        "@maven//:androidx_appcompat_appcompat",
        "@maven//:androidx_constraintlayout_constraintlayout",
    ],
)

android_binary(
    name = "helloworld",
    manifest = "AndroidManifest.xml",
    manifest_values = {
        "applicationId": "com.google.mediapipe.apps.basic",
        "appName": "Hello World",
        "mainActivity": ".MainActivity",
    },
    multidex = "native",
    deps = [
        ":basic_lib",
    ],
)
```

----------------------------------------

TITLE: Running MediaPipe Iris CPU Video Input Application (Bash)
DESCRIPTION: This command executes the previously built MediaPipe Iris application for CPU-based video input. It requires specifying the path to the calculator graph configuration file and the input/output video file paths as side packets. Replace <input video path> and <output video path> with actual file paths.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/iris.md#_snippet_1

LANGUAGE: bash
CODE:
```
bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_tracking_cpu_video_input \
  --calculator_graph_config_file=mediapipe/graphs/iris_tracking/iris_tracking_cpu_video_input.pbtxt \
  --input_side_packets=input_video_path=<input video path>,output_video_path=<output video path>
```

----------------------------------------

TITLE: Setting Bazel Environment Variables for MSVC on Windows
DESCRIPTION: These commands set environment variables required by Bazel for building C++ projects with Microsoft Visual C++ (MSVC) on Windows. BAZEL_VS points to the Visual Studio installation, BAZEL_VC to the VC tools directory, and BAZEL_VC_FULL_VERSION and BAZEL_WINSDK_FULL_VERSION specify the exact versions of the Visual C++ compiler and Windows SDK, respectively. These are crucial for Bazel to locate the correct build tools.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_29

LANGUAGE: Shell
CODE:
```
# Please find the exact paths and version numbers from your local version.
C:\> set BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools
C:\> set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC
C:\> set BAZEL_VC_FULL_VERSION=<Your local VC version>
C:\> set BAZEL_WINSDK_FULL_VERSION=<Your local WinSDK version>
```

----------------------------------------

TITLE: MediaPipe Graph Configuration for Visualizer (Bash)
DESCRIPTION: This snippet shows the raw text proto configuration for a MediaPipe graph, intended for use with the MediaPipe Visualizer. It defines an input stream "in", an output stream "out", and two serially connected `PassThroughCalculator` nodes, illustrating the graph's structure.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_cpp.md#_snippet_2

LANGUAGE: bash
CODE:
```
    input_stream: "in"
    output_stream: "out"
    node {
      calculator: "PassThroughCalculator"
      input_stream: "in"
      output_stream: "out1"
    }
    node {
      calculator: "PassThroughCalculator"
      input_stream: "out1"
      output_stream: "out"
    }
```

----------------------------------------

TITLE: Activating Python Virtual Environment for Source Build
DESCRIPTION: This command creates and activates a Python virtual environment, `mp_env`, specifically for managing dependencies when building the MediaPipe Python package from its source code. It isolates the build environment to prevent conflicts.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python.md#_snippet_5

LANGUAGE: Bash
CODE:
```
$ python3 -m venv mp_env && source mp_env/bin/activate
```

----------------------------------------

TITLE: Running MediaPipe Object Detection with TensorFlow Model (Desktop)
DESCRIPTION: This command runs the MediaPipe object detection application using a TensorFlow model, processing a video file. It outputs GLog messages to stderr, specifies the TensorFlow-specific calculator graph, and requires users to provide paths for the input and output video files.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/object_detection.md#_snippet_3

LANGUAGE: bash
CODE:
```
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_tflite \
  --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_tensorflow_graph.pbtxt \
  --input_side_packets=input_video_path=<input video path>,output_video_path=<output video path>
```

----------------------------------------

TITLE: Registering MediaPipe Subgraph with Bazel BUILD Rule
DESCRIPTION: This code illustrates how to register a MediaPipe subgraph using the `mediapipe_simple_subgraph` Bazel BUILD rule. It links the subgraph's definition file (`graph`), assigns a component name (`register_as`) for its use in other graphs, and declares its necessary dependencies.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/graphs.md#_snippet_3

LANGUAGE: Bazel BUILD
CODE:
```
# Small section of BUILD file for registering the "TwoPassThroughSubgraph"
# subgraph for use by main graph main_pass_throughcals.pbtxt

mediapipe_simple_subgraph(
    name = "twopassthrough_subgraph",
    graph = "twopassthrough_subgraph.pbtxt",
    register_as = "TwoPassThroughSubgraph",
    deps = [
            "//mediapipe/calculators/core:pass_through_calculator",
            "//mediapipe/framework:calculator_graph",
    ],
)
```

----------------------------------------

TITLE: Defining MediaPipe JNI Library in Bazel BUILD
DESCRIPTION: This Bazel `BUILD` rule defines a shared C++ binary (`libmediapipe_jni.so`) for the MediaPipe JNI framework and a `cc_library` to link against it. It ensures the MediaPipe framework's JNI code is built and made available for the Android application, serving as a prerequisite for using MediaPipe graphs.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_31

LANGUAGE: Bazel
CODE:
```
cc_binary(
    name = "libmediapipe_jni.so",
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
    ],
)

cc_library(
    name = "mediapipe_jni_lib",
    srcs = [":libmediapipe_jni.so"],
    alwayslink = 1,
)
```

----------------------------------------

TITLE: Cloning MediaPipe Repository (Bash)
DESCRIPTION: This snippet clones the MediaPipe GitHub repository with a depth of 1 (shallow clone) and then changes the current directory into the newly cloned 'mediapipe' directory. This is the initial step to obtain the MediaPipe source code.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_16

LANGUAGE: bash
CODE:
```
$ git clone --depth 1 https://github.com/google/mediapipe.git

# Change directory into MediaPipe root directory
$ cd mediapipe
```

----------------------------------------

TITLE: Defining PassThroughNodeBuilder Utility Class in C++
DESCRIPTION: This C++ class, `PassThroughNodeBuilder`, is a utility designed to simplify the construction of `PassThroughCalculator` nodes within a MediaPipe graph. It encapsulates the logic for connecting streams and casting types, reducing boilerplate and improving readability compared to manual graph construction.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_6

LANGUAGE: C++
CODE:
```
class PassThroughNodeBuilder {
 public:
  explicit PassThroughNodeBuilder(Graph& graph)
      : node_(graph.AddNode("PassThroughCalculator")) {}

  template <typename T>
  Stream<T> PassThrough(Stream<T> stream) {
    stream.ConnectTo(node_.In(index_));
    return node_.Out(index_++).Cast<T>();
  }

 private:
  int index_ = 0;
  GenericNode& node_;
};
```

----------------------------------------

TITLE: Configuring PassThroughCalculator in MediaPipe Graph Config
DESCRIPTION: This snippet demonstrates the configuration of the `PassThroughCalculator` in a MediaPipe graph definition. It shows how to declare input and output streams and connect them to the calculator node, ensuring that the order of output streams matches the inputs or uses explicit indexing.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_4

LANGUAGE: MediaPipe Graph Config
CODE:
```
input_stream: "float_value"
input_stream: "int_value"
input_stream: "bool_value"

output_stream: "passed_float_value"
output_stream: "passed_int_value"
output_stream: "passed_bool_value"

node {
  calculator: "PassThroughCalculator"
  input_stream: "float_value"
  input_stream: "int_value"
  input_stream: "bool_value"
  # The order must be the same as for inputs (or you can use explicit indexes)
  output_stream: "passed_float_value"
  output_stream: "passed_int_value"
  output_stream: "passed_bool_value"
}
```

----------------------------------------

TITLE: Setting MPPCameraInputSource Delegate and Queue (Objective-C)
DESCRIPTION: Assigns the `ViewController` as the delegate for `_cameraSource` and specifies `_videoQueue` as the dispatch queue on which the delegate methods, such as frame processing, will be invoked.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_9

LANGUAGE: Objective-C
CODE:
```
[_cameraSource setDelegate:self queue:_videoQueue];
```

----------------------------------------

TITLE: Building MediaPipe Demo Dataset with Bazel and Python
DESCRIPTION: This snippet outlines the steps to build the `media_sequence_demo` binary using Bazel and then generate a demo dataset using a Python script. It requires TensorFlow to be installed and the commands to be run from the MediaPipe repository's top directory. The output is TFRecord files in the specified `--path_to_demo_data`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/media_sequence/README.md#_snippet_0

LANGUAGE: Bash
CODE:
```
bazel build -c opt mediapipe/examples/desktop/media_sequence:media_sequence_demo \
  --define MEDIAPIPE_DISABLE_GPU=1

python -m mediapipe.examples.desktop.media_sequence.demo_dataset \
  --alsologtostderr \
  --path_to_demo_data=/tmp/demo_data/ \
  --path_to_mediapipe_binary=bazel-bin/mediapipe/examples/desktop/\
media_sequence/media_sequence_demo  \
  --path_to_graph_directory=mediapipe/graphs/media_sequence/
```

----------------------------------------

TITLE: Setting Clip Classification Metadata (Python/C++)
DESCRIPTION: These snippets demonstrate how to populate `SequenceExample` metadata for video clip classification. They set the video path, clip timestamps, and assign both integer indices and human-readable strings as labels for the entire video clip, used for model training and debugging respectively.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_0

LANGUAGE: python
CODE:
```
# Python: functions from media_sequence.py as ms
sequence = tf.train.SequenceExample()
ms.set_clip_data_path(b"path_to_video", sequence)
ms.set_clip_start_timestamp(1000000, sequence)
ms.set_clip_end_timestamp(6000000, sequence)
ms.set_clip_label_index((4, 3), sequence)
ms.set_clip_label_string((b"run", b"jump"), sequence)
```

LANGUAGE: c++
CODE:
```
// C++: functions from media_sequence.h
tensorflow::SequenceExample sequence;
SetClipDataPath("path_to_video", &sequence);
SetClipStartTimestamp(1000000, &sequence);
SetClipEndTimestamp(6000000, &sequence);
SetClipLabelIndex({4, 3}, &sequence);
SetClipLabelString({"run", "jump"}, &sequence);
```

----------------------------------------

TITLE: Declaring FrameProcessor in Android
DESCRIPTION: Declares a private instance of `FrameProcessor` within the Android Activity. The `FrameProcessor` is a core MediaPipe component responsible for sending camera frames to the MediaPipe graph, running the graph, processing its output, and preparing it for display. This declaration sets up the variable before its initialization.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_37

LANGUAGE: Java
CODE:
```
private FrameProcessor processor;
```

----------------------------------------

TITLE: Generating Unique iOS Bundle ID Prefix (Bash)
DESCRIPTION: This command executes a Python script to generate a unique bundle ID prefix for MediaPipe iOS demo applications. This is crucial for avoiding conflicts when installing apps on an iOS device, especially without a custom provisioning profile.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/ios.md#_snippet_4

LANGUAGE: bash
CODE:
```
python3 mediapipe/examples/ios/link_local_profiles.py
```

----------------------------------------

TITLE: Setting Up Docker for ARM32 Cross-Compilation
DESCRIPTION: This command initiates a Docker environment specifically configured for cross-compiling MediaPipe for ARM32 architectures, such as those found in Raspberry Pi devices. It prepares the necessary toolchains and dependencies within the container for subsequent Bazel builds.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/coral/README.md#_snippet_1

LANGUAGE: bash
CODE:
```
make -C mediapipe/examples/coral PLATFORM=armhf docker
```

----------------------------------------

TITLE: Generating Custom Kinetics-Formatted Data with MediaPipe
DESCRIPTION: This snippet demonstrates how to prepare custom video data in the Kinetics format using MediaPipe. It involves downloading a sample video, creating a custom CSV file mapping video paths and time segments, building the `media_sequence_demo` binary, and then running a Python script to process the data. TensorFlow is a prerequisite, and the commands should be run from the MediaPipe repository's top directory.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/media_sequence/README.md#_snippet_2

LANGUAGE: Bash
CODE:
```
echo "Credit for this video belongs to: ESA/Hubble; Music: Johan B. Monell"
wget https://cdn.spacetelescope.org/archives/videos/medium_podcast/heic1608c.mp4 -O /tmp/heic1608c.mp4
CUSTOM_CSV=/tmp/custom_kinetics.csv
VIDEO_PATH=/tmp/heic1608c.mp4
echo -e "video,time_start,time_end,split\n${VIDEO_PATH},0,10,custom" > ${CUSTOM_CSV}

bazel build -c opt mediapipe/examples/desktop/media_sequence:media_sequence_demo \
  --define MEDIAPIPE_DISABLE_GPU=1

python -m mediapipe.examples.desktop.media_sequence.kinetics_dataset \
  --alsologtostderr \
  --splits_to_process=custom \
  --path_to_custom_csv=${CUSTOM_CSV} \
  --video_path_format_string={video} \
  --path_to_kinetics_data=/tmp/ms/kinetics/ \
  --path_to_mediapipe_binary=bazel-bin/mediapipe/examples/desktop/\
media_sequence/media_sequence_demo  \
  --path_to_graph_directory=mediapipe/graphs/media_sequence/
```

----------------------------------------

TITLE: Building MediaPipe Python Wheel Package from Source
DESCRIPTION: This command builds a Python wheel (`.whl`) distribution package for MediaPipe from its source code. A wheel file is a standard, pre-built format for Python distributions, simplifying installation and deployment.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python.md#_snippet_8

LANGUAGE: Bash
CODE:
```
(mp_env)mediapipe$ python3 setup.py bdist_wheel
```

----------------------------------------

TITLE: Identifying Missing Packets with DefaultInputStreamHandler Logs
DESCRIPTION: This log message from `DefaultInputStreamHandler` indicates that an input set was filled at a specific timestamp (ts: 1) but was missing packets from the 'INPUT_B:0:input_b' stream. This helps diagnose scenarios where `Calculator::Process` is called with incomplete data due to unexpected timestamp bound increases.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/troubleshooting.md#_snippet_5

LANGUAGE: Log Output
CODE:
```
[INFO] SomeCalculator: Filled input set at ts: 1 with MISSING packets in input streams: INPUT_B:0:input_b.
```

----------------------------------------

TITLE: Configuring Bazel WORKSPACE for MacPorts OpenCV/FFmpeg
DESCRIPTION: These Bazel `new_local_repository` rules are added to the `WORKSPACE` file when using MacPorts. They define external repositories named 'macos_opencv' and 'macos_ffmpeg', pointing to the `/opt` directory where MacPorts typically installs packages, and linking to specific BUILD files for their definitions.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_23

LANGUAGE: BUILD
CODE:
```
new_local_repository(
    name = "macos_opencv",
    build_file = "@//third_party:opencv_macos.BUILD",
    path = "/opt",
)

new_local_repository(
    name = "macos_ffmpeg",
    build_file = "@//third_party:ffmpeg_macos.BUILD",
    path = "/opt",
)
```

----------------------------------------

TITLE: Building MediaPipe Hand Tracking for GPU (Bash)
DESCRIPTION: This Bazel command builds the MediaPipe Hand Tracking example for GPU execution, specifically on Linux. It includes C compiler options (`--copt`) to define `MESA_EGL_NO_X11_HEADERS` and `EGL_NO_X11`, which are crucial for OpenGL ES setup without X11 dependencies, targeting the `hand_tracking_gpu` executable.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/cpp.md#_snippet_2

LANGUAGE: Bash
CODE:
```
bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 \
  mediapipe/examples/desktop/hand_tracking:hand_tracking_gpu
```

----------------------------------------

TITLE: Setting Up Camera Preview Display View (Java)
DESCRIPTION: This Java method, `setupPreviewDisplayView()`, configures the `previewDisplayView`. It initially sets the view's visibility to `GONE` and then adds it to the `preview_display_layout` FrameLayout defined in the XML, preparing it to display camera frames.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_17

LANGUAGE: Java
CODE:
```
private void setupPreviewDisplayView() {
  previewDisplayView.setVisibility(View.GONE);
  ViewGroup viewGroup = findViewById(R.id.preview_display_layout);
  viewGroup.addView(previewDisplayView);
}
```

----------------------------------------

TITLE: Starting MediaPipe Graph After Camera Access in Objective-C
DESCRIPTION: Initiates the MediaPipe graph (`self.mediapipeGraph`) after camera access is granted. It includes error handling for graph startup and ensures the graph is ready to process frames by waiting until it's idle before starting the camera source.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_28

LANGUAGE: Objective-C
CODE:
```
[_cameraSource requestCameraAccessWithCompletionHandler:^void(BOOL granted) {
  if (granted) {
    // Start running self.mediapipeGraph.
    NSError* error;
    if (![self.mediapipeGraph startWithError:&error]) {
      NSLog(@"Failed to start graph: %@", error);
    }
    else if (![self.mediapipeGraph waitUntilIdleWithError:&error]) {
      NSLog(@"Failed to complete graph initial run: %@", error);
    }

    dispatch_async(_videoQueue, ^{
      [_cameraSource start];
    });
  }
}];
```

----------------------------------------

TITLE: Building and Running MediaPipe AutoFlip Binary (Bash)
DESCRIPTION: This command sequence first builds the `run_autoflip` binary using Bazel, disabling GPU support. Subsequently, it executes the built binary, processing a local video file specified by `input_video_path`, saving the output to `output_video_path`, and allowing the user to define the desired `aspect_ratio` for the cropped video. OpenCV 3 is a prerequisite.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/autoflip/README.md#_snippet_1

LANGUAGE: bash
CODE:
```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  mediapipe/examples/desktop/autoflip:run_autoflip

GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/autoflip/run_autoflip \
  --calculator_graph_config_file=mediapipe/examples/desktop/autoflip/autoflip_graph.pbtxt \
  --input_side_packets=input_video_path=/absolute/path/to/the/local/video/file,output_video_path=/absolute/path/to/save/the/output/video/file,aspect_ratio=width:height
```

----------------------------------------

TITLE: Retrieving Input Stream Timestamp Bound (C++)
DESCRIPTION: This snippet illustrates how to retrieve the current timestamp of a packet or empty packet on an input stream in a MediaPipe calculator. This timestamp effectively indicates the current timestamp bound for that input stream, which can be used for calculations or propagation.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/realtime_streams.md#_snippet_4

LANGUAGE: C++
CODE:
```
Timestamp bound = cc->Inputs().Tag("IN").Value().Timestamp();
```

----------------------------------------

TITLE: Building and Installing MediaPipe Python Package from Source
DESCRIPTION: This command builds and installs the MediaPipe Python package from source using `setup.py`, optionally linking against an existing OpenCV installation. This is typically used for custom modifications or specific OpenCV configurations.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python.md#_snippet_7

LANGUAGE: Bash
CODE:
```
(mp_env)mediapipe$ python3 setup.py install --link-opencv
```

----------------------------------------

TITLE: Building MediaPipe Object Detection with TFLite Model (Desktop)
DESCRIPTION: This command builds the MediaPipe object detection application for desktop using a TFLite model. It compiles the application in optimized mode (`-c opt`) and explicitly disables GPU support (`--define MEDIAPIPE_DISABLE_GPU=1`), targeting the `object_detection_tflite` example.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/object_detection.md#_snippet_0

LANGUAGE: bash
CODE:
```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_detection:object_detection_tflite
```

----------------------------------------

TITLE: Building MediaPipe Object Detection with TensorFlow Model (Desktop)
DESCRIPTION: This command builds the MediaPipe object detection application for desktop, specifically configured for TensorFlow CPU inference. It optimizes the build, disables GPU and AWS support, and strips symbols from the binary, targeting the `object_detection_tensorflow` example.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/object_detection.md#_snippet_2

LANGUAGE: bash
CODE:
```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --define no_aws_support=true --linkopt=-s \
mediapipe/examples/desktop/object_detection:object_detection_tensorflow
```

----------------------------------------

TITLE: Building MediaPipe Android Object Detection Example in Docker (Bash)
DESCRIPTION: This snippet builds an Android MediaPipe example (objectdetectiongpu) within the Docker container, optimized for `android_arm64`. It demonstrates how to compile MediaPipe applications for Android devices after the SDK/NDK setup. The output includes the generated APK files.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_41

LANGUAGE: bash
CODE:
```
root@bca08b91ff63:/mediapipe# bazel build -c opt --config=android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetectiongpu:objectdetectiongpu
```

----------------------------------------

TITLE: Setting up Android SDK/NDK for MediaPipe in Docker (Bash)
DESCRIPTION: This snippet runs the 'mediapipe' Docker image and then executes a script inside the container to set up the Android SDK and NDK. This is a prerequisite for building Android-specific MediaPipe examples within the Docker environment.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_40

LANGUAGE: bash
CODE:
```
$ docker run -it --name mediapipe mediapipe:latest

root@bca08b91ff63:/mediapipe# bash ./setup_android_sdk_and_ndk.sh
```

----------------------------------------

TITLE: Enabling ProcessTimestampBounds for Custom Bound Calculation (C++)
DESCRIPTION: This snippet enables `ProcessTimestampBounds` for a MediaPipe calculator, causing `Calculator::Process` to be invoked not just for arriving packets, but also for each new 'settled timestamp'. This allows the calculator to perform custom timestamp bound calculations and propagation, even when only input timestamp bounds are updated.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/realtime_streams.md#_snippet_6

LANGUAGE: C++
CODE:
```
cc->SetProcessTimestampBounds(true);
```

----------------------------------------

TITLE: Compiling Face Detection for Coral USB with Bazel
DESCRIPTION: This command compiles the MediaPipe face detection example for Coral USB devices using Bazel. It includes flags for optimized compilation, portable Darwin support, GPU disablement, enabling Edge TPU USB support, and linking the `libusb-1.0.so` library, which is a prerequisite for USB device communication.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/coral/README.md#_snippet_0

LANGUAGE: bash
CODE:
```
bazel build \
  --compilation_mode=opt \
  --define darwinn_portable=1 \
  --define MEDIAPIPE_DISABLE_GPU=1 \
  --define MEDIAPIPE_EDGE_TPU=usb \
  --linkopt=-l:libusb-1.0.so \
  mediapipe/examples/coral:face_detection_tpu build
```

----------------------------------------

TITLE: Building Single-stage Objectron for Shoes (Android)
DESCRIPTION: This command builds the MediaPipe Objectron Android example using the single-stage model for 3D object detection of shoes. It targets the ARM64 architecture and uses the '--define shoe_1stage=true' flag to select the single-stage shoe model.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_8

LANGUAGE: bash
CODE:
```
bazel build -c opt --config android_arm64 --define shoe_1stage=true mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d:objectdetection3d
```

----------------------------------------

TITLE: Declaring Dispatch Queue for Video Processing (Objective-C)
DESCRIPTION: Declares a `dispatch_queue_t` instance variable named `_videoQueue` within the `ViewController`'s implementation block. This queue is designated for processing camera frames on a separate thread, preventing UI blocking.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_8

LANGUAGE: Objective-C
CODE:
```
// Process camera frames on this queue.
dispatch_queue_t _videoQueue;
```

----------------------------------------

TITLE: Building and Running MediaPipe YouTube-8M Model Inference Binary
DESCRIPTION: This snippet first builds the MediaPipe `model_inference` binary with GPU disabled and optimized for size. Then, it executes the built binary, passing the graph configuration, input paths for features and video, output video path, and parameters for segment size and overlap. `input_sequence_example_path` requires `features.pb` and `input_video_path` needs an absolute path to the local video.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/youtube8m/README.md#_snippet_12

LANGUAGE: bash
CODE:
```
bazel build -c opt --define='MEDIAPIPE_DISABLE_GPU=1' --linkopt=-s \
  mediapipe/examples/desktop/youtube8m:model_inference

# segment_size is the number of seconds window of frames.
# overlap is the number of seconds adjacent segments share.
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/youtube8m/model_inference \
  --calculator_graph_config_file=mediapipe/graphs/youtube8m/local_video_model_inference.pbtxt \
  --input_side_packets=input_sequence_example_path=/tmp/mediapipe/features.pb,input_video_path=/absolute/path/to/the/local/video/file,output_video_path=/tmp/mediapipe/annotated_video.mp4,segment_size=5,overlap=4
```

----------------------------------------

TITLE: Verifying TensorFlow GPU Device Detection
DESCRIPTION: This output snippet shows the console logs from TensorFlow indicating successful detection and initialization of a CUDA-enabled GPU device (Tesla T4). It confirms that TensorFlow has identified and is ready to use the GPU for computation.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/gpu_support.md#_snippet_10

LANGUAGE: Shell
CODE:
```
I external/org_tensorflow/tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
I external/org_tensorflow/tensorflow/core/common_runtime/gpu/gpu_device.cc:1544] Found device 0 with properties: pciBusID: 0000:00:04.0 name: Tesla T4 computeCapability: 7.5 coreClock: 1.59GHz coreCount: 40 deviceMemorySize: 14.75GiB deviceMemoryBandwidth: 298.08GiB/s
I external/org_tensorflow/tensorflow/core/common_runtime/gpu/gpu_device.cc:1686] Adding visible gpu devices: 0
```

----------------------------------------

TITLE: Creating Packets with MakePacket in MediaPipe C++
DESCRIPTION: This C++ snippet demonstrates the creation of a new MediaPipe packet using `MakePacket<T>()`, which directly constructs and encapsulates data. It also illustrates how to generate a new packet with identical data but a different timestamp using the `At()` method, essential for stream synchronization.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/packets.md#_snippet_0

LANGUAGE: c++
CODE:
```
// Create a packet containing some new data.
Packet p = MakePacket<MyDataClass>("constructor_argument");
// Make a new packet with the same data and a different timestamp.
Packet p2 = p.At(Timestamp::PostStream());
```

----------------------------------------

TITLE: Running MediaPipe Feature Extraction (Bash)
DESCRIPTION: This sequence first builds the MediaPipe feature extraction binary using Bazel, disabling GPU and AWS support for a CPU-only build. Then, it executes the compiled binary, specifying the graph configuration file and input/output side packets for the `MediaSequence` metadata and extracted features.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/youtube_8m.md#_snippet_4

LANGUAGE: bash
CODE:
```
bazel build -c opt --linkopt=-s \
  --define MEDIAPIPE_DISABLE_GPU=1 --define no_aws_support=true \
  mediapipe/examples/desktop/youtube8m:extract_yt8m_features

GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/youtube8m/extract_yt8m_features \
  --calculator_graph_config_file=mediapipe/graphs/youtube8m/feature_extraction.pbtxt \
  --input_side_packets=input_sequence_example=/tmp/mediapipe/metadata.pb  \
  --output_side_packets=output_sequence_example=/tmp/mediapipe/features.pb
```

----------------------------------------

TITLE: Defining Android Layout for Hello World (XML)
DESCRIPTION: This XML snippet defines the main layout for the Android application, `activity_main.xml`. It uses `ConstraintLayout` to display a `TextView` centered on the screen, showing the text 'Hello World!'.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_1

LANGUAGE: XML
CODE:
```
<?xml version="1.0" encoding="utf-8"?>
<android.support.constraint.ConstraintLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent">

  <TextView
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="Hello World!"
    app:layout_constraintBottom_toBottomOf="parent"
    app:layout_constraintLeft_toLeftOf="parent"
    app:layout_constraintRight_toRightOf="parent"
    app:layout_constraintTop_toTopOf="parent" />

</android.support.constraint.ConstraintLayout>
```

----------------------------------------

TITLE: Downloading YouTube-8M PCA and Inception Model Data (Bash)
DESCRIPTION: This snippet creates a temporary directory, changes into it, and downloads necessary PCA matrices (inception3, vggish) and the Inception-2015-12-05 model tarball, then extracts the model data. These files are prerequisites for feature extraction.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/youtube8m/README.md#_snippet_1

LANGUAGE: Bash
CODE:
```
mkdir /tmp/mediapipe
cd /tmp/mediapipe
curl -O http://data.yt8m.org/pca_matrix_data/inception3_mean_matrix_data.pb
curl -O http://data.yt8m.org/pca_matrix_data/inception3_projection_matrix_data.pb
curl -O http://data.yt8m.org/pca_matrix_data/vggish_mean_matrix_data.pb
curl -O http://data.yt8m.org/pca_matrix_data/vggish_projection_matrix_data.pb
curl -O http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
tar -xvf /tmp/mediapipe/inception-2015-12-05.tgz
```

----------------------------------------

TITLE: Downloading and Extracting YouTube-8M Baseline Model (Bash)
DESCRIPTION: This snippet downloads the YouTube-8M baseline model archive from `data.yt8m.org` and extracts its contents to the `/tmp/mediapipe` directory. This model is a prerequisite for performing inference with MediaPipe for both web interface and local video processing.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/youtube_8m.md#_snippet_6

LANGUAGE: bash
CODE:
```
curl -o /tmp/mediapipe/yt8m_baseline_saved_model.tar.gz http://data.yt8m.org/models/baseline/saved_model.tar.gz

tar -xvf /tmp/mediapipe/yt8m_baseline_saved_model.tar.gz -C /tmp/mediapipe
```

----------------------------------------

TITLE: MediaPipe Graph for Image Extraction and SequenceExample Processing
DESCRIPTION: This MediaPipe graph defines a pipeline for processing video data, extracting frames, encoding them, and storing them within a SequenceExample. It details the flow from string input to decoded video, frame sampling, image encoding, and final serialization, allowing for customization of parameters like frame_rate and quality.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/media_sequence.md#_snippet_5

LANGUAGE: MediaPipe Graph
CODE:
```
# Convert the string input into a decoded SequenceExample.
node {
  calculator: "StringToSequenceExampleCalculator"
  input_side_packet: "STRING:input_sequence_example"
  output_side_packet: "SEQUENCE_EXAMPLE:parsed_sequence_example"
}

# Unpack the data path and clip timing from the SequenceExample.
node {
  calculator: "UnpackMediaSequenceCalculator"
  input_side_packet: "SEQUENCE_EXAMPLE:parsed_sequence_example"
  output_side_packet: "DATA_PATH:input_video_path"
  output_side_packet: "RESAMPLER_OPTIONS:packet_resampler_options"
  options {
    [type.googleapis.com/mediapipe.UnpackMediaSequenceCalculatorOptions]: {
      base_packet_resampler_options {
        frame_rate: 24.0
        base_timestamp: 0
      }
    }
  }
}

# Decode the entire video.
node {
  calculator: "OpenCvVideoDecoderCalculator"
  input_side_packet: "INPUT_FILE_PATH:input_video_path"
  output_stream: "VIDEO:decoded_frames"
}

# Extract the subset of frames we want to keep.
node {
  calculator: "PacketResamplerCalculator"
  input_stream: "decoded_frames"
  output_stream: "sampled_frames"
  input_side_packet: "OPTIONS:packet_resampler_options"
}

# Encode the images to store in the SequenceExample.
node {
  calculator: "OpenCvImageEncoderCalculator"
  input_stream: "sampled_frames"
  output_stream: "encoded_frames"
  node_options {
    [type.googleapis.com/mediapipe.OpenCvImageEncoderCalculatorOptions]: {
      quality: 80
    }
  }
}

# Store the images in the SequenceExample.
node {
  calculator: "PackMediaSequenceCalculator"
  input_side_packet: "SEQUENCE_EXAMPLE:parsed_sequence_example"
  output_side_packet: "SEQUENCE_EXAMPLE:sequence_example_to_serialize"
  input_stream: "IMAGE:encoded_frames"
}

# Serialize the SequenceExample to a string for storage.
node {
  calculator: "StringToSequenceExampleCalculator"
  input_side_packet: "SEQUENCE_EXAMPLE:sequence_example_to_serialize"
  output_side_packet: "STRING:output_sequence_example"
}
```

----------------------------------------

TITLE: MediaPipe Project Python Dependencies
DESCRIPTION: This snippet provides the complete list of Python packages and their pinned versions, along with their dependency origins, as specified in the `requirements_lock_3_11.txt` file. This file is crucial for setting up the correct Python environment for the MediaPipe project.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/requirements_lock_3_11.txt#_snippet_0

LANGUAGE: Python Requirements
CODE:
```
#
# This file is autogenerated by pip-compile with Python 3.11
# by the following command:
#
#    pip-compile --output-file=mediapipe/opensource_only/requirements_lock_3_11.txt mediapipe/opensource_only/requirements.txt
#
absl-py==2.1.0
    # via -r mediapipe/opensource_only/requirements.txt
attrs==24.2.0
    # via -r mediapipe/opensource_only/requirements.txt
cffi==1.17.1
    # via sounddevice
contourpy==1.3.0
    # via matplotlib
cycler==0.12.1
    # via matplotlib
flatbuffers==24.3.25
    # via -r mediapipe/opensource_only/requirements.txt
fonttools==4.54.1
    # via matplotlib
jax==0.4.30
    # via -r mediapipe/opensource_only/requirements.txt
jaxlib==0.4.30
    # via
    #   -r mediapipe/opensource_only/requirements.txt
    #   jax
kiwisolver==1.4.7
    # via matplotlib
matplotlib==3.9.2
    # via -r mediapipe/opensource_only/requirements.txt
ml-dtypes==0.5.0
    # via
    #   jax
    #   jaxlib
numpy==1.26.4
    # via
    #   -r mediapipe/opensource_only/requirements.txt
    #   contourpy
    #   jax
    #   jaxlib
    #   matplotlib
    #   ml-dtypes
    #   opencv-contrib-python
    #   scipy
opencv-contrib-python==4.10.0.84
    # via -r mediapipe/opensource_only/requirements.txt
opt-einsum==3.4.0
    # via jax
packaging==24.1
    # via matplotlib
pillow==10.4.0
    # via matplotlib
protobuf==4.25.5
    # via -r mediapipe/opensource_only/requirements.txt
pycparser==2.22
    # via cffi
pyparsing==3.1.4
    # via matplotlib
python-dateutil==2.9.0.post0
    # via matplotlib
scipy==1.13.1
    # via
    #   jax
    #   jaxlib
sentencepiece==0.2.0
    # via -r mediapipe/opensource_only/requirements.txt
six==1.16.0
    # via python-dateutil
sounddevice==0.5.0
    # via -r mediapipe/opensource_only/requirements.txt
```

----------------------------------------

TITLE: Installing Python 3 and Six Library (macOS)
DESCRIPTION: These commands install Python 3 using Homebrew, create a symbolic link to make `python` point to `python3.7`, verify the Python version, and then install the `six` compatibility library using `pip3`. These steps ensure the correct Python environment for MediaPipe.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_26

LANGUAGE: bash
CODE:
```
$ brew install python
$ sudo ln -s -f /usr/local/bin/python3.7 /usr/local/bin/python
$ python --version
Python 3.7.4
$ pip3 install --user six
```

----------------------------------------

TITLE: Building and Running MediaPipe YouTube-8M Local Video Inference (Bash)
DESCRIPTION: This snippet first builds the MediaPipe `model_inference` binary, then executes it to perform inference on a local video. It uses `local_video_model_inference.pbtxt` and specifies input/output paths, along with `segment_size` and `overlap` parameters for video processing. Requires `features.pb` and a local video file.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/youtube_8m.md#_snippet_9

LANGUAGE: bash
CODE:
```
bazel build -c opt --define='MEDIAPIPE_DISABLE_GPU=1' --linkopt=-s \
  mediapipe/examples/desktop/youtube8m:model_inference

# segment_size is the number of seconds window of frames.
# overlap is the number of seconds adjacent segments share.
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/youtube8m/model_inference \
  --calculator_graph_config_file=mediapipe/graphs/youtube8m/local_video_model_inference.pbtxt \
  --input_side_packets=input_sequence_example_path=/tmp/mediapipe/features.pb,input_video_path=/absolute/path/to/the/local/video/file,output_video_path=/tmp/mediapipe/annotated_video.mp4,segment_size=5,overlap=4
```

----------------------------------------

TITLE: Running MediaPipe YouTube-8M Web Server (Python)
DESCRIPTION: This Python command starts a local web server for the YouTube-8M model inference viewer. It serves the web interface from the current working directory, allowing interaction with the MediaPipe pipeline via `localhost:8008`. The `absl-py` library is a required dependency.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/youtube_8m.md#_snippet_8

LANGUAGE: python
CODE:
```
python mediapipe/examples/desktop/youtube8m/viewer/server.py --root `pwd`
```

----------------------------------------

TITLE: Building MediaPipe iOS App with Bazel (Bash)
DESCRIPTION: This Bazel command builds a specific MediaPipe iOS application (e.g., HandTrackingGpuApp) for the ARM64 architecture with optimizations enabled. It compiles the application and prepares it for installation on an iOS device. Users may see a permission request from `codesign` during this process to sign the app.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/ios.md#_snippet_8

LANGUAGE: bash
CODE:
```
bazel build -c opt --config=ios_arm64 mediapipe/examples/ios/handtrackinggpu:HandTrackingGpuApp
```

----------------------------------------

TITLE: Initializing Clip Metadata for Spatiotemporal Detection (Python)
DESCRIPTION: This Python snippet initializes a `SequenceExample` with the video data path and overall clip start and end timestamps. This foundational metadata is a prerequisite for adding detailed frame-level bounding box annotations for spatiotemporal detection or object tracking tasks.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_2

LANGUAGE: python
CODE:
```
# Python: functions from media_sequence.py as ms
sequence = tf.train.SequenceExample()
ms.set_clip_data_path(b"path_to_video", sequence)
ms.set_clip_start_timestamp(1000000, sequence)
ms.set_clip_end_timestamp(6000000, sequence)

```

----------------------------------------

TITLE: Building MediaPipe Objectron Desktop Application (CPU) - Bash
DESCRIPTION: This command compiles the MediaPipe Objectron desktop application for CPU-only execution using Bazel. The `-c opt` flag optimizes the build, and `--define MEDIAPIPE_DISABLE_GPU=1` explicitly disables GPU support, ensuring the application runs on systems without GPU acceleration.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_12

LANGUAGE: bash
CODE:
```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_detection_3d:objectron_cpu
```

----------------------------------------

TITLE: Building MediaPipe Object Detection with TensorFlow GPU
DESCRIPTION: This shell command demonstrates how to build the MediaPipe object detection example using Bazel, enabling TensorFlow GPU support via `--config=cuda` and optimizing the build. It also includes flags to disable AWS support and define `MESA_EGL_NO_X11_HEADERS`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/gpu_support.md#_snippet_9

LANGUAGE: Shell
CODE:
```
$ bazel build -c opt --config=cuda --spawn_strategy=local \
    --define no_aws_support=true --copt -DMESA_EGL_NO_X11_HEADERS \
    mediapipe/examples/desktop/object_detection:object_detection_tensorflow
```

----------------------------------------

TITLE: Linking Automatic Provisioning Profiles (Bash)
DESCRIPTION: This Bash command executes a Python script that finds and links the provisioning profiles generated by Xcode's automatic signing feature for all MediaPipe iOS applications. It ensures that Bazel can correctly use these profiles for building. This script must be re-run if profiles expire and Xcode generates new ones.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/ios.md#_snippet_6

LANGUAGE: bash
CODE:
```
python3 mediapipe/examples/ios/link_local_profiles.py
```

----------------------------------------

TITLE: Defining Bazel cc_library for MacPorts OpenCV
DESCRIPTION: This Bazel `cc_library` rule defines how to link against OpenCV when installed via MacPorts. It specifies the dynamic libraries, header files, include paths, and sets `linkstatic` to 1 for static linking, making the OpenCV library available for other Bazel targets.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_24

LANGUAGE: BUILD
CODE:
```
cc_library(
    name = "opencv",
    srcs = glob(
        [
            "local/lib/libopencv_core.dylib",
            "local/lib/libopencv_highgui.dylib",
            "local/lib/libopencv_imgcodecs.dylib",
            "local/lib/libopencv_imgproc.dylib",
            "local/lib/libopencv_video.dylib",
            "local/lib/libopencv_videoio.dylib",
        ],
    ),
    hdrs = glob(["local/include/opencv2/**/*.h*"]),
    includes = ["local/include/"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
```

----------------------------------------

TITLE: Good Practice: Defining MediaPipe Graph Inputs at Start (C++)
DESCRIPTION: This snippet illustrates the recommended practice of defining all graph inputs explicitly at the very beginning of the `BuildGraph` function. This improves readability, reduces errors by centralizing input declarations, and enhances the reusability of helper functions by passing all necessary streams as parameters.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_9

LANGUAGE: C++
CODE:
```
Stream<D> RunSomething(Stream<A> a, Stream<B> b, Stream<C> c, Graph& graph) {
  // ...
}

CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Inputs.
  Stream<A> a = graph.In(0).SetName("a").Cast<A>();
  Stream<B> b = graph.In(1).SetName("b").Cast<B>();
  Stream<C> c = graph.In(2).SetName("c").Cast<C>();

  // 10/100/N lines of code.
  Stream<D> d = RunSomething(a, b, c, graph);
  // ...

  return graph.GetConfig();
}
```

----------------------------------------

TITLE: Initializing MediaPipe Graph in viewDidLoad in Objective-C
DESCRIPTION: Initializes the `mediapipeGraph` property in `viewDidLoad` by calling the `loadGraphFromResource:` helper function with the predefined `kGraphName`. This sets up the graph instance for subsequent operations.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_25

LANGUAGE: Objective-C
CODE:
```
self.mediapipeGraph = [[self class] loadGraphFromResource:kGraphName];
```

----------------------------------------

TITLE: Initializing ExternalTextureConverter in onResume (Java)
DESCRIPTION: This line initializes the `converter` object within the `onResume` lifecycle method. It creates an `ExternalTextureConverter` instance, passing the OpenGL ES context obtained from `eglManager`, which enables the conversion of `SurfaceTexture` frames to standard OpenGL textures.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_28

LANGUAGE: Java
CODE:
```
converter = new ExternalTextureConverter(eglManager.getContext());
```

----------------------------------------

TITLE: Summarizing TFLite Graph for Inspection (TensorFlow)
DESCRIPTION: This optional command uses the TensorFlow `graph_transforms` tool to inspect the input and output tensors of the exported TFLite graph (`tflite_graph.pb`). It helps verify the model's expected input image size (e.g., 320x320) and the names of the output tensors, such as `raw_outputs/box_encodings` and `raw_outputs/class_predictions`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/object_detection_saved_model.md#_snippet_1

LANGUAGE: bash
CODE:
```
bazel run graph_transforms:summarize_graph -- \
    --in_graph=${PATH_TO_MODEL}/tflite_graph.pb
```

----------------------------------------

TITLE: Running MediaPipe Objectron Desktop Application (CPU) - Bash
DESCRIPTION: This command executes the compiled MediaPipe Objectron application on the CPU. It requires specifying the input and output video paths, the path to the 3D landmark model (e.g., for shoes, chairs, cups, or cameras), and a comma-separated list of allowed object labels via side packets.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_13

LANGUAGE: bash
CODE:
```
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/object_detection_3d/objectron_cpu \n  --calculator_graph_config_file=mediapipe/graphs/object_detection_3d/objectron_desktop_cpu.pbtxt \n  --input_side_packets=input_video_path=<input video path>,output_video_path=<output video path>,box_landmark_model_path=<landmark model path>,allowed_labels=<allowed labels>
```

----------------------------------------

TITLE: Building Single-stage Objectron for Chairs (Android)
DESCRIPTION: This command builds the MediaPipe Objectron Android example using the single-stage model for 3D object detection of chairs. It targets the ARM64 architecture and uses the '--define chair_1stage=true' flag to select the single-stage chair model.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_9

LANGUAGE: bash
CODE:
```
bazel build -c opt --config android_arm64 --define chair_1stage=true mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d:objectdetection3d
```

----------------------------------------

TITLE: Including MediaPipe Binary Graph as Android Asset in Bazel
DESCRIPTION: This Bazel `BUILD` rule configuration adds a compiled MediaPipe graph (`.binarypb` file) as an asset to the Android application. The `mobile_gpu_binary_graph` target, generated from a `.pbtxt` graph definition, is included in the `assets` list, making it accessible at runtime for the MediaPipe framework to load and execute.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_33

LANGUAGE: Bazel
CODE:
```
assets = [
  "//mediapipe/graphs/edge_detection:mobile_gpu_binary_graph",
],
assets_dir = "",
```

----------------------------------------

TITLE: Registering MediaPipe Calculator (C++)
DESCRIPTION: This macro registers the `PacketClonerCalculator` class with the MediaPipe framework, making it discoverable and usable within MediaPipe graphs. It is typically placed in the calculator's .cc file after the class definition.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/calculators.md#_snippet_11

LANGUAGE: C++
CODE:
```
REGISTER_CALCULATOR(PacketClonerCalculator);
```

----------------------------------------

TITLE: Adding MediaPipe Graph Calculator Dependencies (Bazel)
DESCRIPTION: This Bazel `BUILD` file snippet adds a dependency to the calculators used by the `edge_detection` MediaPipe graph. This ensures that all necessary calculator implementations are linked into the application, allowing the graph to execute properly.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_20

LANGUAGE: Bazel
CODE:
```
"//mediapipe/graphs/edge_detection:mobile_calculators",
```

----------------------------------------

TITLE: Building and Running MediaPipe YouTube-8M Feature Extractor (Bash)
DESCRIPTION: This snippet first builds the MediaPipe YouTube-8M feature extraction binary using Bazel, disabling GPU support and AWS. Then, it executes the compiled binary, specifying the graph configuration file and input/output side packets for the metadata and extracted features. This is the core step for extracting features from the prepared video.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/youtube8m/README.md#_snippet_4

LANGUAGE: Bash
CODE:
```
bazel build -c opt --linkopt=-s \
  --define MEDIAPIPE_DISABLE_GPU=1 --define no_aws_support=true \
  mediapipe/examples/desktop/youtube8m:extract_yt8m_features

GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/youtube8m/extract_yt8m_features \
  --calculator_graph_config_file=mediapipe/graphs/youtube8m/feature_extraction.pbtxt \
  --input_side_packets=input_sequence_example=/tmp/mediapipe/metadata.pb  \
  --output_side_packets=output_sequence_example=/tmp/mediapipe/features.pb
```

----------------------------------------

TITLE: Building and Running MediaPipe Hello World C++ Example on Windows
DESCRIPTION: These commands demonstrate how to build and execute the MediaPipe 'Hello World!' C++ example using Bazel on Windows. The `bazel build` command compiles the project, disabling GPU support and explicitly setting the Python binary path. The `set GLOG_logtostderr=1` command configures logging, and the final command executes the compiled binary, which should print 'Hello World!' multiple times to the console.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_32

LANGUAGE: Shell
CODE:
```
C:\Users\Username\mediapipe_repo>bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="C://python_36//python.exe" mediapipe/examples/desktop/hello_world

C:\Users\Username\mediapipe_repo>set GLOG_logtostderr=1

C:\Users\Username\mediapipe_repo>bazel-bin\mediapipe\examples\desktop\hello_world\hello_world.exe

# should print:
# I20200514 20:43:12.277598  1200 hello_world.cc:56] Hello World!
# I20200514 20:43:12.278597  1200 hello_world.cc:56] Hello World!
# I20200514 20:43:12.279618  1200 hello_world.cc:56] Hello World!
# I20200514 20:43:12.279618  1200 hello_world.cc:56] Hello World!
# I20200514 20:43:12.279618  1200 hello_world.cc:56] Hello World!
# I20200514 20:43:12.279618  1200 hello_world.cc:56] Hello World!
# I20200514 20:43:12.279618  1200 hello_world.cc:56] Hello World!
# I20200514 20:43:12.279618  1200 hello_world.cc:56] Hello World!
# I20200514 20:43:12.279618  1200 hello_world.cc:56] Hello World!
# I20200514 20:43:12.280613  1200 hello_world.cc:56] Hello World!
```

----------------------------------------

TITLE: Verifying Trace Files on Android Device (ADB Bash)
DESCRIPTION: This `adb shell` command lists the contents of the `/storage/emulated/0/Download` directory on the connected Android device. It's used to confirm that MediaPipe has successfully written the trace log files to the expected location.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/tools/tracing_and_profiling.md#_snippet_4

LANGUAGE: bash
CODE:
```
adb shell "ls -la /storage/emulated/0/Download"
```

----------------------------------------

TITLE: Building MediaPipe Graph with Separated Node Definitions (C++)
DESCRIPTION: This C++ snippet improves graph readability by adding blank lines before and after each AddNode block. This visual separation clearly delineates individual calculator definitions, making it easier to understand the flow and structure of the MediaPipe graph. It defines the same four calculators as the previous example but with enhanced formatting.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_18

LANGUAGE: C++
CODE:
```
CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Inputs.
  Stream<A> a = graph.In(0).Cast<A>();

  auto& node1 = graph.AddNode("Calculator1");
  a.ConnectTo(node1.In("INPUT"));
  Stream<B> b = node1.Out("OUTPUT").Cast<B>();

  auto& node2 = graph.AddNode("Calculator2");
  b.ConnectTo(node2.In("INPUT"));
  Stream<C> c = node2.Out("OUTPUT").Cast<C>();

  auto& node3 = graph.AddNode("Calculator3");
  b.ConnectTo(node3.In("INPUT_B"));
  c.ConnectTo(node3.In("INPUT_C"));
  Stream<D> d = node3.Out("OUTPUT").Cast<D>();

  auto& node4 = graph.AddNode("Calculator4");
  b.ConnectTo(node4.In("INPUT_B"));
  c.ConnectTo(node4.In("INPUT_C"));
  d.ConnectTo(node4.In("INPUT_D"));
  Stream<E> e = node4.Out("OUTPUT").Cast<E>();

  // Outputs.
  b.SetName("b").ConnectTo(graph.Out(0));
  c.SetName("c").ConnectTo(graph.Out(1));
  d.SetName("d").ConnectTo(graph.Out(2));
  e.SetName("e").ConnectTo(graph.Out(3));

  return graph.GetConfig();
}
```

----------------------------------------

TITLE: Importing MPPCameraInputSource Header in Objective-C
DESCRIPTION: Imports the necessary header file for `MPPCameraInputSource` to enable camera access and frame grabbing capabilities within an Objective-C application.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_4

LANGUAGE: Objective-C
CODE:
```
#import "mediapipe/objc/MPPCameraInputSource.h"
```

----------------------------------------

TITLE: Setting MediaPipe VLOG VMODULE Override (Bazel)
DESCRIPTION: This Bazel build command option allows setting `VLOG` levels for specific source files or modules (`--vmodule`) during compilation. It's particularly useful for debugging Android applications where standard `--v` or `--vmodule` flags might not be applicable, enabling granular control over logging verbosity.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/troubleshooting.md#_snippet_12

LANGUAGE: Bazel
CODE:
```
--copt=-DMEDIAPIPE_VLOG_VMODULE=\"*calculator*\"=5"
```

----------------------------------------

TITLE: Downloading PCA and Model Data (Bash)
DESCRIPTION: These commands create a temporary directory, navigate into it, download necessary PCA matrices and Inception v3 model data from Google's servers, and then extract the Inception model archive. This data is crucial for the feature extraction process.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/youtube_8m.md#_snippet_1

LANGUAGE: bash
CODE:
```
mkdir /tmp/mediapipe
cd /tmp/mediapipe
curl -O http://data.yt8m.org/pca_matrix_data/inception3_mean_matrix_data.pb
curl -O http://data.yt8m.org/pca_matrix_data/inception3_projection_matrix_data.pb
curl -O http://data.yt8m.org/pca_matrix_data/vggish_mean_matrix_data.pb
curl -O http://data.yt8m.org/pca_matrix_data/vggish_projection_matrix_data.pb
curl -O http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
tar -xvf /tmp/mediapipe/inception-2015-12-05.tgz
```

----------------------------------------

TITLE: Running MediaPipe Iris Single-Image Depth Estimation Application (Bash)
DESCRIPTION: This command executes the MediaPipe Iris application for single-image depth estimation. It requires specifying the input and output image paths. The GLOG_logtostderr=1 part directs logging output to standard error. Replace <input image path> and <output image path> with actual file paths.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/iris.md#_snippet_3

LANGUAGE: bash
CODE:
```
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_depth_from_image_desktop \
  --input_image_path=<input image path> --output_image_path=<output image path>
```

----------------------------------------

TITLE: Running MediaPipe C++ Hello World Example on WSL
DESCRIPTION: These commands set the GLOG output to stderr and then execute the MediaPipe C++ 'Hello World' example using Bazel. The `MEDIAPIPE_DISABLE_GPU=1` flag is crucial as desktop GPU support is not available in this WSL setup.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_37

LANGUAGE: bash
CODE:
```
export GLOG_logtostderr=1

# Need bazel flag 'MEDIAPIPE_DISABLE_GPU=1' as desktop GPU is currently not supported
bazel run --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/hello_world:hello_world
```

----------------------------------------

TITLE: Downloading MediaPipe Trace Files (ADB Bash)
DESCRIPTION: This `adb pull` command is used to download a specific MediaPipe trace log file from the Android device to the local development machine. This allows for offline analysis of the trace data using tools like the MediaPipe visualizer.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/tools/tracing_and_profiling.md#_snippet_6

LANGUAGE: bash
CODE:
```
adb pull /storage/emulated/0/Download/mediapipe_trace_0.binarypb
```

----------------------------------------

TITLE: Single-Command Docker Compilation for ARM32
DESCRIPTION: This command allows running the entire cross-compilation process for ARM32 within Docker as a single `make` invocation. It specifies the target platform and the internal Docker command to execute, streamlining the build workflow for the `face_detection_tpu` example.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/coral/README.md#_snippet_7

LANGUAGE: bash
CODE:
```
make -C mediapipe/examples/coral \
     PLATFORM=armhf \
     DOCKER_COMMAND="make -C mediapipe/examples/coral BAZEL_TARGET=mediapipe/examples/coral:face_detection_tpu build" \
     docker
```

----------------------------------------

TITLE: Setting Up Docker for ARM64 Cross-Compilation
DESCRIPTION: This command sets up a Docker environment tailored for cross-compiling MediaPipe for ARM64 architectures, suitable for devices like the Coral Dev Board. It ensures that all required cross-compilation tools are available within the Docker container.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/coral/README.md#_snippet_2

LANGUAGE: bash
CODE:
```
make -C mediapipe/examples/coral PLATFORM=arm64 docker
```

----------------------------------------

TITLE: Compiling Coral Face Detection Example for ARMHF
DESCRIPTION: This `make` command compiles the MediaPipe face detection example for ARMHF platforms (e.g., Raspberry Pi) by executing the build process within a Docker container. It encapsulates the cross-compilation steps into a single, convenient command.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/coral/README.md#_snippet_8

LANGUAGE: bash
CODE:
```
make -C mediapipe/examples/coral \
     PLATFORM=armhf \
     DOCKER_COMMAND="make -C mediapipe/examples/coral BAZEL_TARGET=mediapipe/examples/coral:face_detection_tpu build" \
     docker
```

----------------------------------------

TITLE: Setting Output Timestamp Bound with Empty Packet (C++)
DESCRIPTION: This code shows an alternative method to specify the timestamp bound for an output stream in MediaPipe. By adding an empty `Packet()` with a specific timestamp `t` to an output stream, the timestamp bound `t + 1` is implicitly defined, aiding in the scheduling of subsequent calculators.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/realtime_streams.md#_snippet_3

LANGUAGE: C++
CODE:
```
cc->Outputs.Tag("OUT").Add(Packet(), t);
```

----------------------------------------

TITLE: Initializing MPPLayerRenderer in viewDidLoad (Objective-C)
DESCRIPTION: Initializes the `_renderer` object, sets its layer's frame to match the bounds of `_liveView`, adds the renderer's layer as a sublayer to `_liveView`, and configures the frame scaling mode to `MPPFrameScaleModeFillAndCrop`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_14

LANGUAGE: Objective-C
CODE:
```
_renderer = [[MPPLayerRenderer alloc] init];
_renderer.layer.frame = _liveView.layer.bounds;
[_liveView.layer addSublayer:_renderer.layer];
_renderer.frameScaleMode = MPPFrameScaleModeFillAndCrop;
```

----------------------------------------

TITLE: Downloading and Extracting YouTube-8M Baseline Model (Bash)
DESCRIPTION: This snippet downloads the YouTube-8M baseline saved model tarball and extracts its contents into a temporary directory. This model is necessary for performing inference on the YouTube-8M dataset or with a web interface.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/youtube8m/README.md#_snippet_7

LANGUAGE: Bash
CODE:
```
curl -o /tmp/mediapipe/yt8m_baseline_saved_model.tar.gz http://data.yt8m.org/models/baseline/saved_model.tar.gz

tar -xvf /tmp/mediapipe/yt8m_baseline_saved_model.tar.gz -C /tmp/mediapipe
```

----------------------------------------

TITLE: Copying MediaPipe Binaries to Target System
DESCRIPTION: This `scp` command securely copies the entire compiled `mediapipe` folder, including binaries and auxiliary files, from the build machine to a specified remote target system. This step is essential for deploying the MediaPipe applications to the device where they will run.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/coral/README.md#_snippet_10

LANGUAGE: bash
CODE:
```
scp -r mediapipe <user>@<host>:.
```

----------------------------------------

TITLE: Compiling Coral Object Detection Example for ARMHF
DESCRIPTION: This `make` command compiles the MediaPipe object detection example for ARMHF platforms (e.g., Raspberry Pi) by running the build process inside a Docker container. It provides a streamlined way to cross-compile this specific example.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/coral/README.md#_snippet_9

LANGUAGE: bash
CODE:
```
make -C mediapipe/examples/coral \
     PLATFORM=armhf \
     DOCKER_COMMAND="make -C mediapipe/examples/coral BAZEL_TARGET=mediapipe/examples/coral:object_detection_tpu build" \
     docker
```

----------------------------------------

TITLE: Adding Object Tracking Data to MediaSequence in C++
DESCRIPTION: This C++ snippet demonstrates how to initialize a `tensorflow::SequenceExample` and add clip metadata such as data path, start, and end timestamps. It then shows how to populate the sequence with object tracking data, including bounding boxes, timestamps, label indices, label strings, and track strings, similar to the Python example.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_4

LANGUAGE: C++
CODE:
```
tensorflow::SequenceExample sequence;
SetClipDataPath("path_to_video", &sequence);
SetClipStartTimestamp(1000000, &sequence);
SetClipEndTimestamp(6000000, &sequence);

// For an object tracking task with action labels:
std::vector<mediapipe::Location> locations_on_frame_1;
AddBBox(locations_on_frame_1, &sequence);
AddBBoxTimestamp(3000000, &sequence);
AddBBoxLabelIndex({4, 3}, &sequence);
AddBBoxLabelString({"run", "jump"}, &sequence);
AddBBoxTrackString({"id_0", "id_1"}, &sequence);
// AddBBoxClassString({"cls_0", "cls_0"}, &sequence); // if required
std::vector<mediapipe::Location> locations_on_frame_2;
AddBBox(locations_on_frame_2, &sequence);
AddBBoxTimestamp(5000000, &sequence);
AddBBoxLabelIndex({3}, &sequence);
AddBBoxLabelString({"jump"}, &sequence);
AddBBoxTrackString({"id_0"}, &sequence);
// AddBBoxClassString({"cls_0"}, &sequence); // if required
```

----------------------------------------

TITLE: Implementing Luminance Conversion with GlSimpleCalculator in C++
DESCRIPTION: This C++ snippet defines the LuminanceCalculator class, which inherits from GlSimpleCalculator, and provides the implementation for its GlRender method. The GlRender method performs the actual GPU-based image processing to convert RGB images into luminance images using OpenGL, handling vertex and texture data, setting up shaders, and drawing to a texture. It relies on the GlSimpleCalculator base class for managing the OpenGL context and other plumbing.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/gpu.md#_snippet_0

LANGUAGE: C++
CODE:
```
// Converts RGB images into luminance images, still stored in RGB format.
// See GlSimpleCalculator for inputs, outputs and input side packets.
class LuminanceCalculator : public GlSimpleCalculator {
 public:
  absl::Status GlSetup() override;
  absl::Status GlRender(const GlTexture& src,
                        const GlTexture& dst) override;
  absl::Status GlTeardown() override;

 private:
  GLuint program_ = 0;
  GLint frame_;
};
REGISTER_CALCULATOR(LuminanceCalculator);

absl::Status LuminanceCalculator::GlRender(const GlTexture& src,
                                           const GlTexture& dst) {
  static const GLfloat square_vertices[] = {
      -1.0f, -1.0f,  // bottom left
      1.0f,  -1.0f,  // bottom right
      -1.0f, 1.0f,   // top left
      1.0f,  1.0f,   // top right
  };
  static const GLfloat texture_vertices[] = {
      0.0f, 0.0f,  // bottom left
      1.0f, 0.0f,  // bottom right
      0.0f, 1.0f,  // top left
      1.0f, 1.0f,  // top right
  };

  // program
  glUseProgram(program_);
  glUniform1i(frame_, 1);

  // vertex storage
  GLuint vbo[2];
  glGenBuffers(2, vbo);
  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  // vbo 0
  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), square_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(ATTRIB_VERTEX);
  glVertexAttribPointer(ATTRIB_VERTEX, 2, GL_FLOAT, 0, 0, nullptr);

  // vbo 1
  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), texture_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
  glVertexAttribPointer(ATTRIB_TEXTURE_POSITION, 2, GL_FLOAT, 0, 0, nullptr);

  // draw
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  // cleanup
  glDisableVertexAttribArray(ATTRIB_VERTEX);
  glDisableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(2, vbo);

  return absl::OkStatus();
}
```

----------------------------------------

TITLE: Setting Temporal Event Detection Metadata (Python/C++)
DESCRIPTION: These snippets illustrate how to create `SequenceExample` metadata for temporal event detection. They define multiple segments within a video clip by specifying their start and end timestamps, along with corresponding integer indices and string labels for each event.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_1

LANGUAGE: python
CODE:
```
# Python: functions from media_sequence.py as ms
sequence = tf.train.SequenceExample()
ms.set_clip_data_path(b"path_to_video", sequence)
ms.set_clip_start_timestamp(1000000, sequence)
ms.set_clip_end_timestamp(6000000, sequence)

ms.set_segment_start_timestamp((2000000, 4000000), sequence)
ms.set_segment_end_timestamp((3500000, 6000000), sequence)
ms.set_segment_label_index((4, 3), sequence)
ms.set_segment_label_string((b"run", b"jump"), sequence)
```

LANGUAGE: c++
CODE:
```
// C++: functions from media_sequence.h
tensorflow::SequenceExample sequence;
SetClipDataPath("path_to_video", &sequence);
SetClipStartTimestamp(1000000, &sequence);
SetClipEndTimestamp(6000000, &sequence);

SetSegmentStartTimestamp({2000000, 4000000}, &sequence);
SetSegmentEndTimestamp({3500000, 6000000}, &sequence);
SetSegmentLabelIndex({4, 3}, &sequence);
SetSegmentLabelString({"run", "jump"}, &sequence);
```

----------------------------------------

TITLE: Setting Timestamp Offset for Automatic Propagation (C++)
DESCRIPTION: This line of code sets the timestamp offset to 0 for a MediaPipe calculator. This configuration automatically propagates input timestamp bounds to output streams, simplifying real-time graph scheduling by ensuring output bounds are updated even when no packets arrive.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/realtime_streams.md#_snippet_5

LANGUAGE: C++
CODE:
```
cc->SetTimestampOffset(0);
```

----------------------------------------

TITLE: Building Two-stage Objectron for Cameras (Android)
DESCRIPTION: This command builds the MediaPipe Objectron Android example for 3D object detection of cameras using Bazel. It targets the ARM64 architecture and activates the camera model via the '--define camera=true' flag.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_7

LANGUAGE: bash
CODE:
```
bazel build -c opt --config android_arm64 --define camera=true mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d:objectdetection3d
```

----------------------------------------

TITLE: Overriding MediaPipe Trace Log Path (Protobuf Config)
DESCRIPTION: This protobuf configuration snippet demonstrates how to override the default trace log directory for MediaPipe. By setting `trace_log_path` within `profiler_config`, users can specify a custom location for trace files, such as `/sdcard/Download/profiles/`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/tools/tracing_and_profiling.md#_snippet_5

LANGUAGE: protobuf
CODE:
```
profiler_config {
  trace_enabled: true
  enable_profiler: true
  trace_log_path: "/sdcard/Download/profiles/"
}
```

----------------------------------------

TITLE: Building Two-stage Objectron for Shoes (Android)
DESCRIPTION: This command builds the MediaPipe Objectron Android example for 3D object detection of shoes using Bazel. It targets the ARM64 architecture and uses the default two-stage model configuration for shoes.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_4

LANGUAGE: bash
CODE:
```
bazel build -c opt --config android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d:objectdetection3d
```

----------------------------------------

TITLE: Defining Local FFmpeg Repository in Bazel WORKSPACE
DESCRIPTION: This Bazel `new_local_repository` rule, intended for the WORKSPACE file, defines a local repository named 'linux_ffmpeg'. It points to the system's `/usr` directory and uses a specific `BUILD` file (`@//third_party:ffmpeg_linux.BUILD`) to configure the FFmpeg libraries for Bazel, enabling their use in MediaPipe projects.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_9

LANGUAGE: Bazel
CODE:
```
new_local_repository(
      name = "linux_ffmpeg",
      build_file = "@//third_party:ffmpeg_linux.BUILD",
      path = "/usr"
    )
```

----------------------------------------

TITLE: Building Index File Generator Executable
DESCRIPTION: This Bazel command builds the `template_matching_tflite` executable, which is used to generate the KNIFT index file from a collection of template images. The `--define MEDIAPIPE_DISABLE_GPU=1` flag ensures the build is optimized for CPU-only operation.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/knift.md#_snippet_1

LANGUAGE: bash
CODE:
```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
mediapipe/examples/desktop/template_matching:template_matching_tflite
```

----------------------------------------

TITLE: Monitoring Calculator Input Packets with DebugInputStreamHandler
DESCRIPTION: These log messages illustrate the real-time tracking of incoming packets to a MediaPipe calculator using `DebugInputStreamHandler`. They show the timestamp and type of the added packet, along with the current state (number of packets and minimum timestamp) of all input queues for the calculator, aiding in understanding data arrival and buffering.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/troubleshooting.md#_snippet_3

LANGUAGE: Log Output
CODE:
```
[INFO] SomeCalculator: Adding packet (ts:2, type:int) to stream INPUT_B:0:input_b
[INFO] SomeCalculator: INPUT_A:0:input_a num_packets: 0 min_ts: 2
[INFO] SomeCalculator: INPUT_B:0:input_b num_packets: 1 min_ts: 2
```

----------------------------------------

TITLE: Adding MediaPipe Calculator Dependencies to JNI Library
DESCRIPTION: This snippet adds a dependency to specific MediaPipe calculator code, such as `mobile_calculators` for edge detection, within the `libmediapipe_jni.so` build rule. This ensures that the necessary MediaPipe calculators are included in the JNI shared library, making them available for use by the MediaPipe graph.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_32

LANGUAGE: Bazel
CODE:
```
"//mediapipe/graphs/edge_detection:mobile_calculators",
```

----------------------------------------

TITLE: Installing libusb for Coral USB Accelerator
DESCRIPTION: This `apt-get` command installs the `libusb-1.0-0` library on a Debian/Ubuntu-based target system. This library is crucial for enabling communication with Coral USB accelerators connected to the system, allowing MediaPipe applications to utilize the Edge TPU.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/coral/README.md#_snippet_12

LANGUAGE: bash
CODE:
```
sudo apt-get install -y \
   libusb-1.0-0
```

----------------------------------------

TITLE: Retrieving a 16-bit Unsigned Integer from a MediaPipe Packet in Python
DESCRIPTION: Retrieves the 16-bit unsigned integer payload (mapped to C++ `uint16_t`) from a MediaPipe packet using `mp.packet_getter.get_uint`. This method provides access to the unsigned integer content.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_15

LANGUAGE: Python
CODE:
```
get_uint(packet)
```

----------------------------------------

TITLE: Initializing EglManager in onCreate (Java)
DESCRIPTION: This code initializes the `eglManager` object within the `onCreate` lifecycle method of `MainActivity`. It creates a new `EglManager` instance, which is responsible for managing the OpenGL ES context required for texture operations.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_27

LANGUAGE: Java
CODE:
```
eglManager = new EglManager(null);
```

----------------------------------------

TITLE: Building MediaPipe Graph with PassThroughCalculator in C++
DESCRIPTION: This C++ function `BuildGraph` programmatically constructs a MediaPipe graph using the `PassThroughCalculator`. It manually defines input and output streams, connects them to the calculator node, and casts stream types, highlighting the verbosity and potential for errors in this direct approach.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_5

LANGUAGE: C++
CODE:
```
CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Graph inputs.
  Stream<float> float_value = graph.In(0).SetName("float_value").Cast<float>();
  Stream<int> int_value = graph.In(1).SetName("int_value").Cast<int>();
  Stream<bool> bool_value = graph.In(2).SetName("bool_value").Cast<bool>();

  auto& pass_node = graph.AddNode("PassThroughCalculator");
  float_value.ConnectTo(pass_node.In("")[0]);
  int_value.ConnectTo(pass_node.In("")[1]);
  bool_value.ConnectTo(pass_node.In("")[2]);
  Stream<float> passed_float_value = pass_node.Out("")[0].Cast<float>();
  Stream<int> passed_int_value = pass_node.Out("")[1].Cast<int>();
  Stream<bool> passed_bool_value = pass_node.Out("")[2].Cast<bool>();

  // Graph outputs.
  passed_float_value.SetName("passed_float_value").ConnectTo(graph.Out(0));
  passed_int_value.SetName("passed_int_value").ConnectTo(graph.Out(1));
  passed_bool_value.SetName("passed_bool_value").ConnectTo(graph.Out(2));

  // Get `CalculatorGraphConfig` to pass it into `CalculatorGraph`
  return graph.GetConfig();
}
```

----------------------------------------

TITLE: Cropping Scenes with SceneCroppingCalculator (MediaPipe Graph Configuration)
DESCRIPTION: This node determines how to crop each frame using the `SceneCroppingCalculator`. It takes various inputs like aspect ratio, raw video, key frames, detection features, static features, and shot boundaries to produce `cropped_frames`, with options for scene size, key frame cropping, camera motion analysis, and padding.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/autoflip.md#_snippet_6

LANGUAGE: MediaPipe Graph Configuration
CODE:
```
node {
  calculator: "SceneCroppingCalculator"
  input_side_packet: "EXTERNAL_ASPECT_RATIO:aspect_ratio"
  input_stream: "VIDEO_FRAMES:video_raw"
  input_stream: "KEY_FRAMES:video_frames_scaled_downsampled"
  input_stream: "DETECTION_FEATURES:salient_regions"
  input_stream: "STATIC_FEATURES:borders"
  input_stream: "SHOT_BOUNDARIES:shot_change"
  output_stream: "CROPPED_FRAMES:cropped_frames"
  node_options: {
    [type.googleapis.com/mediapipe.autoflip.SceneCroppingCalculatorOptions]: {
      max_scene_size: 600
      key_frame_crop_options: {
        score_aggregation_type: CONSTANT
      }
      scene_camera_motion_analyzer_options: {
        motion_stabilization_threshold_percent: 0.5
        salient_point_bound: 0.499
      }
      padding_parameters: {
        blur_cv_size: 200
        overlay_opacity: 0.6
      }
      target_size_type: MAXIMIZE_TARGET_DIMENSION
    }
  }
}
```

----------------------------------------

TITLE: Configuring FFmpeg C++ Library in Bazel BUILD
DESCRIPTION: This Bazel `cc_library` rule, defined within `ffmpeg_linux.BUILD`, creates a library named 'libffmpeg'. It specifies the linker options to include the core FFmpeg shared libraries (`libavcodec.so`, `libavformat.so`, `libavutil.so`), making them available for C++ projects that depend on FFmpeg functionalities.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_10

LANGUAGE: Bazel
CODE:
```
cc_library(
      name = "libffmpeg",
      linkopts = [
        "-l:libavcodec.so",
        "-l:libavformat.so",
        "-l:libavutil.so"
      ]
    )
```

----------------------------------------

TITLE: Default MediaPipe Trace Log File Paths (Android)
DESCRIPTION: These paths indicate the default locations on an Android device where MediaPipe trace log files are stored. Trace events are appended to these `.binarypb` files, with writing shifting to a successive file every 5 seconds to preserve recent events.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/tools/tracing_and_profiling.md#_snippet_3

LANGUAGE: bash
CODE:
```
/storage/emulated/0/Download/mediapipe_trace_0.binarypb
/storage/emulated/0/Download/mediapipe_trace_1.binarypb
```

----------------------------------------

TITLE: Configuring a Cyclic MediaPipe Graph with Back Edge and Early Close Handler
DESCRIPTION: This Protobuf configuration defines a cyclic MediaPipe graph. It demonstrates annotating the `old_sum` input stream of the `IntAdderCalculator` as a `back_edge` to enable cycles. It also configures the `IntAdderCalculator` to use `EarlyCloseInputStreamHandler` for early termination when its primary input stream is done.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/graphs.md#_snippet_9

LANGUAGE: Protobuf
CODE:
```
node {
  calculator: 'GlobalCountSourceCalculator'
  input_side_packet: 'global_counter'
  output_stream: 'integers'
}
node {
  calculator: 'IntAdderCalculator'
  input_stream: 'integers'
  input_stream: 'old_sum'
  input_stream_info: {
    tag_index: ':1'  # 'old_sum'
    back_edge: true
  }
  output_stream: 'sum'
  input_stream_handler {
    input_stream_handler: 'EarlyCloseInputStreamHandler'
  }
}
node {
  calculator: 'UnitDelayCalculator'
  input_stream: 'sum'
  output_stream: 'old_sum'
}
```

----------------------------------------

TITLE: Implementing Main Activity for Android App (Java)
DESCRIPTION: This Java code defines `MainActivity`, the entry point for the Android application. It extends `AppCompatActivity` and loads the `activity_main.xml` layout when the activity is created, displaying the 'Hello World!' UI.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_2

LANGUAGE: Java
CODE:
```
package com.google.mediapipe.apps.basic;

import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;

/** Bare-bones main activity. */
public class MainActivity extends AppCompatActivity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);
  }
}
```

----------------------------------------

TITLE: Creating a 16-bit Unsigned Integer MediaPipe Packet in Python
DESCRIPTION: Creates a MediaPipe packet with a 16-bit unsigned integer payload (mapped to C++ `uint16_t`) using `mp.packet_creator.create_uint16`. This is suitable for unsigned short integer values.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_14

LANGUAGE: Python
CODE:
```
create_uint16(2**16-1)
```

----------------------------------------

TITLE: Building MediaPipe Iris Single-Image Depth Estimation Application (Bash)
DESCRIPTION: This command builds the MediaPipe Iris application specifically for single-image depth estimation on the CPU. It uses Bazel to compile the C++ example, ensuring GPU is disabled. This build is necessary before running the depth estimation on static images.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/iris.md#_snippet_2

LANGUAGE: bash
CODE:
```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/iris_tracking:iris_depth_from_image_desktop
```

----------------------------------------

TITLE: Creating a Boolean MediaPipe Packet in Python
DESCRIPTION: Creates a MediaPipe packet with a boolean payload using `mp.packet_creator.create_bool`. The payload is immutable after creation, ensuring data integrity within the MediaPipe graph.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_0

LANGUAGE: Python
CODE:
```
create_bool(True)
```

----------------------------------------

TITLE: Creating a 32-bit Unsigned Integer MediaPipe Packet in Python
DESCRIPTION: Creates a MediaPipe packet with a 32-bit unsigned integer payload (mapped to C++ `uint32_t`) using `mp.packet_creator.create_uint32`. This is commonly used for unsigned integer types.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_16

LANGUAGE: Python
CODE:
```
create_uint32(2**32-1)
```

----------------------------------------

TITLE: Running the MediaPipe Command Line Profiler (Custom Columns and Logs) - Bazel/Shell
DESCRIPTION: This command executes the `print_profile` tool with specific options. It demonstrates how to filter the displayed columns to show only those matching `*time*` and `*total*`, and how to specify multiple log files for analysis. The `--` separates Bazel arguments from the program arguments.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/profiler/reporter/README.md#_snippet_2

LANGUAGE: Bazel
CODE:
```
bazel run :print_profile -- --cols "*time*,*total*" --logfiles "<path-to-log>,<path-to-another-log>"
```

----------------------------------------

TITLE: Configuring OpenCV Local Repository for Bazel on Windows
DESCRIPTION: This Bazel `new_local_repository` rule defines how to locate and integrate a locally installed OpenCV build for Windows within the MediaPipe project. It specifies the repository name, the build file (`opencv_windows.BUILD`) that defines how to build OpenCV, and the `path` to the OpenCV build directory. This configuration is essential if OpenCV is not installed in the default location.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_31

LANGUAGE: Bazel
CODE:
```
new_local_repository(
    name = "windows_opencv",
    build_file = "@//third_party:opencv_windows.BUILD",
    path = "C:\\<path to opencv>\\build",
)
```

----------------------------------------

TITLE: Adding Camera Facing Metadata to AndroidManifest (XML)
DESCRIPTION: This XML snippet adds a `meta-data` tag within the `<application>` block of `AndroidManifest.xml`. It defines a placeholder `cameraFacingFront` which allows specifying the default camera (front or back) via build-time configuration, avoiding code changes.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_20

LANGUAGE: XML
CODE:
```
...
      <meta-data android:name="cameraFacingFront" android:value="${cameraFacingFront}"/>
  </application>
</manifest>
```

----------------------------------------

TITLE: Using Custom MediaPipe Type and Packet Methods (Python)
DESCRIPTION: This Python snippet demonstrates how to import and utilize the custom `my_type_binding` and `my_packet_methods` modules. It shows the creation of a custom type instance and its encapsulation within a MediaPipe packet, followed by retrieving the custom type from the packet.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_30

LANGUAGE: Python
CODE:
```
import my_type_binding
import my_packet_methods

packet = my_packet_methods.create_my_type(my_type_binding.MyType())
my_type = my_packet_methods.get_my_type(packet)
```

----------------------------------------

TITLE: Disabling AVXVNNIINT8 Optimization (Bazel)
DESCRIPTION: This Bazel build definition is used to disable support for the `avxvnniint8` compiler optimization in the CPU backend. It is typically added to the `.bazelrc` file and may be necessary when using older Clang versions (e.g., Clang 18 or older) to avoid build issues.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/troubleshooting.md#_snippet_13

LANGUAGE: Bazel
CODE:
```
build --define=xnn_enable_avxvnniint8=false
```

----------------------------------------

TITLE: Adding Object Tracking Data to MediaSequence in Python
DESCRIPTION: This snippet demonstrates how to add bounding box, timestamp, label, and track string data for an object tracking task to a MediaSequence object in Python. It shows adding data for two different frames, including numerical indices and byte string labels, and how to associate them with a sequence.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_3

LANGUAGE: Python
CODE:
```
loctions_on_frame_1 = np.array([[0.1, 0.2, 0.3 0.4],
                                [0.2, 0.3, 0.4, 0.5]])
ms.add_bbox(locations_on_frame_1, sequence)
ms.add_bbox_timestamp(3000000, sequence)
ms.add_bbox_label_index((4, 3), sequence)
ms.add_bbox_label_string((b"run", b"jump"), sequence)
ms.add_bbox_track_string((b"id_0", b"id_1"), sequence)
# ms.add_bbox_class_string(("cls_0", "cls_0"), sequence)  # if required
locations_on_frame_2 = locations_on_frame_1[0]
ms.add_bbox(locations_on_frame_2, sequence)
ms.add_bbox_timestamp(5000000, sequence)
ms.add_bbox_label_index((3), sequence)
ms.add_bbox_label_string((b"jump",), sequence)
ms.add_bbox_track_string((b"id_0",), sequence)
# ms.add_bbox_class_string(("cls_0",), sequence)  # if required
```

----------------------------------------

TITLE: Creating Packets with Adopt in MediaPipe C++
DESCRIPTION: This C++ snippet presents an alternative method for packet creation using `Adopt()`, where pre-existing data is transferred to a new packet, giving the packet ownership. It shows how to create unique data and then associate it with a packet, also applying a specific timestamp with `At()`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/packets.md#_snippet_1

LANGUAGE: c++
CODE:
```
// Create some new data.
auto data = absl::make_unique<MyDataClass>("constructor_argument");
// Create a packet to own the data.
Packet p = Adopt(data.release()).At(Timestamp::PostStream());
```

----------------------------------------

TITLE: Exporting TensorFlow SSD Graph for TFLite Conversion (MediaPipe)
DESCRIPTION: This command exports a frozen TensorFlow graph (`tflite_graph.pb` and `tflite_graph.pbtxt`) from a trained TensorFlow Object Detection API model. It prepares the graph for TFLite conversion by excluding post-processing operations (`add_postprocessing_op=False`), as these are handled by MediaPipe calculators. This step requires the `pipeline.config` and checkpoint files (`model.ckpt`) from the trained model.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/object_detection_saved_model.md#_snippet_0

LANGUAGE: bash
CODE:
```
PATH_TO_MODEL=path/to/the/model
bazel run object_detection:export_tflite_ssd_graph -- \
    --pipeline_config_path ${PATH_TO_MODEL}/pipeline.config \
    --trained_checkpoint_prefix ${PATH_TO_MODEL}/model.ckpt \
    --output_directory ${PATH_TO_MODEL} \
    --add_postprocessing_op=False
```

----------------------------------------

TITLE: Replicating TimestampOffset(0) with ProcessTimestampBounds (C++)
DESCRIPTION: This example demonstrates how to manually replicate the behavior of `SetTimestampOffset(0)` using `SetProcessTimestampBounds(true)` and explicit timestamp bound setting. In `Open`, `ProcessTimestampBounds` is enabled, and in `Process`, the output stream's next timestamp bound is set to match the input's next allowed timestamp, ensuring timely propagation.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/realtime_streams.md#_snippet_7

LANGUAGE: C++
CODE:
```
absl::Status Open(CalculatorContext* cc) {
  cc->SetProcessTimestampBounds(true);
}

absl::Status Process(CalculatorContext* cc) {
  cc->Outputs.Tag("OUT").SetNextTimestampBound(
      cc->InputTimestamp().NextAllowedInStream());
}
```

----------------------------------------

TITLE: Building MediaPipe YouTube-8M Inference Binary for Web (Bash)
DESCRIPTION: This command builds the MediaPipe `model_inference` binary using Bazel, specifically for use with the web interface. It's configured for optimized performance (`-c opt`), disables GPU support (`MEDIAPIPE_DISABLE_GPU=1`), and strips symbols (`-s`) from the output binary.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/youtube_8m.md#_snippet_7

LANGUAGE: bash
CODE:
```
bazel build -c opt --define='MEDIAPIPE_DISABLE_GPU=1' --linkopt=-s \
  mediapipe/examples/desktop/youtube8m:model_inference
```

----------------------------------------

TITLE: Defining Pybind11 Extension Build Rules (Bazel)
DESCRIPTION: These Bazel build rules define two Pybind11 extensions: `my_type_binding` for the custom type and `my_packet_methods` for custom packet operations. They specify source files and dependencies required for building the Python bindings.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_29

LANGUAGE: Bazel
CODE:
```
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
    name = "my_type_binding",
    srcs = ["my_type_binding.cc"],
    deps = [":my_type"],
)

pybind_extension(
    name = "my_packet_methods",
    srcs = ["my_packet_methods.cc"],
    deps = [
        ":my_type",
        "//mediapipe/framework:packet"
    ],
)
```

----------------------------------------

TITLE: Preparing Video Stream for Encoding with VideoPreStreamCalculator (MediaPipe Graph Configuration)
DESCRIPTION: This node prepares the video stream for encoding using `VideoPreStreamCalculator`. It fetches frame format and dimensions from `cropped_frames` and copies frame rate and duration from the original `video_header`, outputting `output_frames_video_header`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/autoflip.md#_snippet_7

LANGUAGE: MediaPipe Graph Configuration
CODE:
```
node {
  calculator: "VideoPreStreamCalculator"
  # Fetch frame format and dimension from input frames.
  input_stream: "FRAME:cropped_frames"
  # Copying frame rate and duration from original video.
  input_stream: "VIDEO_PRESTREAM:video_header"
  output_stream: "output_frames_video_header"
}
```

----------------------------------------

TITLE: Verifying CUDA and CuDNN Installation
DESCRIPTION: These commands verify the successful installation of CUPTI, CUDA, CuDNN, and NVCC by listing key directories and checking the NVCC compiler version. This ensures all necessary components for TensorFlow GPU inference are correctly set up.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/gpu_support.md#_snippet_6

LANGUAGE: bash
CODE:
```
$ ls /usr/local/cuda/extras/CUPTI
/lib64
libcupti.so       libcupti.so.10.1.208  libnvperf_host.so        libnvperf_target.so
libcupti.so.10.1  libcupti_static.a     libnvperf_host_static.a

$ ls /usr/local/cuda-10.1
LICENSE  bin  extras   lib64      libnvvp           nvml  samples  src      tools
README   doc  include  libnsight  nsightee_plugins  nvvm  share    targets  version.txt

$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243

$ ls /usr/lib/x86_64-linux-gnu/ | grep libcudnn.so
libcudnn.so
libcudnn.so.7
libcudnn.so.7.6.4
```

----------------------------------------

TITLE: Installing OpenCV Contrib Library for OpenCV 4.5 - Bash
DESCRIPTION: This command installs the `libopencv-contrib-dev` package, which is specifically required on Debian 11 and Ubuntu 21.04 when OpenCV 4.5 is installed via `libopencv-video-dev`. It provides additional OpenCV modules.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_3

LANGUAGE: Bash
CODE:
```
$ sudo apt-get install -y libopencv-contrib-dev
```

----------------------------------------

TITLE: Monitoring Currently Running MediaPipe Calculators
DESCRIPTION: This log output, part of MediaPipe's graph runtime monitoring, lists the calculators that are currently active and executing their `Calculator::Process` method. It provides an immediate overview of the graph's active components, helping to identify which parts of the graph are currently processing data.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/troubleshooting.md#_snippet_6

LANGUAGE: Log Output
CODE:
```
Running calculators: PacketClonerCalculator, RectTransformationCalculator
```

----------------------------------------

TITLE: Building and Running YouTube-8M Model Inference on Dataset (Bash)
DESCRIPTION: This snippet builds the MediaPipe YouTube-8M model inference binary using Bazel, disabling GPU. It then executes the binary to perform inference on a downloaded YouTube-8M TFRecord dataset, specifying input and output streams for annotations and YT8M IDs.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/youtube8m/README.md#_snippet_8

LANGUAGE: Bash
CODE:
```
bazel build -c opt --define='MEDIAPIPE_DISABLE_GPU=1' --linkopt=-s \
mediapipe/examples/desktop/youtube8m:model_inference

GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/youtube8m/model_inference \
  --calculator_graph_config_file=mediapipe/graphs/youtube8m/yt8m_dataset_model_inference.pbtxt \
  --input_side_packets=tfrecord_path=/tmp/mediapipe/trainpj.tfrecord,record_index=0,desired_segment_size=5 \
  --output_stream=annotation_summary \
  --output_stream_file=/tmp/summary \
  --output_side_packets=yt8m_id \
  --output_side_packets_file=/tmp/yt8m_id
```

----------------------------------------

TITLE: Installing MSVC Runtime for Python on Windows
DESCRIPTION: This command installs the 'msvc-runtime' Python package, which can resolve 'ImportError: DLL load failed' issues on Windows by providing necessary Visual C++ runtime DLLs. It serves as an alternative to installing the official 'vc_redist.x64.exe' from Microsoft.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/troubleshooting.md#_snippet_1

LANGUAGE: bash
CODE:
```
python -m pip install msvc-runtime
```

----------------------------------------

TITLE: Cleaning Up 3D Object Files with obj_cleanup.sh
DESCRIPTION: This shell script is the first step in preprocessing 3D assets for the GlAnimationOverlayCalculator. It takes an input directory of .obj files and an intermediate output directory, performing necessary cleanup before further processing by the ObjParser.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/instant_motion_tracking.md#_snippet_0

LANGUAGE: Shell
CODE:
```
./mediapipe/graphs/object_detection_3d/obj_parser/obj_cleanup.sh [INPUT_DIR] [INTERMEDIATE_OUTPUT_DIR]
```

----------------------------------------

TITLE: Updating Environment Variables for CUDA
DESCRIPTION: These commands update the `PATH` and `LD_LIBRARY_PATH` environment variables to include CUDA binaries and libraries. Running `sudo ldconfig` updates the shared library cache, ensuring the system can find the new CUDA libraries.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/gpu_support.md#_snippet_5

LANGUAGE: bash
CODE:
```
$ export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
$ export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64,/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
$ sudo ldconfig
```

----------------------------------------

TITLE: Compiling MediaSequence Demo Binary with Bazel (Bash)
DESCRIPTION: This command compiles the `media_sequence_demo` C++ binary using Bazel. The `-c opt` flag specifies an optimized build, and `--define MEDIAPIPE_DISABLE_GPU=1` disables GPU support, ensuring the binary is built for CPU-only operation. This binary is crucial for multimedia processing speed within MediaSequence.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/media_sequence.md#_snippet_1

LANGUAGE: bash
CODE:
```
bazel build -c opt mediapipe/examples/desktop/media_sequence:media_sequence_demo --define MEDIAPIPE_DISABLE_GPU=1
```

----------------------------------------

TITLE: Defining PacketClonerCalculator GetContract Method in C++
DESCRIPTION: The `GetContract` method specifies the input and output stream requirements for the `PacketClonerCalculator`. It sets all input streams to 'Any' type and ensures that the output streams have the same type as their corresponding 'base' input streams. The last input stream is designated as the 'tick' signal.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/calculators.md#_snippet_8

LANGUAGE: C++
CODE:
```
  static absl::Status GetContract(CalculatorContract* cc) {
    const int tick_signal_index = cc->Inputs().NumEntries() - 1;
    // cc->Inputs().NumEntries() returns the number of input streams
    // for the PacketClonerCalculator
    for (int i = 0; i < tick_signal_index; ++i) {
      cc->Inputs().Index(i).SetAny();
      // cc->Inputs().Index(i) returns the input stream pointer by index
      cc->Outputs().Index(i).SetSameAs(&cc->Inputs().Index(i));
    }
    cc->Inputs().Index(tick_signal_index).SetAny();
    return absl::OkStatus();
  }
```

----------------------------------------

TITLE: Preparing Dataset with MediaSequence Python Demo (Bash)
DESCRIPTION: This command executes the `demo_dataset` Python module to download and prepare video data. It requires Python 2.7 or 3.5+ with TensorFlow 1.14+ installed. Key parameters define paths for data storage, the compiled MediaPipe binary, and MediaPipe graphs, facilitating the extraction of images and annotations into TFRecords files.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/media_sequence.md#_snippet_2

LANGUAGE: bash
CODE:
```
python -m mediapipe.examples.desktop.media_sequence.demo_dataset \
  --path_to_demo_data=/tmp/demo_data/ \
  --path_to_mediapipe_binary=bazel-bin/mediapipe/examples/desktop/media_sequence/media_sequence_demo \
  --path_to_graph_directory=mediapipe/graphs/media_sequence/
```

----------------------------------------

TITLE: Setting Class Segmentation Label String Mapping - MediaPipe
DESCRIPTION: This snippet demonstrates how to define a mapping from pixel values within the segmentation image to descriptive class label strings, enhancing readability and understanding of the segmentation output.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_63

LANGUAGE: Python
CODE:
```
set_class_segmentation_class_label_string
```

LANGUAGE: C++
CODE:
```
SetClassSegmentationClassLabelString
```

----------------------------------------

TITLE: Installing Core Utilities with Pacman on Windows
DESCRIPTION: This command uses the Pacman package manager, typically found in MSYS2 environments, to install essential development utilities: git for version control, patch for applying diffs, and unzip for archive extraction. These tools are prerequisites for building MediaPipe.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_28

LANGUAGE: Shell
CODE:
```
C:\> pacman -S git patch unzip
```

----------------------------------------

TITLE: Adding MediaPipe to PYTHONPATH (Bash)
DESCRIPTION: This command appends the current working directory (which should be the root of the cloned MediaPipe repository) to the `PYTHONPATH` environment variable. This allows Python to locate and import modules from the MediaPipe repository, specifically `read_demo_dataset.py`, for reading the prepared TensorFlow data.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/media_sequence.md#_snippet_3

LANGUAGE: bash
CODE:
```
PYTHONPATH="${PYTHONPATH};"+`pwd`
```

----------------------------------------

TITLE: Retrieving Custom Type from Packet (C++)
DESCRIPTION: This C++ snippet, likely part of a Pybind11 binding, demonstrates error handling for packet data type mismatches and the successful retrieval of a custom `MyType` object from a MediaPipe packet. It ensures type safety when accessing packet contents.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_28

LANGUAGE: C++
CODE:
```
PyErr_SetString(PyExc_ValueError, "Packet data type mismatch.");
              return py::error_already_set();
            }
            return packet.Get<MyType>();
          });
    }
    }  // namespace mediapipe
```

----------------------------------------

TITLE: Implementing a Unit Delay Calculator in C++
DESCRIPTION: This C++ calculator provides a unit delay for an integer stream. It outputs an initial packet with value 0 at timestamp 0 in its `Open()` method to initialize the loop. In `Process()`, it delays the input packet by one time unit, assuming input timestamps are sequential (0, 1, 2, ...).
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/graphs.md#_snippet_8

LANGUAGE: C++
CODE:
```
class UnitDelayCalculator : public Calculator {
 public:
   static absl::Status FillExpectations(
       const CalculatorOptions& extendable_options, PacketTypeSet* inputs,
       PacketTypeSet* outputs, PacketTypeSet* input_side_packets) {
     inputs->Index(0)->Set<int>("An integer.");
     outputs->Index(0)->Set<int>("The input delayed by one time unit.");
     return absl::OkStatus();
   }

   absl::Status Open() final {
     Output()->Add(new int(0), Timestamp(0));
     return absl::OkStatus();
   }

   absl::Status Process() final {
     const Packet& packet = Input()->Value();
     Output()->AddPacket(packet.At(packet.Timestamp().NextAllowedInStream()));
     return absl::OkStatus();
   }
};
```

----------------------------------------

TITLE: Encoding Cropped Video with OpenCvVideoEncoderCalculator (MediaPipe Graph Configuration)
DESCRIPTION: This node encodes the final cropped video output using the `OpenCvVideoEncoderCalculator`. It takes the `cropped_frames` and `output_frames_video_header` as input streams, along with `output_video_path` and `audio_path` as side packets, configuring the encoder for 'avc1' codec and 'mp4' format.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/autoflip.md#_snippet_8

LANGUAGE: MediaPipe Graph Configuration
CODE:
```
node {
  calculator: "OpenCvVideoEncoderCalculator"
  input_stream: "VIDEO:cropped_frames"
  input_stream: "VIDEO_PRESTREAM:output_frames_video_header"
  input_side_packet: "OUTPUT_FILE_PATH:output_video_path"
  input_side_packet: "AUDIO_FILE_PATH:audio_path"
  node_options: {
    [type.googleapis.com/mediapipe.OpenCvVideoEncoderCalculatorOptions]: {
      codec: "avc1"
      video_format: "mp4"
    }
  }
}
```

----------------------------------------

TITLE: Adopting MPPInputSourceDelegate Protocol in ViewController (Objective-C)
DESCRIPTION: Updates the `ViewController` interface definition to conform to the `MPPInputSourceDelegate` protocol. This enables the `ViewController` to act as a delegate for `MPPInputSource` objects, receiving and processing incoming frames.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_7

LANGUAGE: Objective-C
CODE:
```
@interface ViewController () <MPPInputSourceDelegate>
```

----------------------------------------

TITLE: Defining PacketClonerCalculator Class in C++
DESCRIPTION: This snippet defines the `PacketClonerCalculator` class, inheriting from `CalculatorBase`. It's designed to synchronize multiple input streams by outputting the most recent packets from 'base' streams whenever a packet arrives on a 'tick' stream. It includes an example configuration demonstrating its usage within a MediaPipe graph.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/calculators.md#_snippet_7

LANGUAGE: C++
CODE:
```
// This takes packets from N+1 streams, A_1, A_2, ..., A_N, B.
// For every packet that appears in B, outputs the most recent packet from each
// of the A_i on a separate stream.

#include <vector>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"

namespace mediapipe {

// For every packet received on the last stream, output the latest packet
// obtained on all other streams. Therefore, if the last stream outputs at a
// higher rate than the others, this effectively clones the packets from the
// other streams to match the last.
//
// Example config:
// node {
//   calculator: "PacketClonerCalculator"
//   input_stream: "first_base_signal"
//   input_stream: "second_base_signal"
//   input_stream: "tick_signal"
//   output_stream: "cloned_first_base_signal"
//   output_stream: "cloned_second_base_signal"
// }
//
class PacketClonerCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    const int tick_signal_index = cc->Inputs().NumEntries() - 1;
    // cc->Inputs().NumEntries() returns the number of input streams
    // for the PacketClonerCalculator
    for (int i = 0; i < tick_signal_index; ++i) {
      cc->Inputs().Index(i).SetAny();
      // cc->Inputs().Index(i) returns the input stream pointer by index
      cc->Outputs().Index(i).SetSameAs(&cc->Inputs().Index(i));
    }
    cc->Inputs().Index(tick_signal_index).SetAny();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) final {
    tick_signal_index_ = cc->Inputs().NumEntries() - 1;
    current_.resize(tick_signal_index_);
    // Pass along the header for each stream if present.
    for (int i = 0; i < tick_signal_index_; ++i) {
      if (!cc->Inputs().Index(i).Header().IsEmpty()) {
        cc->Outputs().Index(i).SetHeader(cc->Inputs().Index(i).Header());
        // Sets the output stream of index i header to be the same as
        // the header for the input stream of index i
      }
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    // Store input signals.
    for (int i = 0; i < tick_signal_index_; ++i) {
      if (!cc->Inputs().Index(i).Value().IsEmpty()) {
        current_[i] = cc->Inputs().Index(i).Value();
      }
    }

    // Output if the tick signal is non-empty.
    if (!cc->Inputs().Index(tick_signal_index_).Value().IsEmpty()) {
      for (int i = 0; i < tick_signal_index_; ++i) {
        if (!current_[i].IsEmpty()) {
          cc->Outputs().Index(i).AddPacket(
              current_[i].At(cc->InputTimestamp()));
          // Add a packet to output stream of index i a packet from inputstream i
          // with timestamp common to all present inputs
        } else {
          cc->Outputs().Index(i).SetNextTimestampBound(
              cc->InputTimestamp().NextAllowedInStream());
          // if current_[i], 1 packet buffer for input stream i is empty, we will set
          // next allowed timestamp for input stream i to be current timestamp + 1
        }
      }
    }
    return absl::OkStatus();
  }

 private:
  std::vector<Packet> current_;
  int tick_signal_index_;
};
```

----------------------------------------

TITLE: Building MediaPipe Demo and Processing Charades Dataset
DESCRIPTION: This snippet provides the necessary Bash and Python commands to build the MediaPipe media_sequence_demo binary and then process the Charades dataset. It specifies paths for the dataset, the compiled MediaPipe binary, and the graph directory, noting the significant time required due to the dataset's size.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/media_sequence.md#_snippet_4

LANGUAGE: Bash
CODE:
```
bazel build -c opt mediapipe/examples/desktop/media_sequence:media_sequence_demo --define MEDIAPIPE_DISABLE_GPU=1
```

LANGUAGE: Python
CODE:
```
python -m mediapipe.examples.desktop.media_sequence.charades_dataset \
  --alsologtostderr \
  --path_to_charades_data=/tmp/demo_data/ \
  --path_to_mediapipe_binary=bazel-bin/mediapipe/examples/desktop/media_sequence/media_sequence_demo \
  --path_to_graph_directory=mediapipe/graphs/media_sequence/
```

----------------------------------------

TITLE: Setting Instance Segmentation Class Label Index Mapping - MediaPipe
DESCRIPTION: This snippet illustrates how to define a mapping from pixel values within the instance segmentation image to specific class label indices, crucial for interpreting the segmentation output.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_70

LANGUAGE: Python
CODE:
```
set_instance_segmentation_class_label_index
```

LANGUAGE: C++
CODE:
```
SetInstanceSegmentationClassLabelIndex
```

----------------------------------------

TITLE: Installing OpenCV via MacPorts (macOS)
DESCRIPTION: This command uses MacPorts to install the OpenCV libraries. This is an alternative to Homebrew for users preferring MacPorts as their package manager.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_22

LANGUAGE: bash
CODE:
```
$ port install opencv
```

----------------------------------------

TITLE: Registering Protobuf Descriptors for Graph Options in MediaPipe Bazel
DESCRIPTION: This Bazel BUILD file snippet shows the necessary dependencies for enabling reflection-based graph option handling. It highlights that a mediapipe_simple_subgraph must depend on the <options_proto>_options_lib target (e.g., :face_detection_options_lib) generated by mediapipe_proto_library to correctly process graph options.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/graphs.md#_snippet_7

LANGUAGE: bazel
CODE:
```
mediapipe_proto_library(
    name = "face_detection_proto",
    srcs = ["face_detection.proto"],
    ...
)

mediapipe_simple_subgraph(
    name = "face_detection",
    graph = "face_detection.pbtxt",
    register_as = "FaceDetection",
    deps = [
        ":face_detection_cc_proto",
        ":face_detection_options_lib",      # <-- required
        ...
    ]
)
```

----------------------------------------

TITLE: Transforming 3D Landmarks from Object to Camera Frame - Mathematical
DESCRIPTION: This mathematical formula describes how to transform 3D landmark coordinates from an object's local coordinate frame to the camera's coordinate frame. It involves scaling a unit box by the object's 'scale', applying a 'rotation' matrix, and then adding a 'translation' vector.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_14

LANGUAGE: mathematical
CODE:
```
landmarks_3d = rotation * scale * unit_box + translation
```

----------------------------------------

TITLE: Adding Multiple Encoded Instance Segmentation Masks - MediaPipe
DESCRIPTION: This snippet illustrates how to add multiple encoded segmentation masks for object instances, useful when handling overlapping masks within the same timestep.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_66

LANGUAGE: Python
CODE:
```
add_instance_segmentation_multi_encoded
```

LANGUAGE: C++
CODE:
```
AddInstanceSegmentationEncoded
```

----------------------------------------

TITLE: Downloading and Extracting YouTube-8M Baseline Model
DESCRIPTION: This snippet downloads the YouTube-8M baseline model tarball from `data.yt8m.org` to `/tmp/mediapipe` and then extracts its contents into the same directory. This is a prerequisite for running the model inference.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/youtube8m/README.md#_snippet_11

LANGUAGE: bash
CODE:
```
curl -o /tmp/mediapipe/yt8m_baseline_saved_model.tar.gz http://data.yt8m.org/models/baseline/saved_model.tar.gz

tar -xvf /tmp/mediapipe/yt8m_baseline_saved_model.tar.gz -C /tmp/mediapipe
```

----------------------------------------

TITLE: HTML Structure for MediaPipe Holistic Web App
DESCRIPTION: This HTML snippet sets up the basic page structure for a MediaPipe Holistic web application. It includes necessary MediaPipe utility scripts from CDN, defines a video element for input, and a canvas element for rendering the output.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md#_snippet_2

LANGUAGE: HTML
CODE:
```
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js" crossorigin="anonymous"></script>
</head>

<body>
  <div class="container">
    <video class="input_video"></video>
    <canvas class="output_canvas" width="1280px" height="720px"></canvas>
  </div>
</body>
</html>
```

----------------------------------------

TITLE: HTML Setup for MediaPipe Objectron Web Application
DESCRIPTION: This HTML snippet provides the basic structure for a web application using MediaPipe Objectron. It includes necessary script imports for MediaPipe utility libraries and defines video and canvas elements for input and output display.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_2

LANGUAGE: HTML
CODE:
```
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/control_utils_3d/control_utils_3d.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/objectron/objectron.js" crossorigin="anonymous"></script>
</head>

<body>
  <div class="container">
    <video class="input_video"></video>
    <canvas class="output_canvas" width="1280px" height="720px"></canvas>
  </div>
</body>
</html>
```

----------------------------------------

TITLE: Setting Class Segmentation Label Index Mapping - MediaPipe
DESCRIPTION: This snippet illustrates how to define a mapping from pixel values within the segmentation image to specific class label indices, crucial for interpreting the segmentation output.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_62

LANGUAGE: Python
CODE:
```
set_class_segmentation_class_label_index
```

LANGUAGE: C++
CODE:
```
SetClassSegmentationClassLabelIndex
```

----------------------------------------

TITLE: Installing Python 'six' Library (Bash)
DESCRIPTION: This command installs the 'six' Python library using pip3. The 'six' library provides compatibility utilities between Python 2 and Python 3, and is a required dependency for TensorFlow, which MediaPipe relies on.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/ios.md#_snippet_2

LANGUAGE: bash
CODE:
```
pip3 install --user six
```

----------------------------------------

TITLE: Switching OpenCV Versions and Building Android App
DESCRIPTION: This set of commands temporarily switches the MediaPipe build configuration from OpenCV 3 to OpenCV 4 to resolve compatibility issues with NDK 17+ and `knnMatch`. After building and installing the Android template matching application, it switches back to OpenCV 3.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/knift.md#_snippet_0

LANGUAGE: bash
CODE:
```
# Switch to OpenCV 4
sed -i -e 's:3.4.3/opencv-3.4.3:4.0.1/opencv-4.0.1:g' WORKSPACE
sed -i -e 's:libopencv_java3:libopencv_java4:g' third_party/opencv_android.BUILD

# Build and install app
bazel build -c opt --config=android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/templatematchingcpu:templatematchingcpu
adb install -r bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/templatematchingcpu/templatematchingcpu.apk

# Switch back to OpenCV 3
sed -i -e 's:4.0.1/opencv-4.0.1:3.4.3/opencv-3.4.3:g' WORKSPACE
sed -i -e 's:libopencv_java4:libopencv_java3:g' third_party/opencv_android.BUILD
```

----------------------------------------

TITLE: Retrieving a 64-bit Unsigned Integer from a MediaPipe Packet in Python
DESCRIPTION: Retrieves the 64-bit unsigned integer payload (mapped to C++ `uint64_t`) from a MediaPipe packet using `mp.packet_getter.get_uint`. This method supports large unsigned integer retrieval.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_19

LANGUAGE: Python
CODE:
```
get_uint(packet)
```

----------------------------------------

TITLE: Configuring SceneCroppingCalculator for Visualization in MediaPipe
DESCRIPTION: This configuration snippet modifies the 'SceneCroppingCalculator' node within a MediaPipe graph to enable two additional output streams: 'KEY_FRAME_CROP_REGION_VIZ_FRAMES' and 'SALIENT_POINT_FRAME_VIZ_FRAMES'. These streams provide visual debugging information, showing the cropping window and detected salient points per frame. It also sets various options for scene cropping, camera motion analysis, and padding.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/autoflip.md#_snippet_9

LANGUAGE: bash
CODE:
```
node {
  calculator: "SceneCroppingCalculator"
  input_side_packet: "EXTERNAL_ASPECT_RATIO:aspect_ratio"
  input_stream: "VIDEO_FRAMES:video_raw"
  input_stream: "KEY_FRAMES:video_frames_scaled_downsampled"
  input_stream: "DETECTION_FEATURES:salient_regions"
  input_stream: "STATIC_FEATURES:borders"
  input_stream: "SHOT_BOUNDARIES:shot_change"
  output_stream: "CROPPED_FRAMES:cropped_frames"
  output_stream: "KEY_FRAME_CROP_REGION_VIZ_FRAMES:key_frame_crop_viz_frames"
  output_stream: "SALIENT_POINT_FRAME_VIZ_FRAMES:salient_point_viz_frames"
  node_options: {
    [type.googleapis.com/mediapipe.autoflip.SceneCroppingCalculatorOptions]: {
      max_scene_size: 600
      key_frame_crop_options: {
        score_aggregation_type: CONSTANT
      }
      scene_camera_motion_analyzer_options: {
        motion_stabilization_threshold_percent: 0.5
        salient_point_bound: 0.499
      }
      padding_parameters: {
        blur_cv_size: 200
        overlay_opacity: 0.6
      }
      target_size_type: MAXIMIZE_TARGET_DIMENSION
    }
  }
}
```

----------------------------------------

TITLE: Detecting Objects with AutoFlipObjectDetectionSubgraph (MediaPipe Graph Configuration)
DESCRIPTION: This node uses the `AutoFlipObjectDetectionSubgraph` calculator to find objects on a downsampled video stream. It takes `video_frames_scaled_downsampled` as input and outputs `object_detections`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/autoflip.md#_snippet_3

LANGUAGE: MediaPipe Graph Configuration
CODE:
```
node {
  calculator: "AutoFlipObjectDetectionSubgraph"
  input_stream: "VIDEO:video_frames_scaled_downsampled"
  output_stream: "DETECTIONS:object_detections"
}
```

----------------------------------------

TITLE: Generating MediaSequence Metadata (Bash)
DESCRIPTION: This command runs a MediaPipe Python script to generate a `MediaSequence` metadata file (`metadata.pb`) from an input video. The `clip_end_time_sec` parameter should be adjusted to match the video's length, providing essential input for the feature extraction graph.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/youtube_8m.md#_snippet_3

LANGUAGE: bash
CODE:
```
python -m mediapipe.examples.desktop.youtube8m.generate_input_sequence_example \
  --path_to_input_video=/absolute/path/to/the/local/video/file \
  --clip_end_time_sec=120
```

----------------------------------------

TITLE: Coupled MediaPipe Graph Construction (Bad Practice) - C++
DESCRIPTION: This C++ code demonstrates a poorly structured MediaPipe graph where nodes are tightly coupled, leading to difficulties in refactoring and maintenance. It shows direct connections between nodes using `node.Out("OUTPUT").ConnectTo(nodeX.In("INPUT"))`, which duplicates output calls and obscures data flow, deviating from the decoupled nature of proto representations.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_13

LANGUAGE: C++
CODE:
```
CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Inputs.
  Stream<A> a = graph.In(0).Cast<A>();

  auto& node1 = graph.AddNode("Calculator1");
  a.ConnectTo(node1.In("INPUT"));

  auto& node2 = graph.AddNode("Calculator2");
  node1.Out("OUTPUT").ConnectTo(node2.In("INPUT"));  // Bad.

  auto& node3 = graph.AddNode("Calculator3");
  node1.Out("OUTPUT").ConnectTo(node3.In("INPUT_B"));  // Bad.
  node2.Out("OUTPUT").ConnectTo(node3.In("INPUT_C"));  // Bad.

  auto& node4 = graph.AddNode("Calculator4");
  node1.Out("OUTPUT").ConnectTo(node4.In("INPUT_B"));  // Bad.
  node2.Out("OUTPUT").ConnectTo(node4.In("INPUT_C"));  // Bad.
  node3.Out("OUTPUT").ConnectTo(node4.In("INPUT_D"));  // Bad.

  // Outputs.
  node1.Out("OUTPUT").SetName("b").ConnectTo(graph.Out(0));  // Bad.
  node2.Out("OUTPUT").SetName("c").ConnectTo(graph.Out(1));  // Bad.
  node3.Out("OUTPUT").SetName("d").ConnectTo(graph.Out(2));  // Bad.
  node4.Out("OUTPUT").SetName("e").ConnectTo(graph.Out(3));  // Bad.

  return graph.GetConfig();
}
```

----------------------------------------

TITLE: Building MediaPipe Graph with Monolithic Node Definitions (C++)
DESCRIPTION: This C++ snippet demonstrates a less readable way to construct a MediaPipe CalculatorGraphConfig. All calculator nodes are defined sequentially without visual separation, making it difficult to discern individual node boundaries and their respective inputs/outputs. It defines four calculators (Calculator1 through Calculator4) and connects their streams, finally exposing b, c, d, and e as graph outputs.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_17

LANGUAGE: C++
CODE:
```
CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Inputs.
  Stream<A> a = graph.In(0).Cast<A>();
  auto& node1 = graph.AddNode("Calculator1");
  a.ConnectTo(node1.In("INPUT"));
  Stream<B> b = node1.Out("OUTPUT").Cast<B>();
  auto& node2 = graph.AddNode("Calculator2");
  b.ConnectTo(node2.In("INPUT"));
  Stream<C> c = node2.Out("OUTPUT").Cast<C>();
  auto& node3 = graph.AddNode("Calculator3");
  b.ConnectTo(node3.In("INPUT_B"));
  c.ConnectTo(node3.In("INPUT_C"));
  Stream<D> d = node3.Out("OUTPUT").Cast<D>();
  auto& node4 = graph.AddNode("Calculator4");
  b.ConnectTo(node4.In("INPUT_B"));
  c.ConnectTo(node4.In("INPUT_C"));
  d.ConnectTo(node4.In("INPUT_D"));
  Stream<E> e = node4.Out("OUTPUT").Cast<E>();
  // Outputs.
  b.SetName("b").ConnectTo(graph.Out(0));
  c.SetName("c").ConnectTo(graph.Out(1));
  d.SetName("d").ConnectTo(graph.Out(2));
  e.SetName("e").ConnectTo(graph.Out(3));

  return graph.GetConfig();
}
```

----------------------------------------

TITLE: Building MediaPipe with OpenGL ES 3.0 or Below Support
DESCRIPTION: This Bazel command is used to build MediaPipe targets when the GPU only supports OpenGL ES 3.0 or below. The `--copt -DMEDIAPIPE_DISABLE_GL_COMPUTE` flag disables GL compute functionality, which is not supported on older ES versions.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/gpu_support.md#_snippet_4

LANGUAGE: bash
CODE:
```
$ bazel build --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 --copt -DMEDIAPIPE_DISABLE_GL_COMPUTE <my-target>
```

----------------------------------------

TITLE: Adding Encoded Class Segmentation Image - MediaPipe
DESCRIPTION: This snippet demonstrates how to add an encoded image of class labels at a specific timestep using MediaPipe's API. It's used for storing the raw segmentation mask data, representing the class labels for each pixel.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_56

LANGUAGE: Python
CODE:
```
add_class_segmentation_encoded
```

LANGUAGE: C++
CODE:
```
AddClassSegmentationEncoded
```

----------------------------------------

TITLE: Adding Class Segmentation Timestamp - MediaPipe
DESCRIPTION: This snippet shows how to associate a timestamp in microseconds with the class labels, ensuring proper temporal synchronization of segmentation data.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_57

LANGUAGE: Python
CODE:
```
add_class_segmentation_timestamp
```

LANGUAGE: C++
CODE:
```
AddClassSegmentationTimestamp
```

----------------------------------------

TITLE: Generating MediaSequence Metadata from Video (Bash)
DESCRIPTION: This snippet runs a MediaPipe Python script to generate a `MediaSequence` metadata file (`metadata.pb`) from a local video file. This metadata is crucial for defining the input video's properties, including its path and clip duration, for subsequent feature extraction.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/youtube8m/README.md#_snippet_3

LANGUAGE: Bash
CODE:
```
# change clip_end_time_sec to match the length of your video.
python -m mediapipe.examples.desktop.youtube8m.generate_input_sequence_example \
  --path_to_input_video=/absolute/path/to/the/local/video/file \
  --clip_end_time_sec=120
```

----------------------------------------

TITLE: Retrieving Single Feature from MediaPipe Singular Feature List (Python/C++)
DESCRIPTION: This function retrieves a single feature from a MediaPipe singular feature list at a specified `index`. The returned feature is of the appropriate type (string, int64, or float). An optional `prefix` can be used to scope the feature retrieval.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_23

LANGUAGE: python
CODE:
```
get_feature_at(index, example [, prefix])
```

LANGUAGE: c++
CODE:
```
GetFeatureAt([const string& prefix,] const tf::SE& example, const int index)
```

----------------------------------------

TITLE: Adding Instance Segmentation Timestamp - MediaPipe
DESCRIPTION: This snippet shows how to associate a timestamp in microseconds with the object instance labels, ensuring proper temporal synchronization of instance segmentation data.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_65

LANGUAGE: Python
CODE:
```
add_instance_segmentation_timestamp
```

LANGUAGE: C++
CODE:
```
AddInstanceSegmentationTimestamp
```

----------------------------------------

TITLE: Getting Default Parser for MediaPipe Singular Feature Lists (Python)
DESCRIPTION: This Python-only function returns the `tf.io.FixedLenSequenceFeature` parser, which is suitable for handling fixed-length sequence features. It is used for parsing feature data during TensorFlow operations.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_25

LANGUAGE: python
CODE:
```
get_feature_default_parser()
```

----------------------------------------

TITLE: Running Face Detection with MediaPipe on Coral TPU (Shell)
DESCRIPTION: This command executes the MediaPipe face detection model on a desktop with a Coral TPU. It uses the `face_detection_desktop_live.pbtxt` calculator graph configuration file to define the processing pipeline. The `GLOG_logtostderr=1` prefix ensures that logs are output to standard error.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/coral/README.md#_snippet_13

LANGUAGE: Shell
CODE:
```
GLOG_logtostderr=1 ./face_detection_tpu --calculator_graph_config_file \n    mediapipe/examples/coral/graphs/face_detection_desktop_live.pbtxt
```

----------------------------------------

TITLE: Retrieving a 32-bit Signed Integer from a MediaPipe Packet in Python
DESCRIPTION: Retrieves the 32-bit signed integer payload (mapped to C++ `int32_t`) from a MediaPipe packet using `mp.packet_getter.get_int`. This allows extraction of the integer value.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_9

LANGUAGE: Python
CODE:
```
get_int(packet)
```

----------------------------------------

TITLE: Installing OpenGL ES Utilities on Linux
DESCRIPTION: These commands install necessary Mesa development packages and utilities on a Linux desktop to check for OpenGL ES capabilities. `glxinfo` is used to probe the GPU for OpenGL ES version information.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/gpu_support.md#_snippet_1

LANGUAGE: bash
CODE:
```
$ sudo apt-get install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev
$ sudo apt-get install mesa-utils
$ glxinfo | grep -i opengl
```

----------------------------------------

TITLE: Adding Encoded Instance Segmentation Image - MediaPipe
DESCRIPTION: This snippet demonstrates how to add an encoded image of object instance labels at a specific timestep using MediaPipe's API. It's used for storing raw instance segmentation mask data.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_64

LANGUAGE: Python
CODE:
```
add_instance_segmentation_encoded
```

LANGUAGE: C++
CODE:
```
AddInstanceSegmentationEncoded
```

----------------------------------------

TITLE: Adding Bounding Box Y-Min (Python/C++)
DESCRIPTION: Adds a list of normalized minimum y values for bounding boxes in a frame. This function is used to specify the top edge of a bounding box.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_33

LANGUAGE: Python
CODE:
```
add_bbox_ymin
```

LANGUAGE: C++
CODE:
```
AddBBoxYMin
```

----------------------------------------

TITLE: Checking Feature Presence in MediaPipe Context Features (Python/C++)
DESCRIPTION: This function checks if a specific feature is present within a MediaPipe context example. It returns a boolean value indicating its presence. An optional `prefix` can be provided to specify a namespace for the feature.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_16

LANGUAGE: python
CODE:
```
has_feature(example [, prefix])
```

LANGUAGE: c++
CODE:
```
HasFeature([const string& prefix,] const tf::SE& example)
```

----------------------------------------

TITLE: Running Index File Generation for Custom Templates
DESCRIPTION: This command executes the `template_matching_tflite` program to process a directory of template images and generate a KNIFT index file. It requires specifying the input directory, file suffix (e.g., 'png'), and the desired output filename for the index.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/knift.md#_snippet_2

LANGUAGE: bash
CODE:
```
bazel-bin/mediapipe/examples/desktop/template_matching/template_matching_tflite \
--calculator_graph_config_file=mediapipe/graphs/template_matching/index_building.pbtxt \
--input_side_packets="file_directory=<template image directory>,file_suffix=png,output_index_filename=<output index filename>"
```

----------------------------------------

TITLE: Adding Full Point (Python/C++)
DESCRIPTION: A special accessor that operates on point/x and point/y with a single call. This provides a convenient way to add both coordinates of a 2D point simultaneously.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_40

LANGUAGE: Python
CODE:
```
add_bbox_point
```

LANGUAGE: C++
CODE:
```
AddBBoxPoint
```

----------------------------------------

TITLE: Reading Extracted Features in Python
DESCRIPTION: This Python snippet demonstrates how to load and parse the `features.pb` file, which contains the extracted video and audio features, into a TensorFlow `SequenceExample` object for further processing or analysis.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/youtube_8m.md#_snippet_5

LANGUAGE: python
CODE:
```
import tensorflow as tf

sequence_example = open('/tmp/mediapipe/features.pb', 'rb').read()
print(tf.train.SequenceExample.FromString(sequence_example))
```

----------------------------------------

TITLE: Retrieving a 32-bit Unsigned Integer from a MediaPipe Packet in Python
DESCRIPTION: Retrieves the 32-bit unsigned integer payload (mapped to C++ `uint32_t`) from a MediaPipe packet using `mp.packet_getter.get_uint`. This allows extraction of the unsigned integer value.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_17

LANGUAGE: Python
CODE:
```
get_uint(packet)
```

----------------------------------------

TITLE: Directly Projecting from Camera to Pixel Space - Mathematical
DESCRIPTION: This set of equations provides an alternative method for directly projecting 3D points from the camera's coordinate system (X, Y, Z) to pixel coordinates. This method uses camera parameters (fx_pixel, fy_pixel, px_pixel, py_pixel) that are already defined in pixel space.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_17

LANGUAGE: mathematical
CODE:
```
x_pixel = -fx_pixel * X / Z + px_pixel
y_pixel =  fy_pixel * Y / Z + py_pixel
```

----------------------------------------

TITLE: Retrieving a Single-Precision Float from a MediaPipe Packet in Python
DESCRIPTION: Retrieves the single-precision float payload (mapped to C++ `float`) from a MediaPipe packet using `mp.packet_getter.get_float`. This method also handles double-precision floats.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_21

LANGUAGE: Python
CODE:
```
get_float(packet)
```

----------------------------------------

TITLE: Retrieving a 16-bit Signed Integer from a MediaPipe Packet in Python
DESCRIPTION: Retrieves the 16-bit signed integer payload (mapped to C++ `int16_t`) from a MediaPipe packet using `mp.packet_getter.get_int`. This method provides access to the integer content.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_7

LANGUAGE: Python
CODE:
```
get_int(packet)
```

----------------------------------------

TITLE: Creating a Double-Precision Float MediaPipe Packet in Python
DESCRIPTION: Creates a MediaPipe packet with a double-precision float payload (mapped to C++ `double`) using `mp.packet_creator.create_double`. This is used for higher precision floating-point numbers.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_22

LANGUAGE: Python
CODE:
```
create_double(1.1)
```

----------------------------------------

TITLE: Retrieving a Double-Precision Float from a MediaPipe Packet in Python
DESCRIPTION: Retrieves the double-precision float payload (mapped to C++ `double`) from a MediaPipe packet using `mp.packet_getter.get_float`. This method is versatile for both float and double types.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_23

LANGUAGE: Python
CODE:
```
get_float(packet)
```

----------------------------------------

TITLE: Adding Feature Timestamp - MediaPipe Audio
DESCRIPTION: This snippet shows how to associate a timestamp with a set of audio features. Timestamps are crucial for synchronizing features with the media timeline, typically in microseconds.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_74

LANGUAGE: Python
CODE:
```
add_feature_timestamp
```

LANGUAGE: C++
CODE:
```
AddFeatureTimestamp
```

----------------------------------------

TITLE: Adding Text Content - MediaPipe Text
DESCRIPTION: This snippet adds time-aligned segments of text as a feature list. It's ideal for captions, ASR results, or other text snippets that correspond to specific points in time within the media.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_84

LANGUAGE: Python
CODE:
```
add_text_content
```

LANGUAGE: C++
CODE:
```
AddTextContent
```

----------------------------------------

TITLE: Installing OpenCV Runtime Libraries on Linux
DESCRIPTION: This `apt-get` command installs the necessary OpenCV runtime development libraries on a Debian/Ubuntu-based target system. These libraries are prerequisites for MediaPipe applications that rely on OpenCV for image processing and computer vision functionalities.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/coral/README.md#_snippet_11

LANGUAGE: bash
CODE:
```
sudo apt-get install -y \
    libopencv-core-dev \
    libopencv-highgui-dev \
    libopencv-calib3d-dev \
    libopencv-features2d-dev \
    libopencv-imgproc-dev \
    libopencv-video-dev
```

----------------------------------------

TITLE: Adding Multiple Encoded Class Segmentation Masks - MediaPipe
DESCRIPTION: This snippet illustrates how to add multiple encoded segmentation masks for class labels, useful when handling overlapping masks within the same timestep.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_58

LANGUAGE: Python
CODE:
```
add_class_segmentation_multi_encoded
```

LANGUAGE: C++
CODE:
```
AddClassSegmentationMultiEncoded
```

----------------------------------------

TITLE: Setting Class Segmentation Image Height - MediaPipe
DESCRIPTION: This snippet shows how to set the height in pixels for the class segmentation image, providing essential dimension information for processing.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_60

LANGUAGE: Python
CODE:
```
set_class_segmentation_height
```

LANGUAGE: C++
CODE:
```
SetClassSegmentationHeight
```

----------------------------------------

TITLE: Setting Class Segmentation Image Width - MediaPipe
DESCRIPTION: This snippet demonstrates how to set the width in pixels for the class segmentation image, providing essential dimension information for processing.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_61

LANGUAGE: Python
CODE:
```
set_class_segmentation_width
```

LANGUAGE: C++
CODE:
```
SetClassSegmentationWidth
```

----------------------------------------

TITLE: Installing Core Development Packages on WSL
DESCRIPTION: This command updates the package list and installs essential development tools and dependencies required for MediaPipe, including build-essential, git, python, zip, adb, and openjdk-8-jdk.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_33

LANGUAGE: bash
CODE:
```
sudo apt-get update && sudo apt-get install -y build-essential git python zip adb openjdk-8-jdk
```

----------------------------------------

TITLE: Setting Feature Sample Rate - MediaPipe Audio
DESCRIPTION: This snippet sets the rate at which features appear per second, such as the rate of STFT windows for a spectrogram. It defines the temporal density of the extracted features.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_75

LANGUAGE: Python
CODE:
```
set_feature_sample_rate
```

LANGUAGE: C++
CODE:
```
SetFeatureSampleRate
```

----------------------------------------

TITLE: Setting Instance Segmentation Object Class Index Mapping - MediaPipe
DESCRIPTION: This snippet illustrates how to define a mapping from pixel values within the instance segmentation image to specific object class indices, allowing for detailed classification of segmented objects.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_72

LANGUAGE: Python
CODE:
```
set_instance_segmentation_object_class_index
```

LANGUAGE: C++
CODE:
```
SetInstanceSegmentationObjectClassIndex
```

----------------------------------------

TITLE: Setting Instance Segmentation Class Label String Mapping - MediaPipe
DESCRIPTION: This snippet demonstrates how to define a mapping from pixel values within the instance segmentation image to descriptive class label strings, enhancing readability and understanding of the segmentation output.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_71

LANGUAGE: Python
CODE:
```
set_instance_segmentation_class_label_string
```

LANGUAGE: C++
CODE:
```
SetInstanceSegmentationClassLabelString
```

----------------------------------------

TITLE: Setting Text Context Token ID - MediaPipe Text
DESCRIPTION: This snippet allows storing large blocks of text as a list of token IDs within the context. This is useful for pre-processed text data where each word or sub-word has a numerical identifier.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_82

LANGUAGE: Python
CODE:
```
set_text_context_token_id
```

LANGUAGE: C++
CODE:
```
SetTextContextTokenId
```

----------------------------------------

TITLE: Adding Text Embedding - MediaPipe Text
DESCRIPTION: This snippet adds a floating-point vector (embedding) for the corresponding text token. Text embeddings are used to represent text semantically for tasks like similarity comparison or classification.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_88

LANGUAGE: Python
CODE:
```
add_text_embedding
```

LANGUAGE: C++
CODE:
```
AddTextEmbedding
```

----------------------------------------

TITLE: Adding Bounding Box Y-Max (Python/C++)
DESCRIPTION: Adds a list of normalized maximum y values for bounding boxes in a frame. This function is used to specify the bottom edge of a bounding box.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_35

LANGUAGE: Python
CODE:
```
add_bbox_ymax
```

LANGUAGE: C++
CODE:
```
AddBBoxYMax
```

----------------------------------------

TITLE: Adding Bounding Box X-Max (Python/C++)
DESCRIPTION: Adds a list of normalized maximum x values for bounding boxes in a frame. This function is used to specify the right edge of a bounding box.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_36

LANGUAGE: Python
CODE:
```
add_bbox_xmax
```

LANGUAGE: C++
CODE:
```
AddBBoxXMax
```

----------------------------------------

TITLE: Adding Feature to MediaPipe Singular Feature List (Python/C++)
DESCRIPTION: This function appends a single feature of the appropriate type to a MediaPipe singular feature list. It modifies the example in place. An optional `prefix` can be used to specify the feature's namespace.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_24

LANGUAGE: python
CODE:
```
add_feature(value, example [, prefix])
```

LANGUAGE: c++
CODE:
```
AddFeature([const string& prefix,], const TYPE& value, tf::SE* example)
```

----------------------------------------

TITLE: Installing MediaPipe Build Dependencies (macOS)
DESCRIPTION: These Homebrew commands install necessary development tools for building MediaPipe from source on macOS. This includes the Protobuf compiler and CMake, which is required if OpenCV needs to be built from source.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python.md#_snippet_4

LANGUAGE: Bash
CODE:
```
$ brew install protobuf

# If you need to build opencv from source.
$ brew install cmake
```

----------------------------------------

TITLE: Setting Text Context Content - MediaPipe Text
DESCRIPTION: This snippet provides storage for large blocks of text within the context of the media. It's suitable for descriptive text or full document content.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_81

LANGUAGE: Python
CODE:
```
set_text_context_content
```

LANGUAGE: C++
CODE:
```
SetTextContextContent
```

----------------------------------------

TITLE: Setting Text Context Embedding - MediaPipe Text
DESCRIPTION: This snippet is used to store large blocks of text as embeddings (floating-point vectors) in the context. Text embeddings capture semantic meaning and are often used for similarity searches or machine learning tasks.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_83

LANGUAGE: Python
CODE:
```
set_text_context_embedding
```

LANGUAGE: C++
CODE:
```
SetTextContextEmbedding
```

----------------------------------------

TITLE: Adding Text Timestamp - MediaPipe Text
DESCRIPTION: This snippet adds a timestamp indicating when a specific text token occurs in microseconds. This is crucial for synchronizing text with audio or video content.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_85

LANGUAGE: Python
CODE:
```
add_text_timestamp
```

LANGUAGE: C++
CODE:
```
AddTextTimestamp
```

----------------------------------------

TITLE: Configuring MediaPipe Hello World Graph (C++)
DESCRIPTION: This C++ snippet defines the `PrintHelloWorld()` function, which configures a `CalculatorGraphConfig` proto. The graph consists of two `PassThroughCalculator` nodes connected in series, taking an "in" stream and producing an "out" stream. This configuration is essential for defining the data flow within MediaPipe.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_cpp.md#_snippet_1

LANGUAGE: c++
CODE:
```
absl::Status PrintHelloWorld() {
  // Configures a simple graph, which concatenates 2 PassThroughCalculators.
  CalculatorGraphConfig config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
    input_stream: "in"
    output_stream: "out"
    node {
      calculator: "PassThroughCalculator"
      input_stream: "in"
      output_stream: "out1"
    }
    node {
      calculator: "PassThroughCalculator"
      input_stream: "out1"
      output_stream: "out"
    }
  )");
```

----------------------------------------

TITLE: Setting MediaPipe Context Features (Python/C++)
DESCRIPTION: This function clears any existing feature and then stores a new list of features of the appropriate type within a MediaPipe context example. It requires the `values` to be provided as a list or vector. An optional `prefix` can be used to specify the feature's namespace.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_19

LANGUAGE: python
CODE:
```
set_feature(values, example [, prefix])
```

LANGUAGE: c++
CODE:
```
SetFeature([const string& prefix,], const vector<TYPE>& values, tf::SE* example)
```

----------------------------------------

TITLE: Retrieving ApplicationInfo in onCreate (Java)
DESCRIPTION: This Java code, placed within the `onCreate` method of `MainActivity`, attempts to retrieve the application's `ApplicationInfo` object. This object is necessary to access metadata defined in the `AndroidManifest.xml`, such as the camera facing preference.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_23

LANGUAGE: Java
CODE:
```
try {
  applicationInfo =
      getPackageManager().getApplicationInfo(getPackageName(), PackageManager.GET_META_DATA);

```

----------------------------------------

TITLE: Installing Python 'six' Library for TensorFlow Interoperability
DESCRIPTION: This command installs the 'six' Python library using pip3, which is a prerequisite for MediaPipe to function correctly when interoperating with TensorFlow. The '--user' flag ensures the package is installed in the user's home directory.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_0

LANGUAGE: Shell
CODE:
```
pip3 install --user six
```

----------------------------------------

TITLE: Adding Point Radius (Python/C++)
DESCRIPTION: Adds a list of radii for points in a frame. This can be used to represent the size or influence area of a point.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_41

LANGUAGE: Python
CODE:
```
add_bbox_point_radius
```

LANGUAGE: C++
CODE:
```
AddBBoxRadius
```

----------------------------------------

TITLE: Clearing MediaPipe Context Features (Python/C++)
DESCRIPTION: This function clears a specific feature from a MediaPipe context example. It modifies the example in place, removing the feature entirely. An optional `prefix` allows for clearing features within a specific namespace.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_18

LANGUAGE: python
CODE:
```
clear_feature(example [, prefix])
```

LANGUAGE: c++
CODE:
```
ClearFeature([const string& prefix,] tf::SE* example)
```

----------------------------------------

TITLE: Configuring Bazel for OpenCV 4 (Source Build) - Bazel
DESCRIPTION: This Bazel `WORKSPACE` and `cc_library` configuration is for MediaPipe to link against OpenCV 4 libraries built from source and installed to `/usr/local`. It adjusts the `path` for the local repository and includes `hdrs` for header discovery.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_7

LANGUAGE: Bazel
CODE:
```
# WORKSPACE
new_local_repository(
  name = "linux_opencv",
  build_file = "@//third_party:opencv_linux.BUILD",
  path = "/usr/local",
)

# opencv_linux.BUILD for OpenCV 4 installed to /usr/local
cc_library(
  name = "opencv",
  hdrs = glob([

```

----------------------------------------

TITLE: Retrieving a 64-bit Signed Integer from a MediaPipe Packet in Python
DESCRIPTION: Retrieves the 64-bit signed integer payload (mapped to C++ `int64_t`) from a MediaPipe packet using `mp.packet_getter.get_int`. This method supports large integer retrieval.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_11

LANGUAGE: Python
CODE:
```
get_int(packet)
```

----------------------------------------

TITLE: Symlinking Custom Provisioning Profile (Bash)
DESCRIPTION: This command sequence navigates to the MediaPipe directory and creates a symbolic link from a manually downloaded provisioning profile to the expected location within the MediaPipe project structure. This allows Bazel to use the custom profile for signing iOS applications, overriding any automatically generated profiles if they exist.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/ios.md#_snippet_7

LANGUAGE: bash
CODE:
```
cd mediapipe
ln -s ~/Downloads/MyProvisioningProfile.mobileprovision mediapipe/provisioning_profile.mobileprovision
```

----------------------------------------

TITLE: Adding Full Bounding Box (Python/C++)
DESCRIPTION: A special accessor that operates on ymin, xmin, ymax, and xmax with a single call. This provides a convenient way to add all bounding box coordinates simultaneously.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_37

LANGUAGE: Python
CODE:
```
add_bbox
```

LANGUAGE: C++
CODE:
```
AddBBox
```

----------------------------------------

TITLE: Adding Text Token ID - MediaPipe Text
DESCRIPTION: This snippet adds an integer ID for the corresponding text token. This is useful when text has been tokenized and each token is mapped to a unique numerical identifier.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_89

LANGUAGE: Python
CODE:
```
add_text_token_id
```

LANGUAGE: C++
CODE:
```
AddTextTokenId
```

----------------------------------------

TITLE: Clearing Singular Context Feature in MediaPipe MediaSequence (Python)
DESCRIPTION: Clears a singular context feature from a MediaSequence example. This effectively removes the feature from the example. An optional prefix can be used to target specific features.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_9

LANGUAGE: Python
CODE:
```
clear_feature(example [, prefix])
```

----------------------------------------

TITLE: Creating an 8-bit Signed Integer MediaPipe Packet in Python
DESCRIPTION: Creates a MediaPipe packet with an 8-bit signed integer payload (mapped to C++ `int8_t`) using `mp.packet_creator.create_int8`. This is useful for byte-sized integer data.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_4

LANGUAGE: Python
CODE:
```
create_int8(2**7-1)
```

----------------------------------------

TITLE: Connecting Camera Frames to ExternalTextureConverter via SurfaceHolder.Callback (Java)
DESCRIPTION: This code adds a `SurfaceHolder.Callback` to `previewDisplayView` to handle surface lifecycle events. In `surfaceChanged`, it computes the optimal display size for camera frames and then connects `previewFrameTexture` to the `converter`, configuring the converter's output dimensions. This pipes camera output to the `ExternalTextureConverter` for MediaPipe processing.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_30

LANGUAGE: Java
CODE:
```
previewDisplayView
 .getHolder()
 .addCallback(
     new SurfaceHolder.Callback() {
       @Override
       public void surfaceCreated(SurfaceHolder holder) {}

       @Override
       public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
         // (Re-)Compute the ideal size of the camera-preview display (the area that the
         // camera-preview frames get rendered onto, potentially with scaling and rotation)
         // based on the size of the SurfaceView that contains the display.
         Size viewSize = new Size(width, height);
         Size displaySize = cameraHelper.computeDisplaySizeFromViewSize(viewSize);

         // Connect the converter to the camera-preview frames as its input (via
         // previewFrameTexture), and configure the output width and height as the computed
         // display size.
         converter.setSurfaceTextureAndAttachToGLContext(
             previewFrameTexture, displaySize.getWidth(), displaySize.getHeight());
       }

       @Override
       public void surfaceDestroyed(SurfaceHolder holder) {}
     });
```

----------------------------------------

TITLE: Getting Size of MediaPipe List Feature Lists (Python/C++)
DESCRIPTION: This function returns the number of feature sequences associated with a specific key in a MediaPipe list feature. If the feature is absent, it returns 0. An optional `prefix` can be used to specify the feature's namespace.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_26

LANGUAGE: python
CODE:
```
get_feature_size(example [, prefix])
```

LANGUAGE: c++
CODE:
```
GetFeatureSize([const string& prefix,] const tf::SE& example)
```

----------------------------------------

TITLE: Retrieving Repeated Feature from MediaPipe List Feature Lists (Python/C++)
DESCRIPTION: This function retrieves a repeated feature (a sequence of features) from a MediaPipe list feature at a specified `index`. The returned feature is comparable to a list or vector of string, int64, or float. An optional `prefix` can be used to scope the feature retrieval.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_27

LANGUAGE: python
CODE:
```
get_feature_at(index, example [, prefix])
```

LANGUAGE: c++
CODE:
```
GetFeatureAt([const string& prefix,] const tf::SE& example, const int index)
```

----------------------------------------

TITLE: Implementing viewWillAppear for Camera Initialization (Objective-C)
DESCRIPTION: This snippet shows the basic implementation of the `viewWillAppear` method in an iOS `ViewController`. It calls the superclass's implementation, preparing the view for appearance and serving as an entry point for camera setup.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_16

LANGUAGE: Objective-C
CODE:
```
-(void)viewWillAppear:(BOOL)animated {
  [super viewWillAppear:animated];
}
```

----------------------------------------

TITLE: Adding Bounding Box X-Min (Python/C++)
DESCRIPTION: Adds a list of normalized minimum x values for bounding boxes in a frame. This function is used to specify the left edge of a bounding box.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_34

LANGUAGE: Python
CODE:
```
add_bbox_xmin
```

LANGUAGE: C++
CODE:
```
AddBBoxXMin
```

----------------------------------------

TITLE: Creating a 16-bit Signed Integer MediaPipe Packet in Python
DESCRIPTION: Creates a MediaPipe packet with a 16-bit signed integer payload (mapped to C++ `int16_t`) using `mp.packet_creator.create_int16`. This is suitable for short integer values.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_6

LANGUAGE: Python
CODE:
```
create_int16(2**15-1)
```

----------------------------------------

TITLE: Declaring MPPCameraInputSource Instance in ViewController (Objective-C)
DESCRIPTION: Declares an instance variable `_cameraSource` of type `MPPCameraInputSource` within the `ViewController`'s implementation block. This object handles camera access using the `AVCaptureSession` library.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_5

LANGUAGE: Objective-C
CODE:
```
@implementation ViewController {
  // Handles camera access via AVCaptureSession library.
  MPPCameraInputSource* _cameraSource;
}
```

----------------------------------------

TITLE: Getting Size of MediaPipe Singular Feature List (Python/C++)
DESCRIPTION: This function returns the number of features associated with a specific key in a MediaPipe singular feature list. If the feature is absent, it returns 0. An optional `prefix` can be used to specify the feature's namespace.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_22

LANGUAGE: python
CODE:
```
get_feature_size(example [, prefix])
```

LANGUAGE: c++
CODE:
```
GetFeatureSize([const string& prefix,] const tf::SE&(example)
```

----------------------------------------

TITLE: Adding Region Label Confidence (Python/C++)
DESCRIPTION: For each region, adds the confidence or weight associated with its label. If a region has multiple labels, the region must be duplicated for each label with its corresponding confidence.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_53

LANGUAGE: Python
CODE:
```
add_bbox_label_confidence
```

LANGUAGE: C++
CODE:
```
AddBBoxLabelConfidence
```

----------------------------------------

TITLE: Adding Sequence of Features to MediaPipe List Feature Lists (Python/C++)
DESCRIPTION: This function appends a sequence of features of the appropriate type to a MediaPipe list feature. It modifies the example in place. An optional `prefix` can be used to specify the feature's namespace.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_28

LANGUAGE: python
CODE:
```
add_feature(value, example [, prefix])
```

LANGUAGE: c++
CODE:
```
AddFeature([const string& prefix,], const vector<TYPE>& value, tf::SE* example)
```

----------------------------------------

TITLE: Adding Number of Regions (Python/C++)
DESCRIPTION: Adds the total number of boxes or other regions present in a frame. This value should be 0 for unannotated frames.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_47

LANGUAGE: Python
CODE:
```
add_bbox_num_regions
```

LANGUAGE: C++
CODE:
```
AddBBoxNumRegions
```

----------------------------------------

TITLE: Retrieving Feature Key for MediaPipe Context Features (Python/C++)
DESCRIPTION: This function returns the key string used by related functions to identify a feature within a MediaPipe context example. An optional `prefix` can be provided to generate a key for a specific namespace.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_20

LANGUAGE: python
CODE:
```
get_feature_key([prefix])
```

LANGUAGE: c++
CODE:
```
GetFeatureKey([const string& prefix])
```

----------------------------------------

TITLE: Getting Feature Key in MediaPipe MediaSequence (C++)
DESCRIPTION: Returns the internal key string used by related functions for a singular context feature. An optional prefix can be provided to generate a prefixed key.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_14

LANGUAGE: C++
CODE:
```
GetFeatureKey([const string& prefix])
```

----------------------------------------

TITLE: Adding 3D Point X-Coordinate (Python/C++)
DESCRIPTION: Adds a list of normalized x values for 3D points in a frame. This function is used to specify the horizontal coordinate of a 3D point.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_42

LANGUAGE: Python
CODE:
```
add_bbox_3d_point_x
```

LANGUAGE: C++
CODE:
```
AddBBox3dPointX
```

----------------------------------------

TITLE: Retrieving MediaPipe Context Features (Python/C++)
DESCRIPTION: This function retrieves a sequence feature from a MediaPipe context example. The returned feature is of an appropriate type, comparable to a list or vector of string, int64, or float. An optional `prefix` can be used to scope the feature retrieval.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_17

LANGUAGE: python
CODE:
```
get_feature(example [, prefix])
```

LANGUAGE: c++
CODE:
```
GetFeature([const string& prefix,] const tf::SE& example)
```

----------------------------------------

TITLE: Setting Singular Context Feature in MediaPipe MediaSequence (Python)
DESCRIPTION: Sets or updates a singular context feature in a MediaSequence example. The function first clears any existing feature and then stores the new value of the appropriate type. An optional prefix can be used.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_11

LANGUAGE: Python
CODE:
```
set_feature(value, example [, prefix])
```

----------------------------------------

TITLE: Updating ViewController Interface with MediaPipe Delegates in Objective-C
DESCRIPTION: Updates the `ViewController`'s interface definition to conform to the `MPPGraphDelegate` and `MPPInputSourceDelegate` protocols. This is essential for the `ViewController` to act as a delegate for the MediaPipe graph and input sources, enabling callback reception.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_31

LANGUAGE: Objective-C
CODE:
```
@interface ViewController () <MPPGraphDelegate, MPPInputSourceDelegate>
```

----------------------------------------

TITLE: Adding Region Timestamp (Python/C++)
DESCRIPTION: Adds the timestamp in microseconds for the region annotations. This helps in temporal synchronization of region data.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_46

LANGUAGE: Python
CODE:
```
add_bbox_timestamp
```

LANGUAGE: C++
CODE:
```
AddBBoxTimestamp
```

----------------------------------------

TITLE: Setting Text Language - MediaPipe Text
DESCRIPTION: This snippet is used to set the language for the corresponding text content. It's essential for language-specific processing or display of text features.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_80

LANGUAGE: Python
CODE:
```
set_text_langage
```

LANGUAGE: C++
CODE:
```
SetTextLanguage
```

----------------------------------------

TITLE: Adding Feature Floats - MediaPipe Audio
DESCRIPTION: This snippet demonstrates how to add a list of floating-point numbers as a feature at a specific timestep within MediaPipe's audio processing. It's used for storing numerical data associated with audio frames.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_73

LANGUAGE: Python
CODE:
```
add_feature_floats
```

LANGUAGE: C++
CODE:
```
AddFeatureFloats
```

----------------------------------------

TITLE: Getting Default Parser for MediaPipe List Feature Lists (Python)
DESCRIPTION: This Python-only function returns the `tf.io.VarLenFeature` parser, which is suitable for handling variable-length list features. It is used for parsing feature data during TensorFlow operations.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_29

LANGUAGE: python
CODE:
```
get_feature_default_parser()
```

----------------------------------------

TITLE: Running MediaPipe Image Classifier Benchmark with Custom Parameters (Bazel)
DESCRIPTION: This example shows how to run the MediaPipe image classifier benchmark with custom parameters. It specifies a downloaded model file (`classifier.tflite`) and limits the number of iterations to 200, overriding the default settings.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/benchmark/vision/image_classifier/README.md#_snippet_2

LANGUAGE: Bash
CODE:
```
bazel run -c opt :image_classifier_benchmark -- \
  --model classifier.tflite \
  --iterations 200
```

----------------------------------------

TITLE: Monitoring GPU Utilization with nvidia-smi
DESCRIPTION: This shell command uses `nvidia-smi` to continuously monitor and display the GPU utilization percentage. The `--loop=1` flag updates the output every second, allowing real-time verification of GPU activity during model inference.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/gpu_support.md#_snippet_11

LANGUAGE: Shell
CODE:
```
$ nvidia-smi --query-gpu=utilization.gpu --format=csv --loop=1

0 %
0 %
4 %
5 %
83 %
21 %
22 %
27 %
29 %
100 %
0 %
0%
```

----------------------------------------

TITLE: Reading YouTube-8M Features in Python (Python)
DESCRIPTION: This optional Python snippet demonstrates how to load and print the extracted YouTube-8M features from the `features.pb` file. It uses TensorFlow's `SequenceExample.FromString` method to deserialize the protobuf data, allowing inspection of the extracted features.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/youtube8m/README.md#_snippet_5

LANGUAGE: Python
CODE:
```
import tensorflow as tf

sequence_example = open('/tmp/mediapipe/features.pb', 'rb').read()
print(tf.train.SequenceExample.FromString(sequence_example))
```

----------------------------------------

TITLE: Building MediaPipe Graph with PassThroughNodeBuilder in C++
DESCRIPTION: This C++ function `BuildGraph` demonstrates the improved graph construction using the `PassThroughNodeBuilder` utility class. It significantly reduces the verbosity and error-proneness of connecting multiple streams to a `PassThroughCalculator` by abstracting away repetitive `ConnectTo` and `Cast` calls.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_7

LANGUAGE: C++
CODE:
```
CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Graph inputs.
  Stream<float> float_value = graph.In(0).SetName("float_value").Cast<float>();
  Stream<int> int_value = graph.In(1).SetName("int_value").Cast<int>();
  Stream<bool> bool_value = graph.In(2).SetName("bool_value").Cast<bool>();

  PassThroughNodeBuilder pass_node_builder(graph);
  Stream<float> passed_float_value = pass_node_builder.PassThrough(float_value);
  Stream<int> passed_int_value = pass_node_builder.PassThrough(int_value);
  Stream<bool> passed_bool_value = pass_node_builder.PassThrough(bool_value);

  // Graph outputs.
  passed_float_value.SetName("passed_float_value").ConnectTo(graph.Out(0));
  passed_int_value.SetName("passed_int_value").ConnectTo(graph.Out(1));
  passed_bool_value.SetName("passed_bool_value").ConnectTo(graph.Out(2));

  // Get `CalculatorGraphConfig` to pass it into `CalculatorGraph`
  return graph.GetConfig();
}
```

----------------------------------------

TITLE: Retrieving Singular Context Feature in MediaPipe MediaSequence (C++)
DESCRIPTION: Retrieves a single singular context feature from a MediaSequence example. The function returns the feature value, which can be a string, int64, or float. An optional prefix can be provided.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_8

LANGUAGE: C++
CODE:
```
GetFeature([const string& prefix,] const tf::SE& example)
```

----------------------------------------

TITLE: Initializing Serial Dispatch Queue for Video (Objective-C)
DESCRIPTION: Initializes `_videoQueue` as a serial dispatch queue with a `QOS_CLASS_USER_INTERACTIVE` priority. This ensures that camera frames are processed sequentially and with high responsiveness on a dedicated thread.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_10

LANGUAGE: Objective-C
CODE:
```
dispatch_queue_attr_t qosAttribute = dispatch_queue_attr_make_with_qos_class(
      DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, /*relative_priority=*/0);
_videoQueue = dispatch_queue_create(kVideoQueueLabel, qosAttribute);
```

----------------------------------------

TITLE: Setting Dataset Flags in MediaPipe Example (Python/C++)
DESCRIPTION: This function sets a list of bytes representing dataset-related attributes or flags for a MediaPipe example. The `example/dataset/flag/string` key stores this context bytes list type feature, allowing for flexible tagging of examples.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_32

LANGUAGE: python
CODE:
```
set_example_dataset_flag_string
```

LANGUAGE: c++
CODE:
```
SetExampleDatasetFlagString
```

----------------------------------------

TITLE: Adding Region Track String (Python/C++)
DESCRIPTION: For each region, adds its string track ID. If a region has multiple track IDs, the region must be duplicated for each ID.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_55

LANGUAGE: Python
CODE:
```
add_bbox_track_string
```

LANGUAGE: C++
CODE:
```
AddBBoxTrackString
```

----------------------------------------

TITLE: Running Object Detection with MediaPipe on Coral TPU (Shell)
DESCRIPTION: This command executes the MediaPipe object detection model on a desktop with a Coral TPU. It utilizes the `object_detection_desktop_live.pbtxt` calculator graph configuration file to configure the detection pipeline. The `GLOG_logtostderr=1` prefix directs logging output to standard error.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/coral/README.md#_snippet_14

LANGUAGE: Shell
CODE:
```
GLOG_logtostderr=1 ./object_detection_tpu --calculator_graph_config_file \n    mediapipe/examples/coral/graphs/object_detection_desktop_live.pbtxt
```

----------------------------------------

TITLE: Setting Instance Segmentation Image Format - MediaPipe
DESCRIPTION: This snippet demonstrates how to specify the encoding format (e.g., PNG, JPEG) for the object instance label images, ensuring correct interpretation of the segmentation data.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_67

LANGUAGE: Python
CODE:
```
set_instance_segmentation_format
```

LANGUAGE: C++
CODE:
```
SetInstanceSegmentationFormat
```

----------------------------------------

TITLE: Adding Region Label Index (Python/C++)
DESCRIPTION: For each region, adds its integer label. If a region has multiple labels, the region must be duplicated for each label.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_51

LANGUAGE: Python
CODE:
```
add_bbox_label_index
```

LANGUAGE: C++
CODE:
```
AddBBoxLabelIndex
```

----------------------------------------

TITLE: Adding MediaPipe Graph Data Dependency (Bazel)
DESCRIPTION: This Bazel `BUILD` file snippet adds a data dependency to a specific MediaPipe graph, `mobile_gpu_binary_graph` for edge detection. This ensures the graph's binary is included in the application bundle for runtime use.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_19

LANGUAGE: Bazel
CODE:
```
"//mediapipe/graphs/edge_detection:mobile_gpu_binary_graph",
```

----------------------------------------

TITLE: Converting Frozen Graph to TFLite Model (TensorFlow)
DESCRIPTION: This command converts the previously exported frozen TensorFlow graph (`tflite_graph.pb`) into a TFLite model (`model.tflite`). It specifies crucial parameters like the input and output formats, inference type (FLOAT), the expected input shape (1,320,320,3), and the names of the input and output arrays required for integration with MediaPipe Object Detection graphs.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/object_detection_saved_model.md#_snippet_2

LANGUAGE: bash
CODE:
```
tflite_convert --  \
  --graph_def_file=${PATH_TO_MODEL}/tflite_graph.pb \
  --output_file=${PATH_TO_MODEL}/model.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --inference_type=FLOAT \
  --input_shapes=1,320,320,3 \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays=raw_outputs/box_encodings,raw_outputs/class_predictions
```

----------------------------------------

TITLE: Declaring EglManager and ExternalTextureConverter (Java)
DESCRIPTION: These lines declare private member variables `eglManager` and `converter` within `MainActivity`. `eglManager` manages an OpenGL ES context, and `converter` is used to convert `SurfaceTexture` frames to regular OpenGL textures for MediaPipe.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_26

LANGUAGE: Java
CODE:
```
private EglManager eglManager;
private ExternalTextureConverter converter;
```

----------------------------------------

TITLE: Generating VGGish Frozen Graph (Bash)
DESCRIPTION: This step installs the `tf_slim` Python package, a dependency, and then executes a MediaPipe Python script to generate the VGGish frozen graph. This graph is required for extracting audio features using the VGGish model.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/youtube_8m.md#_snippet_2

LANGUAGE: bash
CODE:
```
cd -

pip3 install tf_slim
python -m mediapipe.examples.desktop.youtube8m.generate_vggish_frozen_graph
```

----------------------------------------

TITLE: Adding Text Confidence - MediaPipe Text
DESCRIPTION: This snippet adds a confidence score indicating the likelihood that the text is correct. This is commonly used with ASR results to assess their accuracy.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_87

LANGUAGE: Python
CODE:
```
add_text_confidence
```

LANGUAGE: C++
CODE:
```
AddTextConfidence
```

----------------------------------------

TITLE: Performing RAG Pipeline Inference with MediaPipe
DESCRIPTION: This snippet demonstrates how to set up and use the RAG Pipeline to augment LLM inference. It initializes necessary filesets, creates an LLM inference instance, configures the RAG pipeline with an embedding model, records external knowledge, and then generates a response based on a query.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/genai_experimental/README.md#_snippet_0

LANGUAGE: JavaScript
CODE:
```
const genaiFileset = await FilesetResolver.forGenAiTasks();
const genaiExperimentalFileset =
  await FilesetResolver.forGenAiExperimentalTasks();
const llmInference = await LlmInference.createFromModelPath(genaiFileset, ...);
const ragPipeline = await RagPipeline.createWithEmbeddingModel(
  genaiExperimentalFileset,
  llmInference,
  EMBEDDING_MODEL_URL,
);
await ragPipeline.recordBatchedMemory([
  'Paris is the capital of France.',
  'Berlin is the capital of Germany.',
]);
const result = await ragPipeline.generateResponse(
  'What is the capital of France?',
);
console.log(result);
```

----------------------------------------

TITLE: Cleaning Up 3D Object Assets (Shell)
DESCRIPTION: This shell script is the first step in processing user-provided 3D assets for MediaPipe. It cleans up .obj files from an INPUT_DIR and places the processed files into an INTERMEDIATE_OUTPUT_DIR, preparing them for further parsing.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_10

LANGUAGE: shell
CODE:
```
./mediapipe/graphs/object_detection_3d/obj_parser/obj_cleanup.sh [INPUT_DIR] [INTERMEDIATE_OUTPUT_DIR]
```

----------------------------------------

TITLE: Declaring Custom Timestamp Change in Contract (New Node API, C++)
DESCRIPTION: This example demonstrates how to declare a custom timestamp change behavior within the calculator contract using `TimestampChange::Arbitrary()`. This is necessary when a calculator modifies the timestamps of its outputs.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/api2/README.md#_snippet_9

LANGUAGE: C++
CODE:
```
MEDIAPIPE_NODE_CONTRACT(kMain, kLoop, kPrevLoop,
                        StreamHandler("ImmediateInputStreamHandler"),
                        TimestampChange::Arbitrary());
```

----------------------------------------

TITLE: Adding 3D Point Z-Coordinate (Python/C++)
DESCRIPTION: Adds a list of normalized z values for 3D points in a frame. This function is used to specify the depth coordinate of a 3D point.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_44

LANGUAGE: Python
CODE:
```
add_bbox_3d_point_z
```

LANGUAGE: C++
CODE:
```
AddBBox3dPointZ
```

----------------------------------------

TITLE: Configuring Bazel for OpenCV 2/3 (Source Build) - Bazel
DESCRIPTION: This Bazel `WORKSPACE` and `cc_library` configuration is for MediaPipe to link against OpenCV 2/3 libraries built from source and installed to `/usr/local`. It adjusts the `path` for the local repository and adds a `-L` flag to `linkopts` to specify the custom library path.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_6

LANGUAGE: Bazel
CODE:
```
# WORKSPACE
new_local_repository(
  name = "linux_opencv",
  build_file = "@//third_party:opencv_linux.BUILD",
  path = "/usr/local",
)

# opencv_linux.BUILD for OpenCV 2/3 installed to /usr/local
cc_library(
  name = "opencv",
  linkopts = [
    "-L/usr/local/lib",
    "-l:libopencv_core.so",
    "-l:libopencv_calib3d.so",
    "-l:libopencv_features2d.so",
    "-l:libopencv_highgui.so",
    "-l:libopencv_imgcodecs.so",
    "-l:libopencv_imgproc.so",
    "-l:libopencv_video.so",
    "-l:libopencv_videoio.so"
  ]
)
```

----------------------------------------

TITLE: Adding Region Label String (Python/C++)
DESCRIPTION: For each region, adds its string label. If a region has multiple labels, the region must be duplicated for each label.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_52

LANGUAGE: Python
CODE:
```
add_bbox_label_string
```

LANGUAGE: C++
CODE:
```
AddBBoxLabelString
```

----------------------------------------

TITLE: Adding Point Y-Coordinate (Python/C++)
DESCRIPTION: Adds a list of normalized y values for points in a frame. This function is used to specify the vertical coordinate of a point.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_39

LANGUAGE: Python
CODE:
```
add_bbox_point_y
```

LANGUAGE: C++
CODE:
```
AddBBoxPointY
```

----------------------------------------

TITLE: Running YouTube-8M Model Inference Web Server (Bash)
DESCRIPTION: This snippet starts a Python web server for the YouTube-8M model inference, requiring `absl-py` to be installed. The server allows users to interact with the inference model via a web browser, typically at `localhost:8008`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/youtube8m/README.md#_snippet_10

LANGUAGE: Bash
CODE:
```
python mediapipe/examples/desktop/youtube8m/viewer/server.py --root `pwd`
```

----------------------------------------

TITLE: Re-establishing SSH Connection with X Forwarding
DESCRIPTION: If `glxinfo` fails with an 'unable to open display' error when connected via SSH, this command re-establishes the SSH connection with X forwarding enabled, allowing graphical applications and GPU probes to function correctly.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/gpu_support.md#_snippet_2

LANGUAGE: bash
CODE:
```
ssh -X <user>@<host>
```

----------------------------------------

TITLE: Checking Feature Presence in MediaPipe MediaSequence (Python)
DESCRIPTION: Checks if a singular context feature is present in a MediaSequence example. An optional prefix can be used to specify the feature namespace. Returns a boolean indicating presence.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_5

LANGUAGE: Python
CODE:
```
has_feature(example [, prefix])
```

----------------------------------------

TITLE: Parsing 3D Object Files with Bazel ObjParser
DESCRIPTION: This Bazel command runs the ObjParser utility, which converts cleaned .obj files from an intermediate directory into a single OpenGL-ready .uuu animation file. The output directory specifies where the final processed asset will be placed. Note that input directories must be absolute paths.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/instant_motion_tracking.md#_snippet_1

LANGUAGE: Bazel
CODE:
```
bazel run -c opt mediapipe/graphs/object_detection_3d/obj_parser:ObjParser -- input_dir=[INTERMEDIATE_OUTPUT_DIR] output_dir=[OUTPUT_DIR]
```

----------------------------------------

TITLE: Running MediaPipe Hello World Example (Bash)
DESCRIPTION: This snippet provides the bash commands to clone the MediaPipe repository, navigate into it, set a logging environment variable, and execute the 'hello_world' example using Bazel. It explicitly disables GPU support for desktop environments and expects 10 'Hello World!' outputs.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_cpp.md#_snippet_0

LANGUAGE: bash
CODE:
```
$ git clone https://github.com/google/mediapipe.git
$ cd mediapipe

$ export GLOG_logtostderr=1
# Need bazel flag 'MEDIAPIPE_DISABLE_GPU=1' as desktop GPU is not supported currently.
$ bazel run --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/hello_world:hello_world

# It should print 10 rows of Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
```

----------------------------------------

TITLE: Running MediaPipe C++ Hello World Example (macOS)
DESCRIPTION: This command sets an environment variable to direct GLOG output to stderr and then uses Bazel to run the MediaPipe C++ 'Hello World' example. The `--define MEDIAPIPE_DISABLE_GPU=1` flag is crucial as desktop GPU support is not currently available, ensuring the example runs successfully on CPU.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_27

LANGUAGE: bash
CODE:
```
$ export GLOG_logtostderr=1
# Need bazel flag 'MEDIAPIPE_DISABLE_GPU=1' as desktop GPU is currently not supported
$ bazel run --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/hello_world:hello_world

# Should print:
# Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
# Hello World!
```

----------------------------------------

TITLE: Configuring Bazel for OpenCV 2/3 (Package Manager) - Bazel
DESCRIPTION: This Bazel `WORKSPACE` and `cc_library` configuration defines how MediaPipe links against OpenCV 2/3 libraries installed via a Debian package manager. It sets up a local repository pointing to `/usr` and specifies the necessary OpenCV shared libraries for linking.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_4

LANGUAGE: Bazel
CODE:
```
# WORKSPACE
new_local_repository(
  name = "linux_opencv",
  build_file = "@//third_party:opencv_linux.BUILD",
  path = "/usr",
)

# opencv_linux.BUILD for OpenCV 2/3 installed from Debian package
cc_library(
  name = "opencv",
  linkopts = [
    "-l:libopencv_core.so",
    "-l:libopencv_calib3d.so",
    "-l:libopencv_features2d.so",
    "-l:libopencv_highgui.so",
    "-l:libopencv_imgcodecs.so",
    "-l:libopencv_imgproc.so",
    "-l:libopencv_video.so",
    "-l:libopencv_videoio.so",
  ]
)
```

----------------------------------------

TITLE: Building Two-stage Objectron for Cups (Android)
DESCRIPTION: This command builds the MediaPipe Objectron Android example for 3D object detection of cups using Bazel. It targets the ARM64 architecture and activates the cup model via the '--define cup=true' flag.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_6

LANGUAGE: bash
CODE:
```
bazel build -c opt --config android_arm64 --define cup=true mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d:objectdetection3d
```

----------------------------------------

TITLE: Converting Localizations to Regions with LocalizationToRegionCalculator (MediaPipe Graph Configuration)
DESCRIPTION: This node converts object detections into regions using the `LocalizationToRegionCalculator`. It processes `object_detections` and outputs `object_regions`, configured to output all signals.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/autoflip.md#_snippet_4

LANGUAGE: MediaPipe Graph Configuration
CODE:
```
node {
  calculator: "LocalizationToRegionCalculator"
  input_stream: "DETECTIONS:object_detections"
  output_stream: "REGIONS:object_regions"
  options {
    [type.googleapis.com/mediapipe.autoflip.LocalizationToRegionCalculatorOptions] {
      output_all_signals: true
    }
  }
}
```

----------------------------------------

TITLE: Clearing Singular Context Feature in MediaPipe MediaSequence (C++)
DESCRIPTION: Clears a singular context feature from a MediaSequence example. This effectively removes the feature from the example. An optional prefix can be used to target specific features.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_10

LANGUAGE: C++
CODE:
```
ClearFeature([const string& prefix,] tf::SE* example)
```

----------------------------------------

TITLE: Initializing and Classifying Audio with MediaPipe Audio Classifier (JavaScript)
DESCRIPTION: This snippet demonstrates how to initialize the MediaPipe Audio Classifier and perform classification on audio data. It requires the `FilesetResolver` to load the WASM module and `AudioClassifier` to create an instance from a pre-trained model path. The `classify` method then processes the `audioData` to return classifications.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/audio/README.md#_snippet_0

LANGUAGE: JavaScript
CODE:
```
const audio = await FilesetResolver.forAudioTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio/wasm"
);
const audioClassifier = await AudioClassifier.createFromModelPath(audio,
    "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite"
);
const classifications = audioClassifier.classify(audioData);
```

----------------------------------------

TITLE: Defining a MediaPipe Graph with Two Nodes
DESCRIPTION: This snippet defines a simple MediaPipe graph configuration. It shows two nodes, 'A' and 'B', with their respective calculator types, input streams, and output streams. Node 'B' depends on the output of node 'A' ('alpha') and another stream ('foo'). This configuration illustrates how data flows between calculators in a graph.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/realtime_streams.md#_snippet_0

LANGUAGE: MediaPipe Graph Configuration
CODE:
```
node {
   calculator: "A"
   input_stream: "alpha_in"
   output_stream: "alpha"
}
node {
   calculator: "B"
   input_stream: "alpha"
   input_stream: "foo"
   output_stream: "beta"
}
```

----------------------------------------

TITLE: Classifying Images with MediaPipe Image Classifier (JavaScript)
DESCRIPTION: This snippet demonstrates how to initialize the MediaPipe Image Classifier and perform image classification. It requires loading the vision tasks WASM module and creating the classifier from a pre-trained model path. The `classify` method then processes an HTML image element to return classification results.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md#_snippet_6

LANGUAGE: JavaScript
CODE:
```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const imageClassifier = await ImageClassifier.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite"
);
const image = document.getElementById("image") as HTMLImageElement;
const classifications = imageClassifier.classify(image);
```

----------------------------------------

TITLE: Building and Running Tulsi for Xcode Project Generation (Bash)
DESCRIPTION: This sequence of commands clones the Tulsi repository, navigates into it, modifies its Bazel configuration to remove Xcode version dependency, and then builds and runs the Tulsi application. Tulsi is used to generate Xcode projects from Bazel build configurations for MediaPipe iOS examples.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/ios.md#_snippet_5

LANGUAGE: bash
CODE:
```
# cd out of the mediapipe directory, then:
git clone https://github.com/bazelbuild/tulsi.git
cd tulsi
# remove Xcode version from Tulsi's .bazelrc (see http://github.com/bazelbuild/tulsi#building-and-installing):
sed -i .orig '/xcode_version/d' .bazelrc
# build and run Tulsi:
sh build_and_run.sh
```

----------------------------------------

TITLE: Cross-Compiling Face Detection for ARM64 in Docker
DESCRIPTION: This Bazel command, run inside the ARM64 Docker environment, cross-compiles the MediaPipe face detection example. It configures the build with the appropriate crosstool, GCC compiler, aarch64 CPU architecture, and includes support for Coral USB Edge TPU and `libusb` linking.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/coral/README.md#_snippet_4

LANGUAGE: bash
CODE:
```
bazel build \
    --crosstool_top=@crosstool//:toolchains \
    --compiler=gcc \
    --cpu=aarch64 \
    --define darwinn_portable=1 \
    --define MEDIAPIPE_DISABLE_GPU=1 \
    --define MEDIAPIPE_EDGE_TPU=usb \
    --linkopt=-l:libusb-1.0.so \
    mediapipe/examples/coral:face_detection_tpu build
```

----------------------------------------

TITLE: Converting Camera Parameters (fx, fy) from Pixel to NDC Space - Mathematical
DESCRIPTION: These formulas define how to convert the focal length camera parameters (fx_pixel, fy_pixel) from pixel space to Normalized Device Coordinates (NDC) space. This conversion is essential for consistency when working with different coordinate system definitions.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_18

LANGUAGE: mathematical
CODE:
```
fx = fx_pixel * 2.0 / image_width
fy = fy_pixel * 2.0 / image_height
```

----------------------------------------

TITLE: Creating an 8-bit Unsigned Integer MediaPipe Packet in Python
DESCRIPTION: Creates a MediaPipe packet with an 8-bit unsigned integer payload (mapped to C++ `uint8_t`) using `mp.packet_creator.create_uint8`. This is ideal for byte data where only positive values are expected.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_12

LANGUAGE: Python
CODE:
```
create_uint8(2**8-1)
```

----------------------------------------

TITLE: Running MediaPipe Hello World C++ Example (Bash)
DESCRIPTION: This command sets the GLOG_logtostderr environment variable to direct logs to stderr and then uses Bazel to run the MediaPipe 'hello_world' C++ example. The `--define MEDIAPIPE_DISABLE_GPU=1` flag is crucial for running on Linux desktops with CPU-only configurations, ensuring the example executes without GPU dependencies.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_19

LANGUAGE: bash
CODE:
```
$ export GLOG_logtostderr=1
# Need bazel flag 'MEDIAPIPE_DISABLE_GPU=1' if you are running on Linux desktop with CPU only
$ bazel run --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/hello_world:hello_world
```

----------------------------------------

TITLE: Extracting Image Embeddings with MediaPipe Image Embedder (JavaScript)
DESCRIPTION: This snippet illustrates the initialization and usage of the MediaPipe Image Embedder to extract numerical embeddings from an image. It involves loading the vision tasks WASM module and instantiating the embedder from a model path. The `embed` method processes an HTML image element to generate embeddings.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md#_snippet_7

LANGUAGE: JavaScript
CODE:
```
const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const imageEmbedder = await ImageEmbedder.createFromModelPath(vision,
    "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite"
);
const image = document.getElementById("image") as HTMLImageElement;
const embeddings = imageSegmenter.embed(image);
```

----------------------------------------

TITLE: Building MediaPipe YouTube-8M Model Inference Binary for Web (Bash)
DESCRIPTION: This snippet builds the MediaPipe YouTube-8M model inference binary using Bazel, disabling GPU support. This binary is a prerequisite for running the Python web server that provides a web interface for model inference.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/youtube8m/README.md#_snippet_9

LANGUAGE: Bash
CODE:
```
bazel build -c opt --define='MEDIAPIPE_DISABLE_GPU=1' --linkopt=-s \
  mediapipe/examples/desktop/youtube8m:model_inference
```

----------------------------------------

TITLE: Setting Feature Number of Channels - MediaPipe Audio
DESCRIPTION: This snippet specifies the number of audio channels present in each stored feature. This is important for correctly interpreting multi-channel audio data.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_76

LANGUAGE: Python
CODE:
```
set_feature_num_channels
```

LANGUAGE: C++
CODE:
```
SetFeatureNumChannels
```

----------------------------------------

TITLE: Setting Class Segmentation Image Format - MediaPipe
DESCRIPTION: This snippet demonstrates how to specify the encoding format (e.g., PNG, JPEG) for the class label images, ensuring correct interpretation of the segmentation data.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_59

LANGUAGE: Python
CODE:
```
set_class_segmentation_format
```

LANGUAGE: C++
CODE:
```
SetClassSegmentationFormat
```

----------------------------------------

TITLE: Implementing MPPInputSourceDelegate for Video Frame Processing (Objective-C)
DESCRIPTION: Implements the `processVideoFrame` delegate method from `MPPInputSourceDelegate`. This method receives `CVPixelBufferRef` frames, verifies the source, retains the image buffer, and dispatches the rendering to the main queue using `_renderer` for display.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_15

LANGUAGE: Objective-C
CODE:
```
// Must be invoked on _videoQueue.
- (void)processVideoFrame:(CVPixelBufferRef)imageBuffer
                timestamp:(CMTime)timestamp
               fromSource:(MPPInputSource*)source {
  if (source != _cameraSource) {
    NSLog(@"Unknown source: %@", source);
    return;
  }
  // Display the captured image on the screen.
  CFRetain(imageBuffer);
  dispatch_async(dispatch_get_main_queue(), ^{
    [_renderer renderPixelBuffer:imageBuffer];
    CFRelease(imageBuffer);
  });
}
```

----------------------------------------

TITLE: Classifying Text with MediaPipe Text Classifier (JavaScript)
DESCRIPTION: This snippet illustrates how to set up and utilize the MediaPipe Text Classifier for categorizing text, such as sentiment analysis. It involves resolving the WASM files, creating a TextClassifier instance from a BERT classifier model, and then applying the classify method to the input textData.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/text/README.md#_snippet_1

LANGUAGE: JavaScript
CODE:
```
const text = await FilesetResolver.forTextTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text/wasm"
);
const textClassifier = await TextClassifier.createFromModelPath(text,
    "https://storage.googleapis.com/mediapipe-models/text_classifier/bert_classifier/float32/1/bert_classifier.tflite"
);
const classifications = textClassifier.classify(textData);
```

----------------------------------------

TITLE: Defining a MediaPipe Graph with PacketClonerCalculator (Protocol Buffer)
DESCRIPTION: This Protocol Buffer snippet defines a simple MediaPipe graph. It specifies three input streams and a single `PacketClonerCalculator` node. The calculator consumes all three input streams and produces two output streams, demonstrating its role in cloning packets within the graph.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/calculators.md#_snippet_12

LANGUAGE: Protocol Buffer
CODE:
```
input_stream: "room_mic_signal"
input_stream: "room_lighting_sensor"
input_stream: "room_video_tick_signal"

node {
   calculator: "PacketClonerCalculator"
   input_stream: "room_mic_signal"
   input_stream: "room_lighting_sensor"
   input_stream: "room_video_tick_signal"
   output_stream: "cloned_room_mic_signal"
   output_stream: "cloned_lighting_sensor"
 }
```

----------------------------------------

TITLE: Configuring a MediaPipe Calculator with Multiple Inputs
DESCRIPTION: This MediaPipe graph configuration snippet defines a calculator named 'SomeCalculator' with two input streams, 'a' and 'b', aliased from 'INPUT_A' and 'INPUT_B' respectively. This setup is used in an example scenario to demonstrate how `DefaultInputStreamHandler` reports missing packets during timestamp settlement.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/troubleshooting.md#_snippet_4

LANGUAGE: MediaPipe Graph Config
CODE:
```
node {
  calculator: "SomeCalculator"
  input_stream: "INPUT_A:a"
  input_stream: "INPUT_B:b"
  ...
}
```

----------------------------------------

TITLE: Setting TensorFlow CUDA Paths
DESCRIPTION: This command sets the `TF_CUDA_PATHS` environment variable, which informs TensorFlow about the locations of CUDA libraries and include files. It includes the CUDA toolkit path, standard library path, and include path for cudablas and libcudnn.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/gpu_support.md#_snippet_7

LANGUAGE: bash
CODE:
```
$ export TF_CUDA_PATHS=/usr/local/cuda-10.1,/usr/lib/x86_64-linux-gnu,/usr/include
```

----------------------------------------

TITLE: Handling SurfaceHolder Lifecycle for FrameProcessor Output in Android
DESCRIPTION: Implements the `surfaceCreated` and `surfaceDestroyed` methods of a custom `SurfaceHolder.Callback`. In `surfaceCreated`, the `Surface` from the `SurfaceHolder` is set as the output target for the `FrameProcessor`'s video output. In `surfaceDestroyed`, the output surface is set to `null` to release resources, ensuring proper handling of the display surface's lifecycle.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_40

LANGUAGE: Java
CODE:
```
@Override
public void surfaceCreated(SurfaceHolder holder) {
  processor.getVideoSurfaceOutput().setSurface(holder.getSurface());
}

@Override
public void surfaceDestroyed(SurfaceHolder holder) {
  processor.getVideoSurfaceOutput().setSurface(null);
}
```

----------------------------------------

TITLE: Building MediaPipe Charades Dataset with Bazel and Python
DESCRIPTION: This snippet provides the commands to build the `media_sequence_demo` binary and then generate the Charades dataset using a specific Python script. It's intended for training/evaluating action recognition models and requires compliance with the Charades data set license. TensorFlow must be installed, and commands executed from the MediaPipe repo root.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/media_sequence/README.md#_snippet_1

LANGUAGE: Bash
CODE:
```
bazel build -c opt mediapipe/examples/desktop/media_sequence:media_sequence_demo \
  --define MEDIAPIPE_DISABLE_GPU=1

python -m mediapipe.examples.desktop.media_sequence.charades_dataset \
  --alsologtostderr \
  --path_to_charades_data=/tmp/charades_data/ \
  --path_to_mediapipe_binary=bazel-bin/mediapipe/examples/desktop/\
media_sequence/media_sequence_demo  \
  --path_to_graph_directory=mediapipe/graphs/media_sequence/
```

----------------------------------------

TITLE: Initializing PacketClonerCalculator Open Method in C++
DESCRIPTION: The `Open` method initializes the `PacketClonerCalculator`. It determines the index of the 'tick' signal and resizes the internal `current_` vector to store the most recent packets from the 'base' input streams. It also propagates headers from input streams to their corresponding output streams if present.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/calculators.md#_snippet_9

LANGUAGE: C++
CODE:
```
  absl::Status Open(CalculatorContext* cc) final {
    tick_signal_index_ = cc->Inputs().NumEntries() - 1;
    current_.resize(tick_signal_index_);
    // Pass along the header for each stream if present.
    for (int i = 0; i < tick_signal_index_; ++i) {
      if (!cc->Inputs().Index(i).Header().IsEmpty()) {
        cc->Outputs().Index(i).SetHeader(cc->Inputs().Index(i).Header());
        // Sets the output stream of index i header to be the same as
        // the header for the input stream of index i
      }
    }
    return absl::OkStatus();
  }
```

----------------------------------------

TITLE: Processing Packets with PacketClonerCalculator Process Method in C++
DESCRIPTION: The `Process` method is the core logic of the `PacketClonerCalculator`. It first updates the `current_` buffer with the latest packets from the 'base' input streams. If a packet is present on the 'tick' signal stream, it then outputs the buffered 'base' packets, synchronized to the 'tick' signal's timestamp. If a buffered packet is empty, it sets the next timestamp bound for that output stream.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/calculators.md#_snippet_10

LANGUAGE: C++
CODE:
```
  absl::Status Process(CalculatorContext* cc) final {
    // Store input signals.
    for (int i = 0; i < tick_signal_index_; ++i) {
      if (!cc->Inputs().Index(i).Value().IsEmpty()) {
        current_[i] = cc->Inputs().Index(i).Value();
      }
    }

    // Output if the tick signal is non-empty.
    if (!cc->Inputs().Index(tick_signal_index_).Value().IsEmpty()) {
      for (int i = 0; i < tick_signal_index_; ++i) {
        if (!current_[i].IsEmpty()) {
          cc->Outputs().Index(i).AddPacket(
              current_[i].At(cc->InputTimestamp()));
          // Add a packet to output stream of index i a packet from inputstream i
          // with timestamp common to all present inputs
        } else {
          cc->Outputs().Index(i).SetNextTimestampBound(
              cc->InputTimestamp().NextAllowedInStream());
          // if current_[i], 1 packet buffer for input stream i is empty, we will set
          // next allowed timestamp for input stream i to be current timestamp + 1
        }
      }
    }
    return absl::OkStatus();
  }
```

----------------------------------------

TITLE: Setting Text Duration - MediaPipe Text
DESCRIPTION: This snippet sets the duration in microseconds for corresponding text tokens. It defines how long a particular text segment is active or displayed.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_86

LANGUAGE: Python
CODE:
```
add_text_duration
```

LANGUAGE: C++
CODE:
```
SetTextDuration
```

----------------------------------------

TITLE: Setting Feature Audio Sample Rate - MediaPipe Audio
DESCRIPTION: This snippet specifies the sample rate of the original audio from which derived features, such as spectrograms, were computed. It provides context for features that are transformations of raw audio.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_79

LANGUAGE: Python
CODE:
```
set_feature_audio_sample_rate
```

LANGUAGE: C++
CODE:
```
SetFeatureAudioSampleRate
```

----------------------------------------

TITLE: Setting Feature Packet Rate - MediaPipe Audio
DESCRIPTION: This snippet sets the rate at which audio packets appear per second. It describes the frequency of discrete audio data blocks.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_78

LANGUAGE: Python
CODE:
```
set_feature_packet_rate
```

LANGUAGE: C++
CODE:
```
SetFeaturePacketRate
```

----------------------------------------

TITLE: Receiving Output Pixel Buffer from MediaPipe Graph in Objective-C
DESCRIPTION: Implements the `mediapipeGraph:didOutputPixelBuffer:fromStream:` delegate method to receive processed `CVPixelBufferRef` frames from the MediaPipe graph's `kOutputStream`. The received pixel buffer is then retained, rendered on the main queue, and released.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_30

LANGUAGE: Objective-C
CODE:
```
- (void)mediapipeGraph:(MPPGraph*)graph
   didOutputPixelBuffer:(CVPixelBufferRef)pixelBuffer
             fromStream:(const std::string&)streamName {
  if (streamName == kOutputStream) {
    // Display the captured image on the screen.
    CVPixelBufferRetain(pixelBuffer);
    dispatch_async(dispatch_get_main_queue(), ^{
      [_renderer renderPixelBuffer:pixelBuffer];
      CVPixelBufferRelease(pixelBuffer);
    });
  }
}
```

----------------------------------------

TITLE: Checking Feature Presence in MediaPipe MediaSequence (C++)
DESCRIPTION: Checks if a singular context feature is present in a MediaSequence example. An optional prefix can be used to specify the feature namespace. Returns a boolean indicating presence.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_6

LANGUAGE: C++
CODE:
```
HasFeature([const string& prefix,] const tf::SE& example)
```

----------------------------------------

TITLE: Adding Full 3D Point (Python/C++)
DESCRIPTION: A special accessor that operates on 3d_point/{x,y,z} with a single call. This provides a convenient way to add all three coordinates of a 3D point simultaneously.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_45

LANGUAGE: Python
CODE:
```
add_bbox_3d_point
```

LANGUAGE: C++
CODE:
```
AddBBox3dPoint
```

----------------------------------------

TITLE: Adding Region Occlusion Status (Python/C++)
DESCRIPTION: For each region, indicates whether it is occluded (1) in the current frame. This provides information about visibility of the region.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_50

LANGUAGE: Python
CODE:
```
add_bbox_is_occluded
```

LANGUAGE: C++
CODE:
```
AddBBoxIsOccluded
```

----------------------------------------

TITLE: Adding Core iOS and MediaPipe Framework Dependencies (Bazel)
DESCRIPTION: This Bazel `BUILD` file snippet defines essential iOS SDK frameworks (AVFoundation, CoreGraphics, CoreMedia) and MediaPipe Objective-C dependencies required for the application to compile and link correctly, enabling camera input and rendering.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_18

LANGUAGE: Bazel
CODE:
```
sdk_frameworks = [
    "AVFoundation",
    "CoreGraphics",
    "CoreMedia",
],
deps = [
    "//mediapipe/objc:mediapipe_framework_ios",
    "//mediapipe/objc:mediapipe_input_sources_ios",
    "//mediapipe/objc:mediapipe_layer_renderer",
],
```

----------------------------------------

TITLE: Configuring Camera Facing in Bazel BUILD File (Bazel)
DESCRIPTION: This Bazel snippet modifies the `manifest_values` attribute within the `helloworld` android binary rule in the `BUILD` file. It sets `cameraFacingFront` to `False`, indicating that the back camera should be used by default for the application.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_21

LANGUAGE: Bazel
CODE:
```
manifest_values = {
    "applicationId": "com.google.mediapipe.apps.basic",
    "appName": "Hello World",
    "mainActivity": ".MainActivity",
    "cameraFacingFront": "False",
},
```

----------------------------------------

TITLE: Cross-Compiling Face Detection for ARM32 in Docker
DESCRIPTION: This Bazel command, executed within the ARM32 Docker environment, cross-compiles the MediaPipe face detection example. It specifies the crosstool, GCC compiler, ARMv7a CPU architecture, and enables Coral USB Edge TPU support, along with linking `libusb`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/coral/README.md#_snippet_3

LANGUAGE: bash
CODE:
```
bazel build \
    --crosstool_top=@crosstool//:toolchains \
    --compiler=gcc \
    --cpu=armv7a \
    --define darwinn_portable=1 \
    --define MEDIAPIPE_DISABLE_GPU=1 \
    --define MEDIAPIPE_EDGE_TPU=usb \
    --linkopt=-l:libusb-1.0.so \
    mediapipe/examples/coral:face_detection_tpu build
```

----------------------------------------

TITLE: Python Project Dependencies
DESCRIPTION: Specifies the Python packages and their exact or range-based version requirements necessary to run the MediaPipe project. This list includes dependencies for machine learning (TensorFlow, tf-models-official), numerical operations (NumPy), image processing (OpenCV), and utility libraries (absl-py, setuptools).
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/model_maker/requirements_bazel.txt#_snippet_0

LANGUAGE: Python
CODE:
```
absl-py
numpy<2
opencv-python
setuptools==70.3.0 # needed due to https://github.com/pypa/setuptools/issues/4487
tensorflow>=2.10,<2.16
tensorflow-addons
tensorflow-datasets
tensorflow-hub
tensorflow-model-optimization<0.8.0
tensorflow-text
tf-models-official>=2.13.2,<2.16.0
```

----------------------------------------

TITLE: Generating VGGish Frozen Graph for MediaPipe (Bash)
DESCRIPTION: This snippet navigates to the MediaPipe root, installs the `tf_slim` Python package, and then executes a MediaPipe Python script to generate the VGGish frozen graph. This graph is essential for audio feature extraction within the YouTube-8M pipeline, requiring Python 2.7 or 3.5+ with TensorFlow 1.14+.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/youtube8m/README.md#_snippet_2

LANGUAGE: Bash
CODE:
```
# cd to the root directory of the MediaPipe repo
cd -

pip3 install tf_slim
python -m mediapipe.examples.desktop.youtube8m.generate_vggish_frozen_graph
```

----------------------------------------

TITLE: Adding Region Annotation Status (Python/C++)
DESCRIPTION: Indicates whether a timestep is annotated (1) or not (0). This helps distinguish between empty frames and frames that are explicitly unannotated.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_48

LANGUAGE: Python
CODE:
```
add_bbox_is_annotated
```

LANGUAGE: C++
CODE:
```
AddBBoxIsAnnotated
```

----------------------------------------

TITLE: Adding Point X-Coordinate (Python/C++)
DESCRIPTION: Adds a list of normalized x values for points in a frame. This function is used to specify the horizontal coordinate of a point.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_38

LANGUAGE: Python
CODE:
```
add_bbox_point_x
```

LANGUAGE: C++
CODE:
```
AddBBoxPointX
```

----------------------------------------

TITLE: Implementing the Process Method in a MediaPipe C++ Calculator
DESCRIPTION: This C++ snippet demonstrates a basic `Process` method for a MediaPipe calculator. It retrieves input data, allocates memory for output, performs calculations, and adds the result to the output stream, returning `absl::OkStatus()` on success.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/calculators.md#_snippet_4

LANGUAGE: C++
CODE:
```
absl::Status MyCalculator::Process() {
  const Matrix& input = Input()->Get<Matrix>();
  std::unique_ptr<Matrix> output(new Matrix(input.rows(), input.cols()));
  // do your magic here....
  //    output->row(n) =  ...
  Output()->Add(output.release(), InputTimestamp());
  return absl::OkStatus();
}
```

----------------------------------------

TITLE: Declaring Live View and Renderer Instances (Objective-C)
DESCRIPTION: Declares an `IBOutlet` for `_liveView`, a `UIView` object intended to display the camera preview. Also declares an instance variable `_renderer` of type `MPPLayerRenderer` to handle the actual rendering of frames within that view.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_13

LANGUAGE: Objective-C
CODE:
```
// Display the camera preview frames.
IBOutlet UIView* _liveView;
// Render frames in a layer.
MPPLayerRenderer* _renderer;
```

----------------------------------------

TITLE: Getting Feature Key in MediaPipe MediaSequence (Python)
DESCRIPTION: Returns the internal key string used by related functions for a singular context feature. An optional prefix can be provided to generate a prefixed key.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_13

LANGUAGE: Python
CODE:
```
get_feature_key([prefix])
```

----------------------------------------

TITLE: Initializing MPPCameraInputSource in viewDidLoad (Objective-C)
DESCRIPTION: Initializes the `_cameraSource` object, sets the capture session preset to high quality (`AVCaptureSessionPresetHigh`), configures the camera to use the back position, and sets the video orientation to portrait.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_6

LANGUAGE: Objective-C
CODE:
```
-(void)viewDidLoad {
  [super viewDidLoad];

  _cameraSource = [[MPPCameraInputSource alloc] init];
  _cameraSource.sessionPreset = AVCaptureSessionPresetHigh;
  _cameraSource.cameraPosition = AVCaptureDevicePositionBack;
  // The frame's native format is rotated with respect to the portrait orientation.
  _cameraSource.orientation = AVCaptureVideoOrientationPortrait;
}
```

----------------------------------------

TITLE: Configuring Android Application Manifest (XML)
DESCRIPTION: This XML snippet is the `AndroidManifest.xml` file, which declares the application's package, minimum and target SDK versions, and registers `MainActivity` as the main launcher activity. It ensures the app starts correctly and defines basic app properties.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_3

LANGUAGE: XML
CODE:
```
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.google.mediapipe.apps.basic">

  <uses-sdk
      android:minSdkVersion="19"
      android:targetSdkVersion="34" />

  <application
      android:allowBackup="true"
      android:label="${appName}"
      android:supportsRtl="true"
      android:theme="@style/AppTheme">
      <activity
          android:name="${mainActivity}"
          android:exported="true"
          android:screenOrientation="portrait">
          <intent-filter>
              <action android:name="android.intent.action.MAIN" />
              <category android:name="android.intent.category.LAUNCHER" />
          </intent-filter>
      </activity>
  </application>

</manifest>
```

----------------------------------------

TITLE: Adding 3D Point Y-Coordinate (Python/C++)
DESCRIPTION: Adds a list of normalized y values for 3D points in a frame. This function is used to specify the vertical coordinate of a 3D point.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_43

LANGUAGE: Python
CODE:
```
add_bbox_3d_point_y
```

LANGUAGE: C++
CODE:
```
AddBBox3dPointY
```

----------------------------------------

TITLE: Setting SDK Versions in AndroidManifest.xml
DESCRIPTION: This XML configuration sets the minimum SDK version required for the application to run (`minSdkVersion=21`) and the target SDK version it's designed for (`targetSdkVersion=34`) within the `AndroidManifest.xml` file. This ensures compatibility and proper permission handling.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_10

LANGUAGE: xml
CODE:
```
<uses-sdk
    android:minSdkVersion="21"
    android:targetSdkVersion="34" />
```

----------------------------------------

TITLE: Setting Singular Context Feature in MediaPipe MediaSequence (C++)
DESCRIPTION: Sets or updates a singular context feature in a MediaSequence example. The function first clears any existing feature and then stores the new value of the appropriate type. An optional prefix can be used.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_12

LANGUAGE: C++
CODE:
```
SetFeature([const string& prefix,], const TYPE& value, tf::SE* example)
```

----------------------------------------

TITLE: Building MediaPipe Docker Image (Bash)
DESCRIPTION: This snippet clones the MediaPipe repository and builds a Docker image tagged 'mediapipe'. This image encapsulates all necessary dependencies for MediaPipe, providing an isolated environment for development and execution. It requires Docker to be installed on the host system.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_38

LANGUAGE: bash
CODE:
```
$ git clone --depth 1 https://github.com/google/mediapipe.git
$ cd mediapipe
$ docker build --tag=mediapipe .
```

----------------------------------------

TITLE: Adding External Storage Write Permission (Android XML)
DESCRIPTION: This XML snippet adds the `MANAGE_EXTERNAL_STORAGE` permission to the AndroidManifest.xml file, enabling the application to write trace logs to external storage. This is a prerequisite for MediaPipe tracing on Android devices.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/tools/tracing_and_profiling.md#_snippet_1

LANGUAGE: XML
CODE:
```
<uses-permission android:name="android.permission.MANAGE_EXTERNAL_STORAGE" />
```

----------------------------------------

TITLE: Setting Instance Segmentation Image Width - MediaPipe
DESCRIPTION: This snippet demonstrates how to set the width in pixels for the instance segmentation image, providing essential dimension information for processing.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_69

LANGUAGE: Python
CODE:
```
set_instance_segmentation_width
```

LANGUAGE: C++
CODE:
```
SetInstanceSegmentationWidth
```

----------------------------------------

TITLE: Defining Camera Access String Resource (XML)
DESCRIPTION: This XML snippet adds a new string resource named `no_camera_access` to `strings.xml`. This string provides a user-friendly message displayed when the application lacks camera permissions, referenced by the `no_camera_access_view` TextView.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_14

LANGUAGE: XML
CODE:
```
<string name="no_camera_access" translatable="false">Please grant camera permissions.</string>
```

----------------------------------------

TITLE: Building iOS Application with Bazel (Specific Example)
DESCRIPTION: This command provides a concrete example of building the HelloWorldApp located at mediapipe/examples/ios/helloworld using Bazel. It compiles the application in optimized mode for the ios_arm64 architecture, producing an .ipa file for deployment.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_3

LANGUAGE: Shell
CODE:
```
bazel build -c opt --config=ios_arm64 mediapipe/examples/ios/helloworld:HelloWorldApp
```

----------------------------------------

TITLE: Building iOS Application with Bazel (Generic Command)
DESCRIPTION: This command demonstrates the generic syntax for building an iOS application using Bazel. It specifies an optimized compilation (-c opt), targets the ios_arm64 architecture, and requires the application's Bazel target path.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_2

LANGUAGE: Shell
CODE:
```
bazel build -c opt --config=ios_arm64 <$APPLICATION_PATH>:HelloWorldApp'
```

----------------------------------------

TITLE: Declaring Calculator Contract (New Node API, C++)
DESCRIPTION: This snippet shows the new declarative approach to defining a calculator's contract using `MEDIAPIPE_NODE_CONTRACT`. It simplifies the process of specifying required input and output ports compared to procedural setup in `GetContract`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/api2/README.md#_snippet_2

LANGUAGE: C++
CODE:
```
MEDIAPIPE_NODE_CONTRACT(kInput, kOutput);
```

----------------------------------------

TITLE: Downloading MediaPipe Image Classifier Model (Bash)
DESCRIPTION: This snippet demonstrates how to navigate to the MediaPipe image classifier benchmark directory and download a specific EfficientNet Lite0 TFLite model using `wget`. This model is then used for benchmarking.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/benchmark/vision/image_classifier/README.md#_snippet_0

LANGUAGE: Bash
CODE:
```
cd mediapipe/mediapipe/tasks/python/benchmark/vision/image_classifier
wget -O classifier.tflite -q https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite
```

----------------------------------------

TITLE: Setting Dataset Name in MediaPipe Example (Python/C++)
DESCRIPTION: This function sets the name of the dataset, including its version, for a MediaPipe example. The `example/dataset_name` key stores this context bytes type feature, providing metadata about the data source.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_31

LANGUAGE: python
CODE:
```
set_example_dataset_name
```

LANGUAGE: c++
CODE:
```
SetExampleDatasetName
```

----------------------------------------

TITLE: Cloning MediaPipe Repository (Bash)
DESCRIPTION: This snippet clones the MediaPipe GitHub repository and navigates into the newly created directory. This is the initial step required to set up the MediaPipe development environment before building any projects.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/autoflip/README.md#_snippet_0

LANGUAGE: bash
CODE:
```
git clone https://github.com/google/mediapipe.git
cd mediapipe
```

----------------------------------------

TITLE: Adding MPPGraph Property to ViewController in Objective-C
DESCRIPTION: Declares a `nonatomic` property `mediapipeGraph` of type `MPPGraph*` within the `ViewController` interface. This property holds the instance of the MediaPipe graph used for processing, initialized in `viewDidLoad` and started in `viewWillAppear:`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_23

LANGUAGE: Objective-C
CODE:
```
@property(nonatomic) MPPGraph* mediapipeGraph;
```

----------------------------------------

TITLE: Declaring Multiple Untagged Input Ports (New Node API, C++)
DESCRIPTION: This example demonstrates the declarative way to define multiple untagged input ports that can accept any type using `Input<AnyType>::Multiple` and an empty string for the tag.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/api2/README.md#_snippet_6

LANGUAGE: C++
CODE:
```
static constexpr Input<AnyType>::Multiple kIn{"";
```

----------------------------------------

TITLE: Sending Output Packet (New Node API, C++)
DESCRIPTION: This snippet shows the streamlined way to send a payload to an output port using the new Node API. The input timestamp is propagated by default, simplifying the `Send` method call.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/api2/README.md#_snippet_8

LANGUAGE: C++
CODE:
```
kPair(cc).Send({kIn(cc)[0].packet(), kIn(cc)[1].packet()});
```

----------------------------------------

TITLE: Adding Region Track Index (Python/C++)
DESCRIPTION: For each region, adds its integer track ID. If a region has multiple track IDs, the region must be duplicated for each ID.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_54

LANGUAGE: Python
CODE:
```
add_bbox_track_index
```

LANGUAGE: C++
CODE:
```
AddBBoxTrackIndex
```

----------------------------------------

TITLE: Declaring MediaPipe Graph Constants in Objective-C
DESCRIPTION: Defines static constants for the MediaPipe graph name (`mobile_gpu`), and the input (`input_video`) and output (`output_video`) stream names. These constants are used throughout the `ViewController` to refer to specific graph components.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_22

LANGUAGE: Objective-C
CODE:
```
static NSString* const kGraphName = @"mobile_gpu";

static const char* kInputStream = "input_video";
static const char* kOutputStream = "output_video";
```

----------------------------------------

TITLE: Installing OpenCV with Yum (Bash)
DESCRIPTION: This command uses the 'yum' package manager to install the 'opencv-devel' package, providing pre-compiled OpenCV libraries. It's a quick option but might install an older version (2.4.5) with known issues.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_17

LANGUAGE: bash
CODE:
```
$ sudo yum install opencv-devel
```

----------------------------------------

TITLE: Setting Example ID in MediaPipe (Python/C++)
DESCRIPTION: This function sets a unique identifier for a MediaPipe example. The `example/id` key is used to store this context bytes type feature, ensuring each example can be uniquely referenced.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_30

LANGUAGE: python
CODE:
```
set_example_id
```

LANGUAGE: c++
CODE:
```
SetExampleId
```

----------------------------------------

TITLE: Installing Python Requirements for MediaPipe Source Build
DESCRIPTION: This command installs all Python packages listed in the `requirements.txt` file, which are necessary dependencies for building the MediaPipe Python package from its source code within the activated virtual environment.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python.md#_snippet_6

LANGUAGE: Bash
CODE:
```
(mp_env)mediapipe$ pip3 install -r requirements.txt
```

----------------------------------------

TITLE: Getting Default Parser for MediaPipe Context Features (Python)
DESCRIPTION: This Python-only function returns the `tf.io.VarLenFeature` parser suitable for handling variable-length context features. It is used for parsing feature data during TensorFlow operations.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_21

LANGUAGE: python
CODE:
```
get_feature_default_parser()
```

----------------------------------------

TITLE: Retrieving Singular Context Feature in MediaPipe MediaSequence (Python)
DESCRIPTION: Retrieves a single singular context feature from a MediaSequence example. The function returns the feature value, which can be a string, int64, or float. An optional prefix can be provided.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_7

LANGUAGE: Python
CODE:
```
get_feature(example [, prefix])
```

----------------------------------------

TITLE: Simplified Cross-Compilation using Makefile Target
DESCRIPTION: This `make` command provides a simplified way to trigger the cross-compilation process for a specific MediaPipe target. It leverages the predefined rules in the `Makefile` to build the `face_detection_tpu` example for the currently configured Docker environment's platform.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/coral/README.md#_snippet_6

LANGUAGE: bash
CODE:
```
make -C mediapipe/examples/coral \
     BAZEL_TARGET=mediapipe/examples/coral:face_detection_tpu \
     build
```

----------------------------------------

TITLE: Importing MPPLayerRenderer Header in Objective-C
DESCRIPTION: Imports the necessary header file for `MPPLayerRenderer`. This utility class is used to display `CVPixelBufferRef` objects, which are the type of images provided by `MPPCameraInputSource`, on the screen.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_12

LANGUAGE: Objective-C
CODE:
```
#import "mediapipe/objc/MPPLayerRenderer.h"
```

----------------------------------------

TITLE: Defining Video Queue Label Constant (Objective-C)
DESCRIPTION: Defines a static constant `kVideoQueueLabel` as a C string. This label provides a unique identifier for the `_videoQueue`, which can be useful for debugging and organization.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_11

LANGUAGE: Objective-C
CODE:
```
static const char* kVideoQueueLabel = "com.google.mediapipe.example.videoQueue";
```

----------------------------------------

TITLE: Declaring ApplicationInfo in MainActivity (Java)
DESCRIPTION: This Java declaration adds an `ApplicationInfo` member variable to `MainActivity`. This object will be used to retrieve metadata, such as the `cameraFacingFront` setting, that is specified in the `AndroidManifest.xml` and configured in the `BUILD` file.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_22

LANGUAGE: Java
CODE:
```
private ApplicationInfo applicationInfo;
```

----------------------------------------

TITLE: Building MediaPipe Face Detection AAR (Bazel)
DESCRIPTION: This specific Bazel command builds the `mediapipe_face_detection.aar` file, referencing the target defined in the previous step. It uses the same optimization and linking flags as the generic build command, ensuring the AAR is optimized for Android. The output confirms the successful generation of the AAR.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/android_archive_library.md#_snippet_2

LANGUAGE: bash
CODE:
```
bazel build -c opt --strip=ALWAYS \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
    --fat_apk_cpu=arm64-v8a,armeabi-v7a \
    --legacy_whole_archive=0 \
    --features=-legacy_whole_archive \
    --copt=-fvisibility=hidden \
    --copt=-ffunction-sections \
    --copt=-fdata-sections \
    --copt=-fstack-protector \
    --copt=-Oz \
    --copt=-fomit-frame-pointer \
    --copt=-DABSL_MIN_LOG_LEVEL=2 \
    --linkopt=-Wl,--gc-sections,--strip-all \
    //mediapipe/examples/android/src/java/com/google/mediapipe/apps/aar_example:mediapipe_face_detection.aar
```

----------------------------------------

TITLE: Getting Default Feature Parser in MediaPipe MediaSequence (Python)
DESCRIPTION: Returns the tf.io.FixedLenFeature parser object for the singular context feature type. This function is Python-only and is used for parsing features from tf.train.SequenceExample.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_15

LANGUAGE: Python
CODE:
```
get_feature_default_parser()
```

----------------------------------------

TITLE: Running MediaPipe Image Classifier Benchmark (Bazel)
DESCRIPTION: This command executes the default MediaPipe image classifier benchmark using Bazel. The `-c opt` flag optimizes the build for performance, and the target specifies the benchmark script to run.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/benchmark/vision/image_classifier/README.md#_snippet_1

LANGUAGE: Bash
CODE:
```
bazel run -c opt //mediapipe/tasks/python/benchmark/vision/image_classifier:image_classifier_benchmark
```

----------------------------------------

TITLE: Importing MediaPipe and Face Mesh Solution
DESCRIPTION: This Python code imports the MediaPipe library, aliasing it as `mp`, and then accesses the `face_mesh` solution from `mp.solutions`. This prepares the environment to utilize MediaPipe's pre-built face mesh capabilities.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python.md#_snippet_2

LANGUAGE: Python
CODE:
```
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
```

----------------------------------------

TITLE: Copying MediaPipe AAR to Android Project Libs - Bash
DESCRIPTION: This command copies the compiled MediaPipe AAR file, typically generated by Bazel, into the `app/libs` directory of an Android Studio project. This makes the AAR available for inclusion as a local dependency.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/android_archive_library.md#_snippet_4

LANGUAGE: bash
CODE:
```
cp bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/aar_example/mediapipe_face_detection.aar
/path/to/your/app/libs/
```

----------------------------------------

TITLE: Testing the MediaPipe Profiler Reporter - Bazel/Shell
DESCRIPTION: This command runs the unit tests for the `reporter` component of the MediaPipe profiler. It ensures the profiler's functionality is working as expected and validates its internal logic.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/profiler/reporter/README.md#_snippet_1

LANGUAGE: Bazel
CODE:
```
bazel test :reporter_test
```

----------------------------------------

TITLE: Creating a String MediaPipe Packet (UTF-8) in Python
DESCRIPTION: Creates a MediaPipe packet with a UTF-8 encoded string payload (mapped to C++ `std::string`) using `mp.packet_creator.create_string`. This supports text data within the MediaPipe framework.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_24

LANGUAGE: Python
CODE:
```
create_string('abc')
```

----------------------------------------

TITLE: Setting Output Timestamp Bound with SetNextTimestampBound (C++)
DESCRIPTION: This snippet demonstrates how to explicitly set the next timestamp bound for an output stream in a MediaPipe calculator. It uses `SetNextTimestampBound()` on an output tag to specify that the output stream's bound should be `t.NextAllowedInStream()`, allowing downstream calculators to be scheduled promptly.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/realtime_streams.md#_snippet_2

LANGUAGE: C++
CODE:
```
cc->Outputs.Tag("OUT").SetNextTimestampBound(t.NextAllowedInStream());
```

----------------------------------------

TITLE: Creating a 32-bit Signed Integer MediaPipe Packet in Python
DESCRIPTION: Creates a MediaPipe packet with a 32-bit signed integer payload (mapped to C++ `int32_t`) using `mp.packet_creator.create_int32`. This is commonly used for standard integer types.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_8

LANGUAGE: Python
CODE:
```
create_int32(2**31-1)
```

----------------------------------------

TITLE: Downloading YouTube-8M Dataset Shard (Bash)
DESCRIPTION: This snippet downloads a specific shard of the YouTube-8M training dataset (`trainpj.tfrecord`) using `curl` and saves it to a temporary directory. This dataset is required for running inference with the YouTube-8M models.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/examples/desktop/youtube8m/README.md#_snippet_6

LANGUAGE: Bash
CODE:
```
curl http://us.data.yt8m.org/2/frame/train/trainpj.tfrecord --output /tmp/mediapipe/trainpj.tfrecord
```

----------------------------------------

TITLE: Importing MediaPipe Graph Header in Objective-C
DESCRIPTION: Imports the necessary `MPPGraph.h` header file, which provides the core MediaPipe graph functionality for Objective-C applications. This is a prerequisite for using `MPPGraph` objects.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_ios.md#_snippet_21

LANGUAGE: Objective-C
CODE:
```
#import "mediapipe/objc/MPPGraph.h"
```

----------------------------------------

TITLE: Adding Region Generated Status (Python/C++)
DESCRIPTION: For each region, indicates whether it was procedurally generated (1) for the current frame. This helps in understanding the origin of region data.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_49

LANGUAGE: Python
CODE:
```
add_bbox_is_generated
```

LANGUAGE: C++
CODE:
```
AddBBoxIsGenerated
```

----------------------------------------

TITLE: Converting Camera Parameters (px, py) from Pixel to NDC Space - Mathematical
DESCRIPTION: These formulas define how to convert the principal point camera parameters (px_pixel, py_pixel) from pixel space to Normalized Device Coordinates (NDC) space. This transformation adjusts the origin and scale of the principal point to align with the NDC system.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_19

LANGUAGE: mathematical
CODE:
```
px = -px_pixel * 2.0 / image_width  + 1.0
py = -py_pixel * 2.0 / image_height + 1.0
```

----------------------------------------

TITLE: Setting Feature Number of Samples - MediaPipe Audio
DESCRIPTION: This snippet defines the number of audio samples contained within each stored feature. It helps in understanding the granularity of the audio data represented by each feature.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_77

LANGUAGE: Python
CODE:
```
set_feature_num_samples
```

LANGUAGE: C++
CODE:
```
SetFeatureNumSamples
```

----------------------------------------

TITLE: Setting Instance Segmentation Image Height - MediaPipe
DESCRIPTION: This snippet shows how to set the height in pixels for the instance segmentation image, providing essential dimension information for processing.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/util/sequence/README.md#_snippet_68

LANGUAGE: Python
CODE:
```
set_instance_segmentation_height
```

LANGUAGE: C++
CODE:
```
SetInstanceSegmentationHeight
```

----------------------------------------

TITLE: Retrieving an 8-bit Unsigned Integer from a MediaPipe Packet in Python
DESCRIPTION: Retrieves the 8-bit unsigned integer payload (mapped to C++ `uint8_t`) from a MediaPipe packet using `mp.packet_getter.get_uint`. This getter is used for unsigned integer types.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/python_framework.md#_snippet_13

LANGUAGE: Python
CODE:
```
get_uint(packet)
```

----------------------------------------

TITLE: Disabling OpenGL ES Support with Bazel
DESCRIPTION: This command disables OpenGL ES support when building MediaPipe targets using Bazel. It is essential for platforms where OpenGL ES is not available, but should not be used on Android or iOS where it is required.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/gpu_support.md#_snippet_0

LANGUAGE: bash
CODE:
```
$ bazel build --define MEDIAPIPE_DISABLE_GPU=1 <my-target>
```

----------------------------------------

TITLE: Handling NameNotFoundException (Java)
DESCRIPTION: This `catch` block handles `NameNotFoundException`, which occurs if application information cannot be found. It logs an error message using `Log.e` with a predefined `TAG` and the exception details, indicating a failure to retrieve necessary application metadata.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_24

LANGUAGE: Java
CODE:
```
} catch (NameNotFoundException e) {
  Log.e(TAG, "Cannot find application info: " + e);
}
```

----------------------------------------

TITLE: Parsing 3D Object Assets to UUU Format (Bash)
DESCRIPTION: This Bazel command is the second step in asset processing, converting cleaned .obj files from an INTERMEDIATE_OUTPUT_DIR into a single .uuu animation file in the specified OUTPUT_DIR. Both input and output directories must be provided as absolute paths.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_11

LANGUAGE: bash
CODE:
```
bazel run -c opt mediapipe/graphs/object_detection_3d/obj_parser:ObjParser -- input_dir=[INTERMEDIATE_OUTPUT_DIR] output_dir=[OUTPUT_DIR]
```

----------------------------------------

TITLE: Building AutoFlip with Bazel (Linux/macOS)
DESCRIPTION: This command builds the AutoFlip pipeline using Bazel, optimizing the compilation for release. It explicitly disables GPU support, making it suitable for CPU-only environments. Requires OpenCV 3.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/autoflip.md#_snippet_0

LANGUAGE: bash
CODE:
```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/autoflip:run_autoflip
```

----------------------------------------

TITLE: Defining Bazel cc_library for MacPorts FFmpeg
DESCRIPTION: This Bazel `cc_library` rule defines how to link against FFmpeg when installed via MacPorts. It specifies the dynamic libraries, header files, include paths, and required linker options for FFmpeg components, ensuring proper linking for other Bazel targets.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/install.md#_snippet_25

LANGUAGE: BUILD
CODE:
```
cc_library(
    name = "libffmpeg",
    srcs = glob(
        [
            "local/lib/libav*.dylib",
        ],
    ),
    hdrs = glob(["local/include/libav*/*.h"]),
    includes = ["local/include/"],
    linkopts = [
        "-lavcodec",
        "-lavformat",
        "-lavutil",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
```

----------------------------------------

TITLE: Defining Android Application Theme (XML)
DESCRIPTION: This XML snippet defines the application's theme in `styles.xml`. It sets `AppTheme` to inherit from `Theme.AppCompat.Light.DarkActionBar` and customizes it by referencing the colors defined in `colors.xml`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_5

LANGUAGE: XML
CODE:
```
<resources>

    <!-- Base application theme. -->
    <style name="AppTheme" parent="Theme.AppCompat.Light.DarkActionBar">
        <!-- Customize your theme here. -->
        <item name="colorPrimary">@color/colorPrimary</item>
        <item name="colorPrimaryDark">@color/colorPrimaryDark</item>
        <item name="colorAccent">@color/colorAccent</item>
    </style>

</resources>
```

----------------------------------------

TITLE: Declaring Input Ports (Old Node API, C++)
DESCRIPTION: This snippet illustrates the traditional method of declaring and checking for the presence of an input port in MediaPipe using a plain string tag and procedural checks within the CalculatorContext.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/api2/README.md#_snippet_0

LANGUAGE: C++
CODE:
```
constexpr char kSelectTag[] = "SELECT";
if (cc->Inputs().HasTag(kSelectTag)) {
  cc->Inputs().Tag(kSelectTag).Set<int>();
}
```

----------------------------------------

TITLE: Building Two-stage Objectron for Chairs (Android)
DESCRIPTION: This command builds the MediaPipe Objectron Android example for 3D object detection of chairs using Bazel. It targets the ARM64 architecture and activates the chair model via the '--define chair=true' flag.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_5

LANGUAGE: bash
CODE:
```
bazel build -c opt --config android_arm64 --define chair=true mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetection3d:objectdetection3d
```

----------------------------------------

TITLE: Defining Android Color Resources (XML)
DESCRIPTION: This XML snippet defines color resources in `colors.xml` for the Android application. It specifies primary, dark primary, and accent colors, which are used for theming the app, particularly with `Theme.AppCompat`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/getting_started/hello_world_android.md#_snippet_4

LANGUAGE: XML
CODE:
```
<?xml version="1.0" encoding="utf-8"?>
<resources>
    <color name="colorPrimary">#008577</color>
    <color name="colorPrimaryDark">#00574B</color>
    <color name="colorAccent">#D81B60</color>
</resources>
```

----------------------------------------

TITLE: Bad Practice: Defining MediaPipe Graph Inputs Mid-Graph (C++)
DESCRIPTION: This snippet demonstrates an anti-pattern where graph inputs are defined dynamically within the graph builder or helper functions. This approach makes it difficult to ascertain the total number of inputs, is error-prone due to hardcoded indices, and limits the reusability of helper functions like `RunSomething`.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/framework_concepts/building_graphs_cpp.md#_snippet_8

LANGUAGE: C++
CODE:
```
Stream<D> RunSomething(Stream<A> a, Stream<B> b, Graph& graph) {
  Stream<C> c = graph.In(2).SetName("c").Cast<C>();  // Bad.
  // ...
}

CalculatorGraphConfig BuildGraph() {
  Graph graph;

  Stream<A> a = graph.In(0).SetName("a").Cast<A>();
  // 10/100/N lines of code.
  Stream<B> b = graph.In(1).SetName("b").Cast<B>()  // Bad.
  Stream<D> d = RunSomething(a, b, graph);
  // ...

  return graph.GetConfig();
}
```

----------------------------------------

TITLE: MediaPipe Foundational Classes
DESCRIPTION: This snippet enumerates the primary class definitions present in the MediaPipe project, representing essential building blocks for various functionalities.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/calculators/tensor/testdata/labelmap.txt#_snippet_0

LANGUAGE: APIDOC
CODE:
```
classA
classB
classC
```

----------------------------------------

TITLE: Accessing Input Packet by Tag (Old Node API, C++)
DESCRIPTION: This example demonstrates the traditional way to retrieve an input packet from the `CalculatorContext` (`cc`) by its string tag, requiring explicit type specification.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/api2/README.md#_snippet_3

LANGUAGE: C++
CODE:
```
int select = cc->Inputs().Tag(kSelectTag).Get<int>();
```

----------------------------------------

TITLE: Setting Any Type for Multiple Untagged Inputs (Old Node API, C++)
DESCRIPTION: This snippet shows the procedural loop used in the old API to iterate through and set any type for multiple untagged input ports, typically in the `GetContract` method.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/api2/README.md#_snippet_5

LANGUAGE: C++
CODE:
```
for (int i = 0; i < cc->Inputs().NumEntries(); ++i) {
  cc->Inputs().Index(i).SetAny();
}
```

----------------------------------------

TITLE: Sending Output Packet (Old Node API, C++)
DESCRIPTION: This code illustrates the traditional method of adding a new packet to an output stream. It involves manually allocating a new object and explicitly providing the input timestamp.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/api2/README.md#_snippet_7

LANGUAGE: C++
CODE:
```
cc->Outputs().Index(0).Add(
    new std::pair<Packet, Packet>(cc->Inputs().Index(0).Value(),
                                  cc->Inputs().Index(1).Value()),
    cc->InputTimestamp());
```

----------------------------------------

TITLE: Projecting 3D Points from Camera to NDC Space - Mathematical
DESCRIPTION: These equations define the projection of 3D points from the camera's coordinate system (X, Y, Z) into Normalized Device Coordinates (NDC). The camera parameters (fx, fy, px, py) are defined in NDC space, and 'Z' represents the depth component.
SOURCE: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/objectron.md#_snippet_15

LANGUAGE: mathematical
CODE:
```
x_ndc = -fx * X / Z + px
y_ndc = -fy * Y / Z + py
z_ndc = 1 / Z
```

TITLE: Example Text Generation - LiteRT LLM Pipeline - Python
DESCRIPTION: This example shows how to use the initialized LiteRT LLM pipeline to generate text. It defines a prompt and then calls the 'generate' method on the pipeline, specifying a maximum number of decode steps to control the length of the generated output.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_29

LANGUAGE: python
CODE:
```
prompt = "What is the primary function of mitochondria within a cell"
output = pipeline.generate(prompt, max_decode_steps = 100)
```

----------------------------------------

TITLE: Performing Pose Landmark Detection and Visualization with MediaPipe Tasks
DESCRIPTION: This code block orchestrates the entire pose landmark detection process. It initializes a `PoseLandmarker` object using the pre-trained model, loads the input image, performs the detection, and then visualizes the results by drawing the detected landmarks onto the image using the `draw_landmarks_on_image` utility function.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/python/[MediaPipe_Python_Tasks]_Pose_Landmarker.ipynb#_snippet_5

LANGUAGE: python
CODE:
```
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("image.jpg")

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
```

----------------------------------------

TITLE: Performing Face Detection and Visualization with MediaPipe Python
DESCRIPTION: This snippet demonstrates how to perform face detection using MediaPipe's Python API. It involves importing necessary modules, initializing a `FaceDetector` with a TFLite model (`detector.tflite`), loading an image from `IMAGE_FILE`, running the detection, and then visualizing the results using OpenCV functions like `cv2.cvtColor` and `cv2_imshow`.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_detector/python/face_detector.ipynb#_snippet_6

LANGUAGE: python
CODE:
```
# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path='detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file(IMAGE_FILE)

# STEP 4: Detect faces in the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
cv2_imshow(rgb_annotated_image)
```

----------------------------------------

TITLE: Performing Text Classification with MediaPipe Tasks in Python
DESCRIPTION: This code block demonstrates the complete workflow for classifying text using MediaPipe Tasks. It imports necessary modules, creates a `TextClassifier` instance from a TFLite model, performs classification on `INPUT_TEXT`, and then processes the result to print the top predicted category and its score.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/text_classification/python/text_classifier.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
from mediapipe.tasks import python
from mediapipe.tasks.python import text

# STEP 2: Create an TextClassifier object.
base_options = python.BaseOptions(model_asset_path="classifier.tflite")
options = text.TextClassifierOptions(base_options=base_options)
classifier = text.TextClassifier.create_from_options(options)

# STEP 3: Classify the input text.
classification_result = classifier.classify(INPUT_TEXT)

# STEP 4: Process the classification result. In this case, print out the most likely category.
top_category = classification_result.classifications[0].categories[0]
print(f'{top_category.category_name} ({top_category.score:.2f})')
```

----------------------------------------

TITLE: Performing Audio Classification with MediaPipe Tasks - Python
DESCRIPTION: This comprehensive snippet demonstrates end-to-end audio classification using MediaPipe Tasks. It initializes the `AudioClassifier` with the downloaded YAMNet model, reads the WAV audio data, segments it into clips, performs inference, and then iterates through the classification results to print the top category and its score for specific timestamps. It requires `numpy`, `scipy.io.wavfile`, and MediaPipe Tasks.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/audio_classifier/python/audio_classification.ipynb#_snippet_5

LANGUAGE: python
CODE:
```
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
from scipy.io import wavfile

# Customize and associate model for Classifier
base_options = python.BaseOptions(model_asset_path='classifier.tflite')
options = audio.AudioClassifierOptions(
    base_options=base_options, max_results=4)

# Create classifier, segment audio clips, and classify
with audio.AudioClassifier.create_from_options(options) as classifier:
  sample_rate, wav_data = wavfile.read(audio_file_name)
  audio_clip = containers.AudioData.create_from_array(
      wav_data.astype(float) / np.iinfo(np.int16).max, sample_rate)
  classification_result_list = classifier.classify(audio_clip)

  assert(len(classification_result_list) == 5)

# Iterate through clips to display classifications
  for idx, timestamp in enumerate([0, 975, 1950, 2925]):
    classification_result = classification_result_list[idx]
    top_category = classification_result.classifications[0].categories[0]
    print(f'Timestamp {timestamp}: {top_category.category_name} ({top_category.score:.2f})')
```

----------------------------------------

TITLE: Initiating Supervised Fine-Tuning (SFT) - TRL & Transformers - Python
DESCRIPTION: This snippet initializes and starts the supervised fine-tuning process using `SFTTrainer` from the TRL library. It configures training arguments such as batch size, learning rate, and optimization strategy, and integrates the previously defined LoRA configuration. The `trainer.train()` method then commences the fine-tuning on the prepared dataset.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_8

LANGUAGE: Python
CODE:
```
import transformers
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset = ds['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=150,
        #num_train_epochs=1,
        # Copied from other hugging face tuning blog posts
        learning_rate=2e-4,
        #fp16=True,
        bf16=True,
        # It makes training faster
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        report_to = "none",
    ),
    peft_config=lora_config,
)
trainer.train()
```

----------------------------------------

TITLE: Loading Gemma-3-1B Model and Tokenizer (Python)
DESCRIPTION: This snippet loads the `google/gemma-3-1b-pt` model and its corresponding tokenizer from HuggingFace. It configures the model for `bfloat16` precision and sets up a custom chat template for the tokenizer, ensuring proper formatting for conversational inputs and outputs. The HuggingFace token is used for authentication during model and tokenizer download.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
import os

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, GemmaTokenizer
from transformers.models.gemma3 import Gemma3ForCausalLM

model_id = 'google/gemma-3-1b-pt'
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
model = Gemma3ForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", token=os.environ['HF_TOKEN'], attn_implementation='eager')
# Set up the chat format
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
```

----------------------------------------

TITLE: Initializing MediaPipe ImageClassifier - Python
DESCRIPTION: This code initializes an `ImageClassifier` object. It sets up `BaseOptions` by specifying the path to the TFLite model (`classifier.tflite`) and then configures `ImageClassifierOptions` to limit the classification results to a maximum of 4. The `classifier` object is then created using these options, making it ready for image classification.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_classification/python/image_classifier.ipynb#_snippet_8

LANGUAGE: python
CODE:
```
base_options = python.BaseOptions(model_asset_path='classifier.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=4)
classifier = vision.ImageClassifier.create_from_options(options)
```

----------------------------------------

TITLE: Initiating Object Detector Model Retraining (Python)
DESCRIPTION: This code initiates the retraining process for the object detection model using the `create()` method of `object_detector.ObjectDetector`. It takes the prepared training and validation datasets, along with the previously configured `options`, to start the resource-intensive training. The resulting `model` object represents the trained model.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_10

LANGUAGE: python
CODE:
```
model = object_detector.ObjectDetector.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options)
```

----------------------------------------

TITLE: Loading and Tokenizing SFT Dataset - Hugging Face Datasets - Python
DESCRIPTION: This snippet loads a supervised fine-tuning (SFT) dataset from Hugging Face and prepares it for training. It defines a `tokenize_function` to format prompt-completion pairs into a chat template, then applies this function to the dataset in a batched manner, creating a 'text' column suitable for language model training.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_7

LANGUAGE: Python
CODE:
```
from datasets import load_dataset

ds = load_dataset("argilla/synthetic-concise-reasoning-sft-filtered")
def tokenize_function(examples):
    # Process all examples in the batch
    prompts = examples["prompt"]
    completions = examples["completion"]
    texts = []
    for prompt, completion in zip(prompts, completions):
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt.strip()}, {"role": "assistant", "content": completion.strip()}], tokenize=False)
        texts.append(text)
    return { "text" : texts }  # Return a list of texts

ds = ds.map(tokenize_function, batched = True)
```

----------------------------------------

TITLE: Performing Text Embedding and Similarity Calculation with MediaPipe - Python
DESCRIPTION: This snippet demonstrates how to initialize the MediaPipe `TextEmbedder` with a downloaded model and then use it to generate embeddings for two text strings. It subsequently calculates and prints the cosine similarity between these embeddings, indicating how semantically similar the texts are. Key parameters include `l2_normalize` and `quantize` for embedding options.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/text_embedder/python/text_embedder.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
from mediapipe.tasks import python
from mediapipe.tasks.python import text

# Create your base options with the model that was downloaded earlier
base_options = python.BaseOptions(model_asset_path='embedder.tflite')

# Set your values for using normalization and quantization
l2_normalize = True #@param {type:"boolean"}
quantize = False #@param {type:"boolean"}

# Create the final set of options for the Embedder
options = text.TextEmbedderOptions(
    base_options=base_options, l2_normalize=l2_normalize, quantize=quantize)

with text.TextEmbedder.create_from_options(options) as embedder:
  # Retrieve the first and second sets of text that will be compared
  first_text = "I'm feeling so good" #@param {type:"string"}
  second_text = "I'm okay I guess" #@param {type:"string"}

  # Convert both sets of text to embeddings
  first_embedding_result = embedder.embed(first_text)
  second_embedding_result = embedder.embed(second_text)

  # Calculate and print similarity
  similarity = text.TextEmbedder.cosine_similarity(
      first_embedding_result.embeddings[0],
      second_embedding_result.embeddings[0])
  print(similarity)
```

----------------------------------------

TITLE: Initializing LiteRT LLM Pipeline
DESCRIPTION: This snippet initializes a `LiteRTLlmPipeline` instance, which is essential for interacting with the loaded language model. It typically requires an `interpreter` for model execution and a `tokenizer` for text processing.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma3_1b_tflite.ipynb#_snippet_3

LANGUAGE: Python
CODE:
```
# Disclaimer: Model performance demonstrated with the Python API in this notebook is not representative of performance on a local device.
pipeline = LiteRTLlmPipeline(interpreter, tokenizer)
```

----------------------------------------

TITLE: Initializing LiteRT LLM Pipeline - Python
DESCRIPTION: This line initializes an instance of the `LiteRTLlmPipeline` class, which is used to manage the LLM inference process. It requires an `interpreter` (likely a TFLite interpreter) and a `tokenizer` object to handle model execution and text tokenization, respectively. This pipeline object will then be used for text generation.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma2_tflite.ipynb#_snippet_14

LANGUAGE: Python
CODE:
```
pipeline = LiteRTLlmPipeline(interpreter, tokenizer)
```

----------------------------------------

TITLE: Converting Gemma Model to LiteRT Format (Python)
DESCRIPTION: This comprehensive script converts a PyTorch Gemma 1B model to the LiteRT format, including 8-bit quantization. It defines conversion flags, creates attention masks for prefill and decode stages, and uses `ai_edge_torch.generative.utilities.converter` to export the model as a TFLite file with specified quantization and KV cache settings.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_14

LANGUAGE: Python
CODE:
```
from absl import flags
from absl import app
import sys
import torch

from ai_edge_torch.generative.examples.gemma3 import gemma3
from ai_edge_torch.generative.layers import kv_cache
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities.export_config import ExportConfig


flags = converter.define_conversion_flags('gemma3-1b')
flags.FLAGS.mask_as_input = True
flags.FLAGS.prefill_seq_lens = [128]
flags.FLAGS.kv_cache_max_len = 1024

def _create_mask(mask_len, kv_cache_max_len):
  mask = torch.full(
      (mask_len, kv_cache_max_len), float('-inf'), dtype=torch.float32
  )
  mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
  return mask


def _create_export_config(
    prefill_seq_lens: list[int], kv_cache_max_len: int
) -> ExportConfig:
  """Creates the export config for the model."""
  export_config = ExportConfig()
  if isinstance(prefill_seq_lens, list):
    prefill_mask = [_create_mask(i, kv_cache_max_len) for i in prefill_seq_lens]
  else:
    prefill_mask = _create_mask(prefill_seq_lens, kv_cache_max_len)

  export_config.prefill_mask = prefill_mask

  decode_mask = torch.full(
      (1, kv_cache_max_len), float('-inf'), dtype=torch.float32
  )
  decode_mask = torch.triu(decode_mask, diagonal=1).unsqueeze(0).unsqueeze(0)
  export_config.decode_mask = decode_mask
  export_config.kvcache_layout = kv_cache.KV_LAYOUT_TRANSPOSED
  return export_config


def convert_to_litert(_):
  with torch.inference_mode(True):
    pytorch_model = gemma3.build_model_1b(
      "/content/merged_model", kv_cache_max_len=flags.FLAGS.kv_cache_max_len,
    )
    converter.convert_to_tflite(
        pytorch_model,
        output_path="/content/",
        output_name_prefix="gemma3_1b_finetune",
        prefill_seq_len=flags.FLAGS.prefill_seq_lens,
        quantize=converter.QuantizationName.DYNAMIC_INT4_BLOCK32,
        lora_ranks=None,
        export_config=_create_export_config(
            prefill_seq_lens=flags.FLAGS.prefill_seq_lens,
            kv_cache_max_len=flags.FLAGS.kv_cache_max_len,
        ),
    )

# Ignore flags passed from the colab runtime.
sys.argv = sys.argv[:1]
app.run(convert_to_litert)
```

----------------------------------------

TITLE: Generating Text with LiteRT LLM Pipeline
DESCRIPTION: This code demonstrates how to use the loaded LiteRT model to generate text. It takes a `prompt` as input and uses the `runner.generate` method to produce an `output`, with an option to specify maximum decoding steps.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma3_1b_tflite.ipynb#_snippet_4

LANGUAGE: Python
CODE:
```
prompt = "What is the capital of France?"
output = runner.generate(prompt, max_decode_steps=None)
```

----------------------------------------

TITLE: Exporting Retrained MediaPipe Model to TensorFlow Lite (Python)
DESCRIPTION: This command exports the retrained MediaPipe model to the TensorFlow Lite format, which is necessary for deployment in applications. The export process also generates required model metadata and a classification label file.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb#_snippet_10

LANGUAGE: python
CODE:
```
model.export_model()
```

----------------------------------------

TITLE: Exporting and Downloading TensorFlow Lite Model
DESCRIPTION: This snippet exports the trained object detection model to a TensorFlow Lite ('.tflite') format, lists the contents of the 'exported_model' directory to confirm the file's presence, and then initiates the download of the 'dogs.tflite' model file for on-device application use.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/tutorials/object_detection/Object_Detection_for_3_dogs.ipynb#_snippet_7

LANGUAGE: python
CODE:
```
model.export_model('dogs.tflite')
!ls exported_model
files.download('exported_model/dogs.tflite')
```

----------------------------------------

TITLE: Evaluating Object Detector Model Performance (Python)
DESCRIPTION: This snippet evaluates the performance of the retrained object detection model using the `evaluate()` method on the validation dataset. It calculates and prints the validation loss and COCO metrics, with 'AP' (Average Precision) being a key indicator of model performance. This step helps assess the model's accuracy and generalization capabilities.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_11

LANGUAGE: python
CODE:
```
loss, coco_metrics = model.evaluate(validation_data, batch_size=4)
print(f"Validation loss: {loss}")
print(f"Validation coco metrics: {coco_metrics}")
```

----------------------------------------

TITLE: Evaluating a Trained Image Classifier Model (Python)
DESCRIPTION: This code snippet shows how to evaluate the performance of a previously trained `ImageClassifier` model (`model_2`) using a `test_data` dataset. It returns the `loss` and `accuracy` metrics, providing insights into the model's generalization capabilities.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb#_snippet_13

LANGUAGE: python
CODE:
```
loss, accuracy = model_2.evaluate(test_data)
```

----------------------------------------

TITLE: Generating Text with Prefill and Decode - LiteRT LLM Pipeline - Python
DESCRIPTION: This is the main entry point for text generation, orchestrating the entire process. It tokenizes the input prompt, initializes the prefill runner, executes the prefill stage to get the initial KV cache, and then calls the decode method to generate the remaining text based on the prefilled context.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_27

LANGUAGE: python
CODE:
```
  def generate(self, prompt: str, max_decode_steps: int | None = None) -> str:
    messages=[{ 'role': 'user', 'content': prompt}]
    token_ids = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    # Initialize the prefill runner with the suitable input size.
    self._init_prefill_runner(len(token_ids))

    # Run prefill.
    # Prefill up to the seond to the last token of the prompt, because the last
    # token of the prompt will be used to bootstrap decode.
    prefill_token_length = len(token_ids) - 1

    print('Running prefill')
    kv_cache = self._run_prefill(token_ids[:prefill_token_length])
    # Run decode.
    print('Running decode')
    actual_max_decode_steps = self._max_kv_cache_seq_len - prefill_token_length - 1
    if max_decode_steps is not None:
      actual_max_decode_steps = min(actual_max_decode_steps, max_decode_steps)
    decode_text = self._run_decode(
        prefill_token_length,
        token_ids[prefill_token_length],
        kv_cache,
        actual_max_decode_steps,
    )
    return decode_text
```

----------------------------------------

TITLE: Training Image Classifier with Custom HParams and Model Options (Python)
DESCRIPTION: This snippet demonstrates how to train an `ImageClassifier` model with custom hyperparameters and model options. It sets the number of training epochs to 15 and a dropout rate of 0.07, overriding the default values to potentially improve model performance. It requires `train_data`, `validation_data`, and a `spec` for the supported model.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb#_snippet_12

LANGUAGE: python
CODE:
```
hparams=image_classifier.HParams(epochs=15, export_dir="exported_model_2")
options = image_classifier.ImageClassifierOptions(supported_model=spec, hparams=hparams)
options.model_options = image_classifier.ModelOptions(dropout_rate = 0.07)
model_2 = image_classifier.ImageClassifier.create(
    train_data = train_data,
    validation_data = validation_data,
    options=options,
)
```

----------------------------------------

TITLE: Invoking LLM Text Generation - Python
DESCRIPTION: This snippet demonstrates how to use the initialized `LiteRTLlmPipeline` to generate text. It sets a `prompt` string and then calls the `generate` method on the `pipeline` object, passing the prompt and setting `max_decode_steps` to `None` for potentially unlimited decoding. The generated text is stored in the `output` variable.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma2_tflite.ipynb#_snippet_15

LANGUAGE: Python
CODE:
```
prompt = "what is 8 mod 6"
output = pipeline.generate(prompt, max_decode_steps = None)
```

----------------------------------------

TITLE: Testing Fine-Tuned Model Inference - Hugging Face Transformers - Python
DESCRIPTION: This snippet demonstrates how to perform text generation inference using the fine-tuned model. It re-initializes a text generation pipeline with the updated model and tokenizer, then generates a response to a given prompt, showcasing the model's improved capabilities after fine-tuning.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_10

LANGUAGE: Python
CODE:
```
from transformers import pipeline
# Let's test the base model before training
prompt = "What is the primary function of mitochondria within a cell?"
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
pipe(prompt, max_new_tokens=100)
```

----------------------------------------

TITLE: Exporting Quantized MediaPipe Model (Python)
DESCRIPTION: Exports the trained model to a TensorFlow Lite (`.tflite`) file, applying the specified post-training quantization. The `model_name` parameter sets the output filename, and the `quantization_config` object dictates the quantization strategy, such as `int8` quantization.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb#_snippet_16

LANGUAGE: Python
CODE:
```
model.export_model(model_name="model_int8.tflite", quantization_config=quantization_config)
```

----------------------------------------

TITLE: Exporting Quantized BERT Text Classifier and Labels (Python)
DESCRIPTION: This snippet exports the BERT-based text classifier as a TFLite model with dynamic range quantization to reduce its size. It imports `quantization`, creates a `QuantizationConfig`, and then uses `bert_model.export_model()` with this configuration, also exporting the labels.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/text_classifier.ipynb#_snippet_12

LANGUAGE: Python
CODE:
```
from mediapipe_model_maker import quantization
quantization_config = quantization.QuantizationConfig.for_dynamic()
bert_model.export_model(quantization_config=quantization_config)
bert_model.export_labels(export_dir=options.hparams.export_dir)
```

----------------------------------------

TITLE: Exporting Retrained Face Stylizer Model (Python)
DESCRIPTION: This snippet exports the retrained face stylizer model to the TensorFlow Lite format, which is essential for integrating it into applications using MediaPipe. The export process automatically generates all required model metadata and a classification label file, preparing the model for deployment.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/face_stylizer.ipynb#_snippet_9

LANGUAGE: python
CODE:
```
face_stylizer_model.export_model()
```

----------------------------------------

TITLE: Running Image Classifier Retraining with MediaPipe (Python)
DESCRIPTION: This code initiates the image classifier retraining process using the `create()` method of `ImageClassifier`. It requires prepared training and validation datasets, along with the previously defined retraining options. This process can be resource-intensive and its duration depends on compute resources.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb#_snippet_8

LANGUAGE: python
CODE:
```
model = image_classifier.ImageClassifier.create(
    train_data = train_data,
    validation_data = validation_data,
    options=options,
)
```

----------------------------------------

TITLE: Installing MediaPipe Library - Python
DESCRIPTION: This command installs the MediaPipe library using pip, Python's package installer. The '-q' flag ensures a quiet installation, suppressing detailed output, which is useful for cleaner notebook execution. This step is a prerequisite for utilizing any MediaPipe functionalities.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/object_detection/python/object_detector.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
!pip install -q mediapipe
```

----------------------------------------

TITLE: Installing MediaPipe Model Maker Package
DESCRIPTION: This code block installs and upgrades the `pip` package manager, then installs the `mediapipe-model-maker` library, which is essential for customizing on-device ML models.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/text_classifier.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
!pip install --upgrade pip
!pip install mediapipe-model-maker
```

----------------------------------------

TITLE: Installing MediaPipe Model Maker Library
DESCRIPTION: This command installs the `mediapipe-model-maker` Python package, which is essential for customizing on-device machine learning models, including the face stylizer. It's a prerequisite for running the subsequent model customization steps.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/face_stylizer.ipynb#_snippet_1

LANGUAGE: Python
CODE:
```
!pip install mediapipe-model-maker
```

----------------------------------------

TITLE: Running MediaPipe Gesture Recognition Inference (Python)
DESCRIPTION: This snippet demonstrates the core steps for performing gesture recognition using MediaPipe. It initializes a `GestureRecognizer` with a specified model, loads input images, performs recognition, and extracts the top gesture and hand landmarks from the results. It requires the `mediapipe` library and a `gesture_recognizer.task` model file. The final line `display_batch_of_images_with_gestures_and_hand_landmarks` implies a visualization function not included in the snippet.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/gesture_recognizer/python/gesture_recognizer.ipynb#_snippet_7

LANGUAGE: Python
CODE:
```
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

images = []
results = []
for image_file_name in IMAGE_FILENAMES:
  # STEP 3: Load the input image.
  image = mp.Image.create_from_file(image_file_name)

  # STEP 4: Recognize gestures in the input image.
  recognition_result = recognizer.recognize(image)

  # STEP 5: Process the result. In this case, visualize it.
  images.append(image)
  top_gesture = recognition_result.gestures[0][0]
  hand_landmarks = recognition_result.hand_landmarks
  results.append((top_gesture, hand_landmarks))

display_batch_of_images_with_gestures_and_hand_landmarks(images, results)
```

----------------------------------------

TITLE: Applying Background Blur with MediaPipe ImageSegmenter in Python
DESCRIPTION: This snippet demonstrates how to initialize a MediaPipe ImageSegmenter, load images, perform segmentation to obtain category masks, convert image color spaces using OpenCV, apply a Gaussian blur to the background, and combine the original and blurred images based on the segmentation mask. It requires MediaPipe, OpenCV (cv2), and NumPy (np) libraries.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_segmentation/python/image_segmentation.ipynb#_snippet_6

LANGUAGE: Python
CODE:
```
# Create the segmenter
with python.vision.ImageSegmenter.create_from_options(options) as segmenter:

  # Loop through available image(s)
  for image_file_name in IMAGE_FILENAMES:

    # Create the MediaPipe Image
    image = mp.Image.create_from_file(image_file_name)

    # Retrieve the category masks for the image
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask

    # Convert the BGR image to RGB
    image_data = cv2.cvtColor(image.numpy_view(), cv2.COLOR_BGR2RGB)

    # Apply effects
    blurred_image = cv2.GaussianBlur(image_data, (55,55), 0)
    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
    output_image = np.where(condition, image_data, blurred_image)

    print(f'Blurred background of {image_file_name}:')
    resize_and_show(output_image)
```

----------------------------------------

TITLE: Loading Object Detection Datasets from Pascal VOC
DESCRIPTION: This snippet loads the training and validation datasets for object detection using 'object_detector.Dataset.from_pascal_voc_folder'. It specifies the folder paths for 'dogs copy/train' and 'dogs copy/validate' and sets cache directories for efficient data handling during model training.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/tutorials/object_detection/Object_Detection_for_3_dogs.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
train_data = object_detector.Dataset.from_pascal_voc_folder(
    'dogs copy/train',
    cache_dir="/tmp/od_data/train",
)

val_data = object_detector.Dataset.from_pascal_voc_folder(
    'dogs copy/validate',
    cache_dir="/tmp/od_data/validatation")
```

----------------------------------------

TITLE: Training MediaPipe Object Detector Model
DESCRIPTION: This snippet configures and trains an object detection model using MediaPipe Model Maker. It sets hyperparameters like 'batch_size=8', 'learning_rate=0.3', and 'epochs=50', specifies 'MOBILENET_V2' as the supported model, and then creates and trains the 'ObjectDetector' using the prepared training and validation data.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/tutorials/object_detection/Object_Detection_for_3_dogs.ipynb#_snippet_5

LANGUAGE: python
CODE:
```
hparams = object_detector.HParams(batch_size=8, learning_rate=0.3, epochs=50, export_dir='exported_model')
options = object_detector.ObjectDetectorOptions(
    supported_model=object_detector.SupportedModels.MOBILENET_V2,
    hparams=hparams
)
model = object_detector.ObjectDetector.create(
    train_data=train_data,
    validation_data=val_data,
    options=options)
```

----------------------------------------

TITLE: Training BERT Text Classifier (Python)
DESCRIPTION: This snippet initiates the training of the BERT-based text classifier using `TextClassifier.create()`. It utilizes the `train_data`, `validation_data`, and the `options` configured for the MobileBERT model. Note that this process can be computationally intensive.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/text_classifier.ipynb#_snippet_10

LANGUAGE: Python
CODE:
```
bert_model = text_classifier.TextClassifier.create(train_data, validation_data, options)
```

----------------------------------------

TITLE: Exporting Object Detector Model to TensorFlow Lite (Python)
DESCRIPTION: This code exports the trained object detection model to the TensorFlow Lite format, including essential metadata like the label map. It first calls `export_model()` to perform the conversion, then lists the contents of the `exported_model` directory, and finally downloads the generated `model.tflite` file. This prepares the model for on-device deployment.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_12

LANGUAGE: python
CODE:
```
model.export_model()
!ls exported_model
files.download('exported_model/model.tflite')
```

----------------------------------------

TITLE: Performing Language Detection Inference with MediaPipe Python API
DESCRIPTION: This comprehensive snippet demonstrates the full workflow for language detection using MediaPipe. It imports required modules, initializes the LanguageDetector with the downloaded model, performs detection on INPUT_TEXT, and then iterates through the results to print detected languages and their confidence scores. It outlines the core steps for using the MediaPipe Text API.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/language_detector/python/[MediaPipe_Python_Tasks]_Language_Detector.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
# STEP 1: Import the necessary modules.
from mediapipe.tasks import python
from mediapipe.tasks.python import text

# STEP 2: Create a LanguageDetector object.
base_options = python.BaseOptions(model_asset_path="detector.tflite")
options = text.LanguageDetectorOptions(base_options=base_options)
detector = text.LanguageDetector.create_from_options(options)

# STEP 3: Get the language detcetion result for the input text.
detection_result = detector.detect(INPUT_TEXT)

# STEP 4: Process the detection result and print the languages detected and
# their scores.

for detection in detection_result.detections:
  print(f'{detection.language_code}: ({detection.probability:.2f})')
```

----------------------------------------

TITLE: Generating Text with LLM Pipeline - Python
DESCRIPTION: This method orchestrates the text generation process for an LLM. It first tokenizes the input prompt, then performs a prefill step to process the initial prompt and generate a KV cache. Finally, it calls the `_run_decode` method to generate the response tokens, returning the complete decoded text. It takes a `prompt` string and an optional `max_decode_steps` integer.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma2_tflite.ipynb#_snippet_13

LANGUAGE: Python
CODE:
```
def generate(self, prompt: str, max_decode_steps: int | None = None) -> str:
    messages=[{ 'role': 'user', 'content': prompt}]
    token_ids = self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    # Initialize the prefill runner with the suitable input size.
    self._init_prefill_runner(len(token_ids))

    # Run prefill.
    # Prefill up to the seond to the last token of the prompt, because the last
    # token of the prompt will be used to bootstrap decode.
    prefill_token_length = len(token_ids) - 1

    print('Running prefill')
    kv_cache = self._run_prefill(token_ids[:prefill_token_length])
    # Run decode.
    print('Running decode')
    actual_max_decode_steps = self._max_kv_cache_seq_len - prefill_token_length - 1
    if max_decode_steps is not None:
      actual_max_decode_steps = min(actual_max_decode_steps, max_decode_steps)
    decode_text = self._run_decode(
        prefill_token_length,
        token_ids[prefill_token_length],
        kv_cache,
        actual_max_decode_steps,
    )
    return decode_text
```

----------------------------------------

TITLE: Creating a MediaPipe Task Bundle for a TFLite Model
DESCRIPTION: This Python snippet demonstrates how to create a task bundle for a converted TFLite model using the `mediapipe.tasks.python.genai.bundler`. It configures the `BundleConfig` with paths to the TFLite model and tokenizer, start/stop tokens, output filename, and an option for bytes-to-unicode mapping, then calls `create_bundle`.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/bundling/llm_bundling.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
tflite_model="PATH/gemma.tflite" # @param {type:"string"}
tokenizer_model="PATH/tokenizer.model" # @param {type:"string"}
start_token="<bos>" # @param {type:"string"}
stop_token="<eos>" # @param {type:"string"}
output_filename="PATH/gemma.task" # @param {type:"string"}
enable_bytes_to_unicode_mapping=False # @param ["False", "True"] {type:"raw"}

config = bundler.BundleConfig(
    tflite_model=tflite_model,
    tokenizer_model=tokenizer_model,
    start_token=start_token,
    stop_tokens=[stop_token],
    output_filename=output_filename,
    enable_bytes_to_unicode_mapping=enable_bytes_to_unicode_mapping,
)
bundler.create_bundle(config)
```

----------------------------------------

TITLE: Performing Object Detection Inference with MediaPipe - Python
DESCRIPTION: This code demonstrates the complete process of running object detection using MediaPipe. It initializes an `ObjectDetector` with a pre-trained model, loads an image, performs the detection, and then visualizes the results by drawing bounding boxes and labels on a copy of the original image, finally displaying the annotated image.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/object_detection/python/object_detector.ipynb#_snippet_6

LANGUAGE: python
CODE:
```
# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file(IMAGE_FILE)

# STEP 4: Detect objects in the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
cv2_imshow(rgb_annotated_image)
```

----------------------------------------

TITLE: Performing Image Embedding and Cosine Similarity (MediaPipe Python)
DESCRIPTION: This snippet demonstrates how to initialize the MediaPipe `ImageEmbedder` with a downloaded model and custom options (L2 normalization, quantization). It then uses the embedder to generate embeddings for two images and calculates their cosine similarity, which indicates how alike the images are.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_embedder/python/image_embedder.ipynb#_snippet_5

LANGUAGE: Python
CODE:
```
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Create options for Image Embedder
base_options = python.BaseOptions(model_asset_path='embedder.tflite')
l2_normalize = True #@param {type:"boolean"}
quantize = True #@param {type:"boolean"}
options = vision.ImageEmbedderOptions(
    base_options=base_options, l2_normalize=l2_normalize, quantize=quantize)


# Create Image Embedder
with vision.ImageEmbedder.create_from_options(options) as embedder:

  # Format images for MediaPipe
  first_image = mp.Image.create_from_file(IMAGE_FILENAMES[0])
  second_image = mp.Image.create_from_file(IMAGE_FILENAMES[1])
  first_embedding_result = embedder.embed(first_image)
  second_embedding_result = embedder.embed(second_image)

  # Calculate and print similarity
  similarity = vision.ImageEmbedder.cosine_similarity(
      first_embedding_result.embeddings[0],
      second_embedding_result.embeddings[0])
  print(similarity)
```

----------------------------------------

TITLE: Evaluating MediaPipe Gesture Recognizer Model Performance (Python)
DESCRIPTION: This snippet evaluates the performance of the trained gesture recognizer model using the `test_data` and a specified `batch_size`. It returns the loss and accuracy metrics, which are then printed to the console, providing an assessment of the model's generalization ability.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/gesture_recognizer.ipynb#_snippet_8

LANGUAGE: python
CODE:
```
loss, acc = model.evaluate(test_data, batch_size=1)
print(f"Test loss:{loss}, Test accuracy:{acc}")
```

----------------------------------------

TITLE: Running Face Stylizer Inference - Python
DESCRIPTION: This code initializes the MediaPipe `FaceStylizer` using the downloaded model and applies the pre-trained style to the input image. It demonstrates how to perform inference using the `stylize` method and then displays the resulting stylized image.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_stylizer/python/face_stylizer.ipynb#_snippet_5

LANGUAGE: Python
CODE:
```
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Create the options that will be used for FaceStylizer
base_options = python.BaseOptions(model_asset_path='face_stylizer.task')
options = vision.FaceStylizerOptions(base_options=base_options)

# Create the face stylizer
with vision.FaceStylizer.create_from_options(options) as stylizer:

  # Loop through demo image(s)
  for image_file_name in IMAGE_FILENAMES:

    # Create the MediaPipe image file that will be stylized
    image = mp.Image.create_from_file(image_file_name)
    # Retrieve the stylized image
    stylized_image = stylizer.stylize(image)

    # Show the stylized image
    rgb_stylized_image = cv2.cvtColor(stylized_image.numpy_view(), cv2.COLOR_BGR2RGB)
    resize_and_show(rgb_stylized_image)
```

----------------------------------------

TITLE: Merging LoRA Weights into Base Model - PEFT - Python
DESCRIPTION: This snippet loads the fine-tuned PEFT model and merges its LoRA adapter weights back into the base model. This process creates a single, consolidated model that can be deployed without requiring the separate PEFT adapter, making it suitable for on-device inference or other deployment scenarios.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_11

LANGUAGE: Python
CODE:
```
from peft import AutoPeftModelForCausalLM
import torch

# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained("gemma3-1b-sft")
# Merge LoRA and base model and save
merged_model = model.merge_and_unload()
```

----------------------------------------

TITLE: Building MediaPipe LLM Task Bundle for Gemma (Python)
DESCRIPTION: This function `build_gemma3_1b_it_q8` creates a MediaPipe LLM task bundle (`.task` file) by configuring the TFLite model, tokenizer, and specific LLM parameters like start/stop tokens and prompt prefixes/suffixes. It uses `llm_bundler.create_bundle` to generate the final deployable task file, which is then saved to `/content/gemma3_1b_it_q8_ekv1280.task`.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_31

LANGUAGE: python
CODE:
```
from mediapipe.tasks.python.genai.bundler import llm_bundler

def build_gemma3_1b_it_q8():
  output_file = "/content/gemma3_1b_it_q8_ekv1280.task"
  tflite_model = "/content/gemma3_1b_finetune_q8_ekv1024.tflite"
  tokenizer_model = (
      "/content/tokenizer.model"
  )
  config = llm_bundler.BundleConfig(
      tflite_model=tflite_model,
      tokenizer_model=tokenizer_model,
      start_token="<bos>",
      stop_tokens=["<eos>"],
      output_filename=output_file,
      enable_bytes_to_unicode_mapping=False,
      prompt_prefix="<start_of_turn>user\n",
      prompt_suffix="<end_of_turn>\n<start_of_turn>model\n"
  )
  llm_bundler.create_bundle(config)

# Build the MediaPipe task bundle.
build_gemma3_1b_it_q8()
```

----------------------------------------

TITLE: Performing Image Segmentation and Mask Visualization with MediaPipe
DESCRIPTION: This code initializes the MediaPipe `ImageSegmenter` using the downloaded DeepLab v3 model. It then processes the test image to generate a category mask, which is used to create a visual representation where the foreground and background are highlighted with distinct colors (white and gray, respectively). It depends on `numpy`, `mediapipe`, and the `resize_and_show` function.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_segmentation/python/image_segmentation.ipynb#_snippet_5

LANGUAGE: python
CODE:
```
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white


# Create the options that will be used for ImageSegmenter
base_options = python.BaseOptions(model_asset_path='deeplabv3.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)

# Create the image segmenter
with vision.ImageSegmenter.create_from_options(options) as segmenter:

  # Loop through demo image(s)
  for image_file_name in IMAGE_FILENAMES:

    # Create the MediaPipe image file that will be segmented
    image = mp.Image.create_from_file(image_file_name);

    # Retrieve the masks for the segmented image
    segmentation_result = segmenter.segment(image);
    category_mask = segmentation_result.category_mask;

    # Generate solid color images for showing the output segmentation mask.
    image_data = image.numpy_view();
    fg_image = np.zeros(image_data.shape, dtype=np.uint8);
    fg_image[:] = MASK_COLOR;
    bg_image = np.zeros(image_data.shape, dtype=np.uint8);
    bg_image[:] = BG_COLOR;

    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2;
    output_image = np.where(condition, fg_image, bg_image);

    print(f'Segmentation mask of {name}:');
    resize_and_show(output_image)
```

----------------------------------------

TITLE: PASCAL VOC Dataset Directory and XML Structure
DESCRIPTION: This snippet describes the PASCAL VOC dataset format, featuring a `data` directory for images and an `Annotations` directory for per-image XML annotation files. It also provides the XML schema for these annotation files, detailing filename, object names, and bounding box coordinates.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_5

LANGUAGE: Text
CODE:
```
<dataset_dir>/
  data/
    <file0>.<jpg/jpeg>
    ...
  Annotations/
    <file0>.xml
    ...
```

LANGUAGE: XML
CODE:
```
<annotation>
  <filename>file0.jpg</filename>
  <object>
    <name>kangaroo</name>
    <bndbox>
      <xmin>233</xmin>
      <ymin>89</ymin>
      <xmax>386</xmax>
      <ymax>262</ymax>
    </bndbox>
  </object>
  <object>
    ...
  </object>
  ...
</annotation>
```

----------------------------------------

TITLE: Training MediaPipe Gesture Recognizer with Custom Hyperparameters (Python)
DESCRIPTION: This snippet demonstrates how to train a new gesture recognizer model with customized hyperparameters. It sets a specific `learning_rate` and `export_dir` via `HParams`, and a `dropout_rate` via `ModelOptions`, then combines them into `GestureRecognizerOptions` before creating and training the model.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/gesture_recognizer.ipynb#_snippet_11

LANGUAGE: python
CODE:
```
hparams = gesture_recognizer.HParams(learning_rate=0.003, export_dir="exported_model_2")
model_options = gesture_recognizer.ModelOptions(dropout_rate=0.2)
options = gesture_recognizer.GestureRecognizerOptions(model_options=model_options, hparams=hparams)
model_2 = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)
```

----------------------------------------

TITLE: Drawing Hand Landmarks on Image - Python
DESCRIPTION: This Python function `draw_landmarks_on_image` visualizes detected hand landmarks and handedness on an input RGB image. It iterates through detected hands, draws landmarks and connections using MediaPipe's drawing utilities, and adds handedness text using OpenCV. It requires `mediapipe`, `numpy`, and `cv2` for image processing and drawing.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#_snippet_3

LANGUAGE: Python
CODE:
```
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image
```

----------------------------------------

TITLE: Running Inference with Hugging Face Pipeline (Python)
DESCRIPTION: This code demonstrates how to perform text generation inference using the Hugging Face `transformers` pipeline. It initializes a text generation pipeline with the `merged_model` and `tokenizer`, then generates a response for a given `prompt` with a maximum of 100 new tokens.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_13

LANGUAGE: Python
CODE:
```
from transformers import pipeline

prompt = "What is the primary function of mitochondria within a cell?"
pipe = pipeline("text-generation", model=merged_model, tokenizer=tokenizer)
pipe(prompt, max_new_tokens=100)
```

----------------------------------------

TITLE: Performing Face Landmark Detection with MediaPipe in Python
DESCRIPTION: This snippet demonstrates the end-to-end process of setting up a MediaPipe FaceLandmarker, loading an image, detecting face landmarks, and visualizing the results. It includes importing necessary modules, configuring the detector with options like blendshapes and transformation matrices, and then processing the output for display.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/python/[MediaPipe_Python_Tasks]_Face_Landmarker.ipynb#_snippet_6

LANGUAGE: python
CODE:
```
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("image.png")

# STEP 4: Detect face landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
```

----------------------------------------

TITLE: Creating and Splitting Image Classification Dataset (Python)
DESCRIPTION: This code demonstrates how to load image data into a `Dataset` object using `image_classifier.Dataset.from_folder` and then split it into training, testing, and validation sets. The data is initially split 80% for training, with the remaining 20% further split equally into 10% for testing and 10% for validation, preparing it for model retraining.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb#_snippet_6

LANGUAGE: python
CODE:
```
data = image_classifier.Dataset.from_folder(image_path)
train_data, remaining_data = data.split(0.8)
test_data, validation_data = remaining_data.split(0.5)
```

----------------------------------------

TITLE: Running Greedy Decode for LLM - Python
DESCRIPTION: This function performs the token decoding process using a greedy sampler. It iteratively predicts the next token based on the current KV cache and input position, stopping when an end-of-sequence token is encountered or `max_decode_steps` is reached. It requires `start_pos`, `start_token_id`, `kv_cache` (from prefill), and `max_decode_steps` as inputs, returning the concatenated decoded text.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma2_tflite.ipynb#_snippet_12

LANGUAGE: Python
CODE:
```
next_pos = start_pos
next_token = start_token_id
decode_text = []
decode_inputs = kv_cache

for _ in range(max_decode_steps):
  decode_inputs.update({
      "tokens": np.array([[next_token]], dtype=np.int32),
      "input_pos": np.array([next_pos], dtype=np.int32),
  })
  decode_outputs = self._decode_runner(**decode_inputs)
  # Output logits has shape (batch=1, 1, vocab_size). We only take the first
  # element.
  logits = decode_outputs.pop("logits")[0][0]
  next_token = self._greedy_sampler(logits)
  if next_token == self._tokenizer.eos_token_id:
    break
  decode_text.append(self._tokenizer.decode(next_token, skip_special_tokens=False))
  print(decode_text[-1], end='', flush=True)
  # Decode outputs includes logits and kv cache. We already poped out
  # logits, so the rest is kv cache. We pass the updated kv cache as input
  # to the next decode step.
  decode_inputs = decode_outputs
  next_pos += 1

print() # print a new line at the end.
return ''.join(decode_text)
```

----------------------------------------

TITLE: Visualizing COCO Dataset Images and Bounding Boxes with Matplotlib
DESCRIPTION: This Python code defines functions to visualize images from a COCO dataset along with their bounding box annotations using Matplotlib. It includes utilities for drawing outlines, boxes, and text labels, then orchestrates loading image and annotation data from `labels.json` to display a specified number of example images with their corresponding object detections. Requires `matplotlib`, `collections`, and `math`.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_7

LANGUAGE: Python
CODE:
```
#@title Visualize the training dataset
import matplotlib.pyplot as plt
from matplotlib import patches, text, patheffects
from collections import defaultdict
import math

def draw_outline(obj):
  obj.set_path_effects([patheffects.Stroke(linewidth=4,  foreground='black'), patheffects.Normal()])
def draw_box(ax, bb):
  patch = ax.add_patch(patches.Rectangle((bb[0],bb[1]), bb[2], bb[3], fill=False, edgecolor='red', lw=2))
  draw_outline(patch)
def draw_text(ax, bb, txt, disp):
  text = ax.text(bb[0],(bb[1]-disp),txt,verticalalignment='top'
  ,color='white',fontsize=10,weight='bold')
  draw_outline(text)
def draw_bbox(ax, annotations_list, id_to_label, image_shape):
  for annotation in annotations_list:
    cat_id = annotation["category_id"]
    bbox = annotation["bbox"]
    draw_box(ax, bbox)
    draw_text(ax, bbox, id_to_label[cat_id], image_shape[0] * 0.05)
def visualize(dataset_folder, max_examples=None):
  with open(os.path.join(dataset_folder, "labels.json"), "r") as f:
    labels_json = json.load(f)
  images = labels_json["images"]
  cat_id_to_label = {item["id"]:item["name"] for item in labels_json["categories"]}
  image_annots = defaultdict(list)
  for annotation_obj in labels_json["annotations"]:
    image_id = annotation_obj["image_id"]
    image_annots[image_id].append(annotation_obj)

  if max_examples is None:
    max_examples = len(image_annots.items())
  n_rows = math.ceil(max_examples / 3)
  fig, axs = plt.subplots(n_rows, 3, figsize=(24, n_rows*8)) # 3 columns(2nd index), 8x8 for each image
  for ind, (image_id, annotations_list) in enumerate(list(image_annots.items())[:max_examples]):
    ax = axs[ind//3, ind%3]
    img = plt.imread(os.path.join(dataset_folder, "images", images[image_id]["file_name"]))
    ax.imshow(img)
    draw_bbox(ax, annotations_list, cat_id_to_label, img.shape)
  plt.show()

visualize(train_dataset_path, 9)
```

----------------------------------------

TITLE: Blurring Image Background with MediaPipe Python
DESCRIPTION: This example demonstrates how to apply a background blur effect using MediaPipe's `InteractiveSegmenter`. It segments an image based on a keypoint ROI, then uses the resulting category mask to selectively apply a Gaussian blur to the background while keeping the foreground sharp. The keypoint is also drawn on the output image.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/interactive_segmentation/python/interactive_segmenter.ipynb#_snippet_9

LANGUAGE: Python
CODE:
```
# Blur the image background based on the segmentation mask.

# Create the segmenter
with python.vision.InteractiveSegmenter.create_from_options(options) as segmenter:

  # Loop through available image(s)
  for image_file_name in IMAGE_FILENAMES:

    # Create the MediaPipe Image
    image = mp.Image.create_from_file(image_file_name)

    # Retrieve the category masks for the image
    roi = RegionOfInterest(format=RegionOfInterest.Format.KEYPOINT,
                           keypoint=NormalizedKeypoint(x, y))
    segmentation_result = segmenter.segment(image, roi)
    category_mask = segmentation_result.category_mask

    # Convert the BGR image to RGB
    image_data = cv2.cvtColor(image.numpy_view(), cv2.COLOR_BGR2RGB)

    # Apply effects
    blurred_image = cv2.GaussianBlur(image_data, (55,55), 0)
    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
    output_image = np.where(condition, image_data, blurred_image)

    # Draw a white dot with black border to denote the point of interest
    thickness, radius = 6, -1
    keypoint_px = _normalized_to_pixel_coordinates(x, y, image.width, image.height)
    cv2.circle(output_image, keypoint_px, thickness + 5, (0, 0, 0), radius)
    cv2.circle(output_image, keypoint_px, thickness, (255, 255, 255), radius)

    print(f'Blurred background of {image_file_name}:')
    resize_and_show(output_image)
```

----------------------------------------

TITLE: Performing Hand Landmark Detection - MediaPipe Python
DESCRIPTION: This snippet demonstrates the core steps for performing hand landmark detection using MediaPipe Tasks. It imports necessary modules, creates an `HandLandmarker` object with specified options (like the model path and number of hands), loads an image, and then executes the detection process to obtain `detection_result`.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#_snippet_6

LANGUAGE: Python
CODE:
```
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("image.jpg")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)
```

----------------------------------------

TITLE: Exporting an FP16 Quantized Model with MediaPipe Model Maker (Python)
DESCRIPTION: This snippet demonstrates how to export a model with post-training float16 quantization. It first ensures the model is in its float state using `restore_float_ckpt()` if QAT was previously run, then applies the `quantization_config` during export. It also shows how to list and download the exported file.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_18

LANGUAGE: Python
CODE:
```
model.restore_float_ckpt()
model.export_model(model_name="model_fp16.tflite", quantization_config=quantization_config)
!ls -lh exported_model
files.download('exported_model/model_fp16.tflite')
```

----------------------------------------

TITLE: Exporting Average Word Embedding Text Classifier and Labels (Python)
DESCRIPTION: This snippet exports the trained average word embedding model as a TFLite file using `model.export_model()`. It also exports the training labels to the specified `export_dir` using `model.export_labels()` for on-device applications.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/text_classifier.ipynb#_snippet_8

LANGUAGE: Python
CODE:
```
model.export_model()
model.export_labels(export_dir=options.hparams.export_dir)
```

----------------------------------------

TITLE: Generating StableLM 3B Conversion Configuration
DESCRIPTION: Generates a `converter.ConversionConfig` object for the StableLM 3B model. This configuration defines the input checkpoint path, vocabulary file location, output directory, and the final TFLite binary path, tailored for the specified backend (CPU/GPU).
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_15

LANGUAGE: Python
CODE:
```
def stablelm_convert_config(backend):
  input_ckpt = '/content/stablelm-3b-4e1t/'
  vocab_model_file = '/content/stablelm-3b-4e1t/'
  output_dir = '/content/intermediate/stablelm-3b-4e1t/'
  output_tflite_file = f'/content/converted_models/stablelm_{backend}.bin'
```

----------------------------------------

TITLE: Generating Gemma 7B Conversion Configuration
DESCRIPTION: Generates a `converter.ConversionConfig` object for the Gemma 7B model. This configuration defines the input checkpoint path, vocabulary file location, output directory, and the final TFLite binary path, tailored for the specified backend (CPU/GPU).
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_13

LANGUAGE: Python
CODE:
```
def gemma7b_convert_config(backend):
  input_ckpt = '/content//gemma-1.1-7b-it/'
  vocab_model_file = '/content//gemma-1.1-7b-it/'
  output_dir = '/content/intermediate//gemma-1.1-7b-it/'
  output_tflite_file = f'/content/converted_models/gemma_{backend}.bin'
  return converter.ConversionConfig(input_ckpt=input_ckpt, ckpt_format='safetensors', model_type='GEMMA_7B', backend=backend, output_dir=output_dir, combine_file_only=False, vocab_model_file=vocab_model_file, output_tflite_file=output_tflite_file)
```

----------------------------------------

TITLE: Configuring Retraining Options for MediaPipe Object Detector (Python)
DESCRIPTION: This snippet configures the essential parameters for retraining an object detection model using MediaPipe Model Maker. It specifies the model architecture (MobileNet-MultiHW-AVG) and the output directory for the exported model, encapsulating these settings into an `ObjectDetectorOptions` object. This setup is a prerequisite for initiating the model retraining process.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_9

LANGUAGE: python
CODE:
```
spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG
hparams = object_detector.HParams(export_dir='exported_model')
options = object_detector.ObjectDetectorOptions(
    supported_model=spec,
    hparams=hparams
)
```

----------------------------------------

TITLE: Highlighting Segmented Object with Color Overlay in MediaPipe Python
DESCRIPTION: This snippet shows how to highlight a segmented object with a custom overlay color. It uses the `InteractiveSegmenter` to obtain a category mask, then creates an overlay image of a specified color. The original image and the overlay are blended using the segmentation mask as an alpha channel, effectively coloring the segmented object. The keypoint is also drawn on the output image.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/interactive_segmentation/python/interactive_segmenter.ipynb#_snippet_10

LANGUAGE: Python
CODE:
```
OVERLAY_COLOR = (100, 100, 0) # cyan

# Create the segmenter
with python.vision.InteractiveSegmenter.create_from_options(options) as segmenter:

  # Loop through available image(s)
  for image_file_name in IMAGE_FILENAMES:

    # Create the MediaPipe Image
    image = mp.Image.create_from_file(image_file_name)

    # Retrieve the category masks for the image
    roi = RegionOfInterest(format=RegionOfInterest.Format.KEYPOINT,
                           keypoint=NormalizedKeypoint(x, y))
    segmentation_result = segmenter.segment(image, roi)
    category_mask = segmentation_result.category_mask

    # Convert the BGR image to RGB
    image_data = cv2.cvtColor(image.numpy_view(), cv2.COLOR_BGR2RGB)

    # Create an overlay image with the desired color (e.g., (255, 0, 0) for red)
    overlay_image = np.zeros(image_data.shape, dtype=np.uint8)
    overlay_image[:] = OVERLAY_COLOR

    # Create the condition from the category_masks array
    alpha = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1

    # Create an alpha channel from the condition with the desired opacity (e.g., 0.7 for 70%)
    alpha = alpha.astype(float) * 0.7

    # Blend the original image and the overlay image based on the alpha channel
    output_image = image_data * (1 - alpha) + overlay_image * alpha
    output_image = output_image.astype(np.uint8)

    # Draw a white dot with black border to denote the point of interest
    thickness, radius = 6, -1
    keypoint_px = _normalized_to_pixel_coordinates(x, y, image.width, image.height)
    cv2.circle(output_image, keypoint_px, thickness + 5, (0, 0, 0), radius)
    cv2.circle(output_image, keypoint_px, thickness, (255, 255, 255), radius)

    print(f'{image_file_name}:')
    resize_and_show(output_image)
```

----------------------------------------

TITLE: Evaluating Object Detection Model Performance
DESCRIPTION: This snippet evaluates the trained object detection model using the validation dataset. It calculates the validation loss and COCO metrics, providing insights into the model's performance on unseen data, with a specified 'batch_size' of 4 for evaluation.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/tutorials/object_detection/Object_Detection_for_3_dogs.ipynb#_snippet_6

LANGUAGE: python
CODE:
```
loss, coco_metrics = model.evaluate(val_data, batch_size=4)
print(f"Validation loss: {loss}")
print(f"Validation coco metrics: {coco_metrics}")
```

----------------------------------------

TITLE: Configuring LoRA for Causal LM - PEFT - Python
DESCRIPTION: This snippet configures LoRA (Low-Rank Adaptation) for fine-tuning a causal language model. It disables Weights & Biases logging and defines a `LoraConfig` object, specifying the rank (`r`), target attention modules for adaptation, and the task type as `CAUSAL_LM`.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_6

LANGUAGE: Python
CODE:
```
os.environ["WANDB_DISABLED"] = "true"

from peft import LoraConfig, PeftModel

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
```

----------------------------------------

TITLE: Defining Pose Landmark Visualization Utility in Python
DESCRIPTION: This Python function, `draw_landmarks_on_image`, is designed to overlay detected pose landmarks and their connections onto an RGB image. It processes the `detection_result` by iterating through each detected pose, converting the landmarks into a `NormalizedLandmarkList` protocol buffer, and then using MediaPipe's drawing utilities to render them on a copy of the original image.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/python/[MediaPipe_Python_Tasks]_Pose_Landmarker.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
#@markdown To better demonstrate the Pose Landmarker API, we have created a set of visualization tools that will be used in this colab. These will draw the landmarks on a detect person, as well as the expected connections between those markers.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image
```

----------------------------------------

TITLE: MediaPipe Face Detection Visualization Utilities - Python
DESCRIPTION: This Python code defines utility functions for visualizing face detection results. The `_normalized_to_pixel_coordinates` function converts normalized coordinates to pixel coordinates, while the `visualize` function draws bounding boxes and keypoints on an input image based on the `detection_result` from MediaPipe.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_detector/python/face_detector.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
from typing import Tuple, Union
import math
import cv2
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image
```

----------------------------------------

TITLE: Configuring Post-Training FP16 Quantization (Python)
DESCRIPTION: This snippet illustrates how to create a `QuantizationConfig` object specifically for float16 post-training quantization using `quantization.QuantizationConfig.for_float16()`. This configuration is then used to modify a trained model for reduced size and improved inference speed.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_17

LANGUAGE: Python
CODE:
```
quantization_config = quantization.QuantizationConfig.for_float16()
```

----------------------------------------

TITLE: Running MediaPipe Image Classifier with Custom Parameters (Python)
DESCRIPTION: This command demonstrates how to run the `classify.py` script with optional parameters to customize the image classification process. It specifies a custom TensorFlow Lite model (`--model`), limits the number of classification results (`--maxResults`), and sets a minimum score threshold (`--scoreThreshold`) for the output.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_classification/raspberry_pi/README.md#_snippet_2

LANGUAGE: Python
CODE:
```
python3 classify.py \
  --model efficientnet_lite0.tflite \
  --maxResults 5 \
  --scoreThreshold 0.5
```

----------------------------------------

TITLE: Configuring BERT Text Classifier Options (Python)
DESCRIPTION: This snippet configures `TextClassifierOptions` for a MobileBERT-based text classifier. It sets `supported_model` to `MOBILEBERT_CLASSIFIER` and defines BERT-specific hyperparameters like `epochs`, `batch_size`, `learning_rate`, and `export_dir` using `BertHParams`.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/text_classifier.ipynb#_snippet_9

LANGUAGE: Python
CODE:
```
supported_model = text_classifier.SupportedModels.MOBILEBERT_CLASSIFIER
hparams = text_classifier.BertHParams(epochs=2, batch_size=48, learning_rate=3e-5, export_dir="bert_exported_models")
options = text_classifier.TextClassifierOptions(supported_model=supported_model, hparams=hparams)
```

----------------------------------------

TITLE: Loading Gemma 3.1B IT Model with LiteRT Pipeline
DESCRIPTION: This Python code imports the `pipeline` module from `litert_tools` and loads the Gemma 3.1B IT model using its community identifier and a specific task file, preparing it for inference.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma3_1b_tflite.ipynb#_snippet_2

LANGUAGE: Python
CODE:
```
from litert_tools.pipeline import pipeline
runner = pipeline.load("litert-community/Gemma3-1B-IT", "Gemma3-1B-IT_seq128_q8_ekv1280.task")
```

----------------------------------------

TITLE: Initializing Hugging Face AutoTokenizer for Gemma (Python)
DESCRIPTION: This code initializes a `transformers.AutoTokenizer` for the `google/gemma-3-1b-pt` model. It then sets a custom `chat_template` to enforce specific conversation formatting, ensuring roles alternate correctly and handling system messages by raising an exception.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_16

LANGUAGE: Python
CODE:
```
from transformers import AutoTokenizer

model_id = 'google/gemma-3-1b-pt'
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
```

----------------------------------------

TITLE: Defining int8 Quantization Configuration (Python)
DESCRIPTION: Creates a `QuantizationConfig` object configured for 8-bit integer (`int8`) quantization using the `for_int8()` class method. This configuration specifies how the model's data types will be reduced, requiring a `train_data` representative dataset for the calibration process.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb#_snippet_15

LANGUAGE: Python
CODE:
```
quantization_config = quantization.QuantizationConfig.for_int8(train_data)
```

----------------------------------------

TITLE: Loading, Classifying, and Processing Images with MediaPipe - Python
DESCRIPTION: This snippet iterates through a list of image filenames, loads each image using `mp.Image.create_from_file`, and then performs classification using the initialized `classifier` object. It processes the classification result by extracting the top category name and score, storing them for later visualization. The `display_batch_of_images` function is then called to show the results.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_classification/python/image_classifier.ipynb#_snippet_9

LANGUAGE: python
CODE:
```
images = []
predictions = []
for image_name in IMAGE_FILENAMES:
  # STEP 3: Load the input image.
  image = mp.Image.create_from_file(image_name)

  # STEP 4: Classify the input image.
  classification_result = classifier.classify(image)

  # STEP 5: Process the classification result. In this case, visualize it.
  images.append(image)
  top_category = classification_result.classifications[0].categories[0]
  predictions.append(f"{top_category.category_name} ({top_category.score:.2f})")

display_batch_of_images(images, predictions)
```

----------------------------------------

TITLE: Running MediaPipe Gesture Recognizer with Default Parameters
DESCRIPTION: This command initiates the MediaPipe gesture recognition application using Python. By default, it utilizes `gesture_recognizer.task` as the model, detects a single hand, and applies default confidence thresholds for detection, presence, and tracking.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/gesture_recognizer/raspberry_pi/README.md#_snippet_1

LANGUAGE: Shell
CODE:
```
python3 recognize.py
```

----------------------------------------

TITLE: Running MediaPipe Object Detection with Default Model in Python
DESCRIPTION: This Python command executes the `detect.py` script to start real-time object detection using the specified TensorFlow Lite model, `efficientdet_lite0.tflite`. It displays the camera feed with detected objects, labels, and scores on a connected monitor.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/object_detection/raspberry_pi/README.md#_snippet_1

LANGUAGE: Python
CODE:
```
python3 detect.py \
  --model efficientdet_lite0.tflite
```

----------------------------------------

TITLE: Running MediaPipe Face Detection with Default Model (Python)
DESCRIPTION: This Python command executes the `detect.py` script to start the real-time face detection application. It specifies `detector.tflite` as the TensorFlow Lite model to be used for inference, displaying results on a connected monitor.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_detector/raspberry_pi/README.md#_snippet_1

LANGUAGE: Python
CODE:
```
python3 detect.py \
  --model detector.tflite
```

----------------------------------------

TITLE: Rerunning Quantization Aware Training after Restoring Checkpoint (Python)
DESCRIPTION: This snippet illustrates how to re-run Quantization Aware Training (QAT) without retraining the base float model. It uses `model.restore_float_ckpt()` to revert the model to its fully trained float state before applying new QAT hyperparameters and re-evaluating.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_14

LANGUAGE: Python
CODE:
```
new_qat_hparams = object_detector.QATHParams(learning_rate=0.9, batch_size=4, epochs=15, decay_steps=5, decay_rate=0.96)
model.restore_float_ckpt()
model.quantization_aware_training(train_data, validation_data, qat_hparams=new_qat_hparams)
qat_loss, qat_coco_metrics = model.evaluate(validation_data)
print(f"QAT validation loss: {qat_loss}")
print(f"QAT validation coco metrics: {qat_coco_metrics}")
```

----------------------------------------

TITLE: Configuring Face Stylizer Retraining Options in Python
DESCRIPTION: This snippet demonstrates how to configure the `FaceStylizerOptions` for retraining a face stylizer model. It specifies the model architecture (`BLAZE_FACE_STYLIZER_256`), defines `swap_layers` to control style application, and sets hyperparameters like `learning_rate`, `epochs`, `batch_size`, and an `export_dir` for the retrained model.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/face_stylizer.ipynb#_snippet_6

LANGUAGE: Python
CODE:
```
face_stylizer_options = face_stylizer.FaceStylizerOptions(
  model=face_stylizer.SupportedModels.BLAZE_FACE_STYLIZER_256,
  model_options=face_stylizer.ModelOptions(swap_layers=[10,11]),
  hparams=face_stylizer.HParams(
      learning_rate=8e-4, epochs=200, batch_size=2, export_dir="exported_model"
  )
)
```

----------------------------------------

TITLE: Utility Functions for Face Landmark Visualization - Python
DESCRIPTION: This snippet defines two Python functions: draw_landmarks_on_image and plot_face_blendshapes_bar_graph. draw_landmarks_on_image overlays detected face landmarks onto an image, while plot_face_blendshapes_bar_graph visualizes face blendshape scores as a bar graph. These utilities are crucial for interpreting MediaPipe's face landmark detection output.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/python/[MediaPipe_Python_Tasks]_Face_Landmarker.ipynb#_snippet_3

LANGUAGE: Python
CODE:
```
#@markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()
```

----------------------------------------

TITLE: Defining Object Detection Visualization Function - Python
DESCRIPTION: This Python function, `visualize`, is designed to draw bounding boxes, category labels, and detection probabilities onto an input image. It leverages OpenCV (`cv2`) for image manipulation, making the object detection results visually interpretable. The function takes an image and a `detection_result` object as input and returns the annotated image.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/object_detection/python/object_detector.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
#@markdown We implemented some functions to visualize the object detection results. <br/> Run the following cell to activate the functions.
import cv2
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image
```

----------------------------------------

TITLE: Configuring Retraining Options for MediaPipe Image Classifier (Python)
DESCRIPTION: This snippet sets up the retraining options for a MediaPipe image classifier. It specifies the model architecture (MobileNetV2) and the output directory for the exported model using `HParams` and `ImageClassifierOptions`. These options are prerequisites for starting the retraining process.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb#_snippet_7

LANGUAGE: python
CODE:
```
spec = image_classifier.SupportedModels.MOBILENET_V2
hparams = image_classifier.HParams(export_dir="exported_model")
options = image_classifier.ImageClassifierOptions(supported_model=spec, hparams=hparams)
```

----------------------------------------

TITLE: Defining Phi-2 Model Conversion Configuration in Python
DESCRIPTION: This Python function `phi2_convert_config` generates a `converter.ConversionConfig` object for the Phi-2 model. It sets predefined paths for the input checkpoint, vocabulary model, and intermediate/final output directories, dynamically naming the TFLite output file based on the specified backend.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_17

LANGUAGE: python
CODE:
```
def phi2_convert_config(backend):
  input_ckpt = '/content/phi-2'
  vocab_model_file = '/content/phi-2/'
  output_dir = '/content/intermediate/phi-2/'
  output_tflite_file = f'/content/converted_models/phi2_{backend}.bin'

  return converter.ConversionConfig(input_ckpt=input_ckpt, ckpt_format='safetensors', model_type='PHI_2', backend=backend, output_dir=output_dir, combine_file_only=False, vocab_model_file=vocab_model_file, output_tflite_file=output_tflite_file)
```

----------------------------------------

TITLE: Visualizing MediaPipe Classification Results with OpenCV (Python)
DESCRIPTION: This snippet processes a MediaPipe classification result by drawing detected landmarks onto the original image. It then converts the annotated image from RGB to BGR format (required by OpenCV's `imshow`) and displays it using `cv2_imshow` (likely a wrapper for `cv2.imshow` in environments like Colab).
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#_snippet_7

LANGUAGE: Python
CODE:
```
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
```

----------------------------------------

TITLE: Visualizing Pose Segmentation Mask in Python
DESCRIPTION: This snippet extracts the segmentation mask generated by the pose landmarker from the detection result. It converts the single-channel mask into a 3-channel image by repeating the mask values across all color channels and then scales them to 255 for visualization, effectively highlighting the detected person's silhouette.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/python/[MediaPipe_Python_Tasks]_Pose_Landmarker.ipynb#_snippet_6

LANGUAGE: python
CODE:
```
segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
cv2_imshow(visualized_mask)
```

----------------------------------------

TITLE: Executing Prefill Operation in LiteRTLlmPipeline (Python)
DESCRIPTION: This method executes the prefill step of the LLM inference, processing an initial sequence of tokens. It handles cases with zero input tokens, prepares the input token IDs and positions as NumPy arrays, and initializes or updates the KV cache before passing inputs to the prefill runner.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_23

LANGUAGE: python
CODE:
```
  def _run_prefill(
      self, prefill_token_ids: Sequence[int],
  ) -> dict[str, np.ndarray]:
    """Runs prefill and returns the kv cache.

    Args:
      prefill_token_ids: The token ids of the prefill input.

    Returns:
      The updated kv cache.
    """
    if not self._prefill_runner:
      raise ValueError("Prefill runner is not initialized.")
    prefill_token_length = len(prefill_token_ids)
    if prefill_token_length == 0:
      return self._init_kv_cache()

    # Prepare the input to be [1, max_seq_len].
    input_token_ids = [0] * self._max_seq_len
    input_token_ids[:prefill_token_length] = prefill_token_ids
    input_token_ids = np.asarray(input_token_ids, dtype=np.int32)
    input_token_ids = np.expand_dims(input_token_ids, axis=0);

    # Prepare the input position to be [max_seq_len].
    input_pos = [0] * self._max_seq_len
    input_pos[:prefill_token_length] = range(prefill_token_length)
    input_pos = np.asarray(input_pos, dtype=np.int32)

    # Initialize kv cache.
    prefill_inputs = self._init_kv_cache()
    # Prepare the tokens and input position inputs.
    prefill_inputs.update({
        "tokens": input_token_ids,
        "input_pos": input_pos,
    })
```

----------------------------------------

TITLE: Generating Falcon 1B Conversion Configuration
DESCRIPTION: Generates a `converter.ConversionConfig` object for the Falcon 1B model. It specifies the input PyTorch checkpoint, vocabulary file, intermediate and final output directories, and the target backend for converting the model to a TFLite binary.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_14

LANGUAGE: Python
CODE:
```
def falcon_convert_config(backend):
  input_ckpt = '/content/falcon-rw-1b/pytorch_model.bin'
  vocab_model_file = '/content/falcon-rw-1b/'
  output_dir = '/content/intermediate/falcon-rw-1b/'
  output_tflite_file = f'/content/converted_models/falcon_{backend}.bin'
  return converter.ConversionConfig(input_ckpt=input_ckpt, ckpt_format='pytorch', model_type='FALCON_RW_1B', backend=backend, output_dir=output_dir, combine_file_only=False, vocab_model_file=vocab_model_file, output_tflite_file=output_tflite_file)
```

----------------------------------------

TITLE: Selecting Optimal Prefill Runner in LiteRTLlmPipeline (Python)
DESCRIPTION: This method selects the most suitable prefill runner from the interpreter's available signatures based on the number of input tokens. It iterates through signatures, identifies those related to 'prefill', and chooses the one with the smallest sequence size that can accommodate the given input tokens.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_22

LANGUAGE: python
CODE:
```
  def _get_prefill_runner(self, num_input_tokens: int) :
    """Gets the prefill runner with the best suitable input size.

    Args:
      num_input_tokens: The number of input tokens.

    Returns:
      The prefill runner with the smallest input size.
    """
    best_signature = None
    delta = sys.maxsize
    max_prefill_len = -1
    for key in self._interpreter.get_signature_list().keys():
      if "prefill" not in key:
        continue
      input_pos = self._interpreter.get_signature_runner(key).get_input_details()[
          "input_pos"
      ]
      # input_pos["shape"] has shape (max_seq_len, )
      seq_size = input_pos["shape"][0]
      max_prefill_len = max(max_prefill_len, seq_size)
      if num_input_tokens <= seq_size and seq_size - num_input_tokens < delta:
        delta = seq_size - num_input_tokens
        best_signature = key
    if best_signature is None:
      raise ValueError(
          "The largest prefill length supported is %d, but we have %d number of input tokens"
          %(max_prefill_len, num_input_tokens)
      )
    return self._interpreter.get_signature_runner(best_signature)
```

----------------------------------------

TITLE: Generating Gemma 2B Conversion Configuration
DESCRIPTION: Generates a `converter.ConversionConfig` object for the Gemma 2B model. This configuration specifies input and output paths, checkpoint format (safetensors), model type, and the target backend (CPU/GPU) for the conversion process to a TFLite binary.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_12

LANGUAGE: Python
CODE:
```
def gemma2b_convert_config(backend):
  input_ckpt = '/content/gemma-2b-it/'
  vocab_model_file = '/content/gemma-2b-it/'
  output_dir = '/content/intermediate/gemma-2b-it/'
  output_tflite_file = f'/content/converted_models/gemma_{backend}.bin'
  return converter.ConversionConfig(input_ckpt=input_ckpt, ckpt_format='safetensors', model_type='GEMMA_2B', backend=backend, output_dir=output_dir, combine_file_only=False, vocab_model_file=vocab_model_file, output_tflite_file=output_tflite_file)
```

----------------------------------------

TITLE: Evaluating Custom-Trained MediaPipe Gesture Recognizer Model (Python)
DESCRIPTION: This snippet evaluates the performance of the `model_2`, which was trained with custom hyperparameters, using the `test_data`. It retrieves and prints the loss and accuracy, allowing for a comparison of performance against models trained with default or different configurations.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/gesture_recognizer.ipynb#_snippet_12

LANGUAGE: python
CODE:
```
loss, accuracy = model_2.evaluate(test_data)
print(f"Test loss:{loss}, Test accuracy:{accuracy}")
```

----------------------------------------

TITLE: Generating Segmentation Mask with MediaPipe Python
DESCRIPTION: This snippet initializes an `InteractiveSegmenter` to process images. It iterates through image files, creates a MediaPipe image, and retrieves a category mask based on a specified region of interest (ROI). The mask is then used to create a solid color foreground and background image, which are combined to visualize the segmentation. A keypoint is drawn to indicate the ROI.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/interactive_segmentation/python/interactive_segmenter.ipynb#_snippet_8

LANGUAGE: Python
CODE:
```
# Create the interactive segmenter
with vision.InteractiveSegmenter.create_from_options(options) as segmenter:

  # Loop through demo image(s)
  for image_file_name in IMAGE_FILENAMES:

    # Create the MediaPipe image file that will be segmented
    image = mp.Image.create_from_file(image_file_name)

    # Retrieve the masks for the segmented image
    roi = RegionOfInterest(format=RegionOfInterest.Format.KEYPOINT,
                           keypoint=NormalizedKeypoint(x, y))
    segmentation_result = segmenter.segment(image, roi)
    category_mask = segmentation_result.category_mask

    # Generate solid color images for showing the output segmentation mask.
    image_data = image.numpy_view()
    fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR

    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
    output_image = np.where(condition, fg_image, bg_image)

    # Draw a white dot with black border to denote the point of interest
    thickness, radius = 6, -1
    keypoint_px = _normalized_to_pixel_coordinates(x, y, image.width, image.height)
    cv2.circle(output_image, keypoint_px, thickness + 5, (0, 0, 0), radius)
    cv2.circle(output_image, keypoint_px, thickness, (255, 255, 255), radius)

    print(f'Segmentation mask of {image_file_name}:')
    resize_and_show(output_image)
```

----------------------------------------

TITLE: Installing Core Fine-tuning Libraries (Python)
DESCRIPTION: This snippet installs several key Python libraries required for large language model fine-tuning, including `bitsandbytes`, `peft`, `trl`, `accelerate`, `datasets`, and `transformers`. These packages provide functionalities for efficient memory usage, parameter-efficient fine-tuning, training loops, distributed training, dataset handling, and transformer model operations, respectively.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
!pip3 install --upgrade -q -U bitsandbytes
!pip3 install --upgrade -q -U peft
!pip3 install --upgrade -q -U trl
!pip3 install --upgrade -q -U accelerate
!pip3 install --upgrade -q -U datasets
!pip3 install --force-reinstall transformers
```

----------------------------------------

TITLE: Training MediaPipe Gesture Recognizer Model (Python)
DESCRIPTION: This snippet initializes hyperparameters and model options, then creates and trains a custom gesture recognizer model using the `create` method. It takes `train_data` and `validation_data` as input, along with `GestureRecognizerOptions` which encapsulate `HParams` for configuration like export directory.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/gesture_recognizer.ipynb#_snippet_7

LANGUAGE: python
CODE:
```
hparams = gesture_recognizer.HParams(export_dir="exported_model")
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)
```

----------------------------------------

TITLE: Creating LiteRT Interpreter - Python
DESCRIPTION: This snippet initializes the LiteRT interpreter using `InterpreterWithCustomOps`. It registers custom GenAI operations, provides the path to the downloaded model, sets the number of threads for execution, and enables experimental delegate features for optimized performance.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma2_tflite.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
interpreter = interpreter_lib.InterpreterWithCustomOps(
    custom_op_registerers=["pywrap_genai_ops.GenAIOpsRegisterer"],
    model_path=model_path,
    num_threads=2,
    experimental_default_delegate_latest_features=True)
```

----------------------------------------

TITLE: Downloading and Preparing Gesture Recognition Dataset
DESCRIPTION: This snippet downloads a sample rock paper scissors dataset from Google Cloud Storage and unzips it. It sets the `dataset_path` variable, which is crucial for subsequent data loading and model training steps.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/gesture_recognizer.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
!wget https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/rps_data_sample.zip
!unzip rps_data_sample.zip
dataset_path = "rps_data_sample"
```

----------------------------------------

TITLE: Downloading and Preparing Dog Dataset
DESCRIPTION: This snippet downloads a zipped dataset of dog images from Google Cloud Storage, unzips it, and then defines the paths for the training and validation subsets of the dataset, preparing them for model training.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/tutorials/object_detection/Object_Detection_for_3_dogs.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
!wget https://storage.googleapis.com/mediapipe-assets/dogs2.zip --no-check-certificate
!unzip dogs2.zip
train_dataset_path = "dogs/train"
validation_dataset_path = "dogs/validate"
```

----------------------------------------

TITLE: Loading and Splitting Dataset with MediaPipe Gesture Recognizer (Python)
DESCRIPTION: This snippet loads a hand gesture dataset from a specified folder, applying MediaPipe's hand detection to extract landmarks and filter out images without hands. It then splits the loaded data into 80% for training, 10% for validation, and 10% for testing, preparing it for model training. The `HandDataPreprocessingParams` can be used to configure shuffling and detection confidence.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/gesture_recognizer.ipynb#_snippet_6

LANGUAGE: python
CODE:
```
data = gesture_recognizer.Dataset.from_folder(
    dirname=dataset_path,
    hparams=gesture_recognizer.HandDataPreprocessingParams()
)
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)
```

----------------------------------------

TITLE: Applying Quantization Aware Training with MediaPipe Model Maker (Python)
DESCRIPTION: This snippet demonstrates how to configure and run Quantization Aware Training (QAT) using `object_detector.QATHParams` and `model.quantization_aware_training`. It shows the initial setup of hyperparameters for QAT and evaluates the model's performance after training.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_13

LANGUAGE: Python
CODE:
```
qat_hparams = object_detector.QATHParams(learning_rate=0.3, batch_size=4, epochs=10, decay_steps=6, decay_rate=0.96)
model.quantization_aware_training(train_data, validation_data, qat_hparams=qat_hparams)
qat_loss, qat_coco_metrics = model.evaluate(validation_data)
print(f"QAT validation loss: {qat_loss}")
print(f"QAT validation coco metrics: {qat_coco_metrics}")
```

----------------------------------------

TITLE: COCO Dataset Directory and JSON Structure
DESCRIPTION: This snippet illustrates the standard directory layout for a COCO dataset, including a `data` folder for images and a `labels.json` file for annotations. It also details the JSON schema for `labels.json`, which contains categories, image metadata, and object annotations with bounding box coordinates.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_4

LANGUAGE: Text
CODE:
```
<dataset_dir>/
  data/
    <img0>.<jpg/jpeg>
    <img1>.<jpg/jpeg>
    ...
  labels.json
```

LANGUAGE: JSON
CODE:
```
{
  "categories":[
    {"id":1, "name":<cat1_name>},
    ...
  ],
  "images":[
    {"id":0, "file_name":"<img0>.<jpg/jpeg>"},
    ...
  ],
  "annotations":[
    {"id":0, "image_id":0, "category_id":1, "bbox":[x-top left, y-top left, width, height]},
    ...
  ]
}
```

----------------------------------------

TITLE: Defining Visualization Utilities for Gesture Recognition - Python
DESCRIPTION: This Python code block defines utility functions for visualizing gesture recognition results and hand landmarks using `matplotlib` and `mediapipe.solutions`. It includes `display_one_image` for showing a single image with a title and `display_batch_of_images_with_gestures_and_hand_landmarks` for displaying multiple images with detected gestures and annotated hand landmarks. It also configures `matplotlib` for cleaner plots and initializes MediaPipe drawing utilities.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/gesture_recognizer/python/gesture_recognizer.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
#@markdown We implemented some functions to visualize the gesture recognition results. <br/> Run the following cell to activate the functions.
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.labelbottom': False,
    'xtick.bottom': False,
    'ytick.labelleft': False,
    'ytick.left': False,
    'xtick.labeltop': False,
    'xtick.top': False,
    'ytick.labelright': False,
    'ytick.right': False
})

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def display_one_image(image, title, subplot, titlesize=16):
    """Displays one image along with the predicted category name and score."""
    plt.subplot(*subplot)
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)


def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
    """Displays a batch of images with the gesture category and its score along with the hand landmarks."""
    # Images and labels.
    images = [image.numpy_view() for image in images]
    gestures = [top_gesture for (top_gesture, _) in results]
    multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]

    # Auto-squaring: this will drop data that does not fit into square or square-ish rectangle.
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # Size and spacing.
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    # Display gestures and hand landmarks.
    for i, (image, gestures) in enumerate(zip(images[:rows*cols], gestures[:rows*cols])):
        title = f"{gestures.category_name} ({gestures.score:.2f})"
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols) * 40 + 3
        annotated_image = image.copy()

        for hand_landmarks in multi_hand_landmarks_list[i]:
          hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
          hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
          ])

          mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        subplot = display_one_image(annotated_image, title, subplot, titlesize=dynamic_titlesize)

    # Layout.
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()
```

----------------------------------------

TITLE: Resizing Token Embeddings and Saving Merged Model (Python)
DESCRIPTION: This snippet resizes the token embeddings of the `merged_model` to match a new vocabulary size (262144) and then saves the model to the 'merged_model' directory. The `safe_serialization` and `max_shard_size` parameters ensure efficient and safe storage of the potentially large model.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_12

LANGUAGE: Python
CODE:
```
merged_model.resize_token_embeddings(262144)
merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")
```

----------------------------------------

TITLE: Exporting an Int8 Quantized Model with MediaPipe Model Maker (Python)
DESCRIPTION: This snippet demonstrates how to export a model that has undergone Quantization Aware Training (QAT) to an int8 quantized TensorFlow Lite format. The `export_model` function automatically determines the quantization type based on previous training steps. It also shows how to list and download the exported file.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_15

LANGUAGE: Python
CODE:
```
model.export_model('model_int8_qat.tflite')
!ls -lh exported_model
files.download('exported_model/model_int8_qat.tflite')
```

----------------------------------------

TITLE: Initializing Gemma2-2B-IT Tokenizer - Python
DESCRIPTION: This snippet initializes a tokenizer for the Gemma2-2B-IT model using `AutoTokenizer.from_pretrained` from the `transformers` library. It loads the pre-trained tokenizer from the specified HuggingFace model identifier, which is essential for processing text inputs for the model.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma2_tflite.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
```

----------------------------------------

TITLE: Greedy Sampling for Token Selection - LiteRT LLM Pipeline - Python
DESCRIPTION: This utility method implements a simple greedy sampling strategy. It takes an array of logits (model outputs) and returns the integer ID of the token with the highest logit score, representing the most probable next token.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_25

LANGUAGE: python
CODE:
```
  def _greedy_sampler(self, logits: np.ndarray) -> int:
    return int(np.argmax(logits))
```

----------------------------------------

TITLE: Retrieving and Printing Image Classification Labels (Python)
DESCRIPTION: This code block retrieves and prints the class labels from the downloaded dataset. It iterates through the subdirectories within the `image_path`, assuming each subdirectory name represents a class label, and appends them to a list. This step is crucial for verifying the data's organization before training.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
print(image_path)
labels = []
for i in os.listdir(image_path):
  if os.path.isdir(os.path.join(image_path, i)):
    labels.append(i)
print(labels)
```

----------------------------------------

TITLE: Downloading and Preparing Example Dataset for Object Detection (Python)
DESCRIPTION: This snippet downloads and extracts an example dataset for object detection model retraining. It uses `wget` to fetch a zip file containing android figurine images, then `unzip` to extract its contents, and finally sets variables for the training and validation dataset paths, which are organized in COCO Dataset format.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_3

LANGUAGE: Python
CODE:
```
!wget https://storage.googleapis.com/mediapipe-tasks/object_detector/android_figurine.zip
!unzip android_figurine.zip
train_dataset_path = "android_figurine/train"
validation_dataset_path = "android_figurine/validation"
```

----------------------------------------

TITLE: Downloading Example Image Dataset (Python)
DESCRIPTION: This snippet downloads and extracts an example dataset of flower photos from a Google Cloud Storage URL. It uses `tf.keras.utils.get_file` to handle the download and extraction, then constructs the local path to the extracted images, which are organized into subdirectories corresponding to class labels.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
image_path = tf.keras.utils.get_file(
    'flower_photos.tgz',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')
```

----------------------------------------

TITLE: Loading COCO Datasets into TFRecord Format with MediaPipe Object Detector
DESCRIPTION: This Python snippet demonstrates how to load COCO-formatted datasets into `object_detector.Dataset` objects using the `from_coco_folder` method. It specifies the dataset path and a `cache_dir` to store the converted TFRecord format, preventing redundant conversions. The snippet then prints the size of the loaded training and validation datasets.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_8

LANGUAGE: Python
CODE:
```
train_data = object_detector.Dataset.from_coco_folder(train_dataset_path, cache_dir="/tmp/od_data/train")
validation_data = object_detector.Dataset.from_coco_folder(validation_dataset_path, cache_dir="/tmp/od_data/validation")
print("train_data size: ", train_data.size)
print("validation_data size: ", validation_data.size)
```

----------------------------------------

TITLE: Greedy Sampling Logits in Python
DESCRIPTION: This helper method implements a simple greedy sampling strategy. It takes an array of logits and returns the index of the token with the highest logit value, effectively selecting the most probable next token.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma2_tflite.ipynb#_snippet_10

LANGUAGE: Python
CODE:
```
  def _greedy_sampler(self, logits: np.ndarray) -> int:
    return int(np.argmax(logits))
```

----------------------------------------

TITLE: Defining Image Visualization Utilities (Python)
DESCRIPTION: This Python code defines utility functions using matplotlib to visualize image classification results. `display_one_image` shows a single image with its predicted title, while `display_batch_of_images` arranges and displays multiple images with their classifications in a grid, adjusting layout for readability. It also configures matplotlib to remove axis spines and ticks for cleaner image display, and depends on the `math` module.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_classification/python/image_classifier.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
from matplotlib import pyplot as plt
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.labelbottom': False,
    'xtick.bottom': False,
    'ytick.labelleft': False,
    'ytick.left': False,
    'xtick.labeltop': False,
    'xtick.top': False,
    'ytick.labelright': False,
    'ytick.right': False
})


def display_one_image(image, title, subplot, titlesize=16):
    """Displays one image along with the predicted category name and score."""
    plt.subplot(*subplot)
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)

def display_batch_of_images(images, predictions):
    """Displays a batch of images with the classifications."""
    # Images and predictions.
    images = [image.numpy_view() for image in images]

    # Auto-squaring: this will drop data that does not fit into square or square-ish rectangle.
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # Size and spacing.
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    # Display.
    for i, (image, prediction) in enumerate(zip(images[:rows*cols], predictions[:rows*cols])):
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols) * 40 + 3
        subplot = display_one_image(image, prediction, subplot, titlesize=dynamic_titlesize)

    # Layout.
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()
```

----------------------------------------

TITLE: Configuring Average Word Embedding Text Classifier Options (Python)
DESCRIPTION: This snippet initializes `TextClassifierOptions` for an average word embedding model. It sets the `supported_model` to `AVERAGE_WORD_EMBEDDING_CLASSIFIER` and defines hyperparameters like `epochs`, `batch_size`, `learning_rate`, and `export_dir` using `AverageWordEmbeddingHParams`.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/text_classifier.ipynb#_snippet_5

LANGUAGE: Python
CODE:
```
supported_model = text_classifier.SupportedModels.AVERAGE_WORD_EMBEDDING_CLASSIFIER
hparams = text_classifier.AverageWordEmbeddingHParams(epochs=10, batch_size=32, learning_rate=0, export_dir="awe_exported_models")
options = text_classifier.TextClassifierOptions(supported_model=supported_model, hparams=hparams)
```

----------------------------------------

TITLE: Creating Dataset for Face Stylizer Training
DESCRIPTION: This line of code creates a dataset object for the face stylizer model using the `Dataset.from_image` method from `mediapipe_model_maker.face_stylizer`. It takes the path to the single stylized image (`style_image_path`) as input, preparing the data in the format required for training the face stylizer model.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/face_stylizer.ipynb#_snippet_5

LANGUAGE: Python
CODE:
```
data = face_stylizer.Dataset.from_image(filename=style_image_path)
```

----------------------------------------

TITLE: Installing MediaPipe Library - Python
DESCRIPTION: This command installs the MediaPipe library using pip, the Python package installer. The -q flag ensures a quiet installation, suppressing verbose output. This is a prerequisite for using MediaPipe Tasks.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/python/[MediaPipe_Python_Tasks]_Face_Landmarker.ipynb#_snippet_1

LANGUAGE: Python
CODE:
```
!pip install -q mediapipe
```

----------------------------------------

TITLE: Exporting MediaPipe Model to TensorFlow Lite and Listing Files (Python)
DESCRIPTION: This snippet exports the trained `model` into a TensorFlow Lite format, including necessary metadata and label files, making it suitable for on-device applications. After export, it uses a shell command to list the contents of the `exported_model` directory, verifying the creation of the TFLite file.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/gesture_recognizer.ipynb#_snippet_9

LANGUAGE: python
CODE:
```
model.export_model()
!ls exported_model
```

----------------------------------------

TITLE: Installing MediaPipe Library - Python
DESCRIPTION: This command installs the MediaPipe library using pip, the Python package installer. The `-q` flag ensures a quiet installation, suppressing verbose output. MediaPipe is a framework for building machine learning pipelines, essential for gesture recognition.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/gesture_recognizer/python/gesture_recognizer.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
!pip install -q mediapipe
```

----------------------------------------

TITLE: Installing MediaPipe Library (Python)
DESCRIPTION: This snippet installs the MediaPipe library using pip. The `-q` flag ensures a quiet installation, suppressing verbose output. This is a prerequisite for using MediaPipe Tasks functionalities.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_embedder/python/image_embedder.ipynb#_snippet_1

LANGUAGE: Python
CODE:
```
!pip install -q mediapipe
```

----------------------------------------

TITLE: Installing MediaPipe Library - Python
DESCRIPTION: This command installs the MediaPipe library using pip. The '-q' flag ensures a quiet installation, suppressing verbose output. MediaPipe is a prerequisite for utilizing its various machine learning solutions, including hand landmark detection.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#_snippet_1

LANGUAGE: Python
CODE:
```
!pip install -q mediapipe
```

----------------------------------------

TITLE: Evaluating Retrained MediaPipe Image Classifier Performance (Python)
DESCRIPTION: This snippet evaluates the performance of the retrained model against a test dataset. It calculates and prints the test loss and accuracy, which are crucial metrics for assessing model quality. High accuracy is generally desired, but caution against overfitting is advised.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb#_snippet_9

LANGUAGE: python
CODE:
```
loss, acc = model.evaluate(test_data)
print(f'Test loss:{loss}, Test accuracy:{acc}')
```

----------------------------------------

TITLE: Initializing LiteRTLlmPipeline for LLM Inference (Python)
DESCRIPTION: This constructor initializes the `LiteRTLlmPipeline` class, setting up the model interpreter and tokenizer. It also obtains the signature runner for the 'decode' operation immediately, while the 'prefill' runner is initialized dynamically later based on input size.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_19

LANGUAGE: python
CODE:
```
class LiteRTLlmPipeline:

  def __init__(self, interpreter, tokenizer):
    """Initializes the pipeline."""
    self._interpreter = interpreter
    self._tokenizer = tokenizer

    self._prefill_runner = None
    self._decode_runner = self._interpreter.get_signature_runner("decode")
```

----------------------------------------

TITLE: Installing MediaPipe Library
DESCRIPTION: This command installs the MediaPipe library using pip, a Python package installer. The `-q` flag ensures a quiet installation, suppressing verbose output. This is a prerequisite for running MediaPipe samples.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/interactive_segmentation/python/interactive_segmenter.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
!pip install -q mediapipe
```

----------------------------------------

TITLE: Running Prefill Stage - LiteRT LLM Pipeline - Python
DESCRIPTION: This method handles the prefill stage of the LLM, processing initial tokens to populate the KV cache. It dynamically generates an attention mask based on input details and ensures that only the KV cache is returned, discarding intermediate logits.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_24

LANGUAGE: python
CODE:
```
    if "mask" in self._prefill_runner.get_input_details().keys():
      # For prefill, mask has shape [batch=1, 1, seq_len, kv_cache_size].
      # We want mask[0, 0, i, j] = 0 for j<=i and -inf otherwise.
      prefill_inputs["mask"] = _get_mask(
          shape=self._prefill_runner.get_input_details()["mask"]["shape"],
          k=1,
      )
    prefill_outputs = self._prefill_runner(**prefill_inputs)
    if "logits" in prefill_outputs:
      # Prefill outputs includes logits and kv cache. We only output kv cache.
      prefill_outputs.pop("logits")

    return prefill_outputs
```

----------------------------------------

TITLE: Installing MediaPipe Library (Python)
DESCRIPTION: This command installs the MediaPipe library using pip, the Python package installer. The -q flag ensures a quiet installation, suppressing verbose output. This is a prerequisite for using MediaPipe Tasks.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_classification/python/image_classifier.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
!pip install -q mediapipe
```

----------------------------------------

TITLE: Installing LiteRT Dependency - Python
DESCRIPTION: This snippet installs the `ai-edge-litert` library, which is a prerequisite for using the LiteRT interpreter for on-device AI models. It uses the `pip` package manager.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma2_tflite.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
!pip install ai-edge-litert
```

----------------------------------------

TITLE: Installing MediaPipe Python Library
DESCRIPTION: This command installs the MediaPipe Python library using pip, which is a fundamental dependency for running any MediaPipe tasks, including image segmentation. The `-q` flag ensures a quiet installation.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_segmentation/python/image_segmentation.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
!pip install -q mediapipe
```

----------------------------------------

TITLE: Evaluating Face Stylizer Model Performance (Python)
DESCRIPTION: This snippet evaluates the performance of a retrained face stylizer model by reconstructing a style image. It uses OpenCV (cv2) for image resizing and display, and the 'face_stylizer_model' to perform the stylization. The reconstructed image is then displayed to assess the model's convergence and quality, helping to determine if the model is suitable for new data.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/face_stylizer.ipynb#_snippet_8

LANGUAGE: python
CODE:
```
print('Input style image')
resized_style_cv_image = cv2.resize(style_cv_image, (256, 256))
cv2_imshow(resized_style_cv_image)

eval_output = face_stylizer_model.stylize(data)
eval_output_data = eval_output.gen_tf_dataset()
iterator = iter(eval_output_data)

reconstruct_style_image = (tf.squeeze(iterator.get_next()).numpy())
test_output_image = cv2.cvtColor(reconstruct_style_image, cv2.COLOR_RGB2BGR)
print('\nReconstructed style image')
cv2_imshow(test_output_image)
```

----------------------------------------

TITLE: Downloading Gemma 7B Model Files
DESCRIPTION: Downloads specific files for the Gemma 7B instruction-tuned model from Hugging Face. It requires a Hugging Face token for authentication and fetches tokenizer and model safetensors files into a local directory using `hf_hub_download`.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_8

LANGUAGE: Python
CODE:
```
def gemma7b_download(token):
  REPO_ID = "google/gemma-1.1-7b-it"
  FILENAMES = ["tokenizer.json", "tokenizer_config.json", "model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors", "model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors"]
  os.environ['HF_TOKEN'] = token
  with out:
    for filename in FILENAMES:
      hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir="./gemma-1.1-7b-it")
```

----------------------------------------

TITLE: Downloading Falcon 1B Model Files
DESCRIPTION: Downloads specific files for the Falcon 1B model from Hugging Face. It fetches tokenizer files and the PyTorch model binary into a local directory using `hf_hub_download`. This model does not require a Hugging Face token.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_9

LANGUAGE: Python
CODE:
```
def falcon_download():
  REPO_ID = "tiiuae/falcon-rw-1b"
  FILENAMES = ["tokenizer.json", "tokenizer_config.json", "pytorch_model.bin"]
  with out:
    for filename in FILENAMES:
      hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir="./falcon-rw-1b")
```

----------------------------------------

TITLE: Running MediaPipe Object Detection with Custom Parameters in Python
DESCRIPTION: This Python command runs the `detect.py` script for real-time object detection, allowing customization of the model, maximum number of detection results (`maxResults`), and the confidence score threshold (`scoreThreshold`). It uses `efficientdet_lite0.tflite` as the model, limits results to 5, and sets the score threshold to 0.3.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/object_detection/raspberry_pi/README.md#_snippet_2

LANGUAGE: Python
CODE:
```
python3 detect.py \
  --model efficientdet_lite0.tflite \
  --maxResults 5 \
  --scoreThreshold 0.3
```

----------------------------------------

TITLE: Running MediaPipe Audio Classifier with Custom Parameters (Python)
DESCRIPTION: This command demonstrates how to run the audio classification script with specific optional parameters. It sets the TensorFlow Lite model to `yamnet.tflite` and limits the classification results to 5, overriding the default values for more controlled inference.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/audio_classifier/raspberry_pi/README.md#_snippet_2

LANGUAGE: Python
CODE:
```
python3 classify.py \
    --model yamnet.tflite \
    --maxResults 5
```

----------------------------------------

TITLE: Evaluating Average Word Embedding Text Classifier (Python)
DESCRIPTION: This snippet evaluates the trained average word embedding model on the `validation_data`. It uses `model.evaluate()` to get performance metrics (loss and accuracy) and prints them to the console.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/text_classifier.ipynb#_snippet_7

LANGUAGE: Python
CODE:
```
metrics = model.evaluate(validation_data)
print(f'Test loss:{metrics[0]}, Test accuracy:{metrics[1]}')
```

----------------------------------------

TITLE: Running MediaPipe Gesture Recognizer with Custom Parameters
DESCRIPTION: This command executes the MediaPipe gesture recognition application, allowing customization of key operational parameters. It demonstrates how to specify a custom model file, increase the maximum number of hands to detect, and adjust the minimum confidence score for hand detection.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/gesture_recognizer/raspberry_pi/README.md#_snippet_2

LANGUAGE: Shell
CODE:
```
python3 recognize.py \
  --model gesture_recognizer.task \
  --numHands 2 \
  --minHandDetectionConfidence 0.5
```

----------------------------------------

TITLE: Setting HuggingFace Token in Colab Environment (Python)
DESCRIPTION: This snippet imports the `os` module and `userdata` from `google.colab` to securely retrieve and set the HuggingFace token as an environment variable. This token is essential for accessing models and tokenizers from HuggingFace, ensuring proper authentication for downloads.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
```

----------------------------------------

TITLE: Downloading Gemma 2B Model Files
DESCRIPTION: Downloads specific files for the Gemma 2B instruction-tuned model from Hugging Face. It sets the `HF_TOKEN` environment variable for authentication and uses `hf_hub_download` to fetch tokenizer and model safetensors files into a local directory.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_7

LANGUAGE: Python
CODE:
```
def gemma2b_download(token):
  REPO_ID = "google/gemma-2b-it"
  FILENAMES = ["tokenizer.json", "tokenizer_config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
  os.environ['HF_TOKEN'] = token
  with out:
    for filename in FILENAMES:
      hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir="./gemma-2b-it")
```

----------------------------------------

TITLE: Downloading Phi 2 Model Files
DESCRIPTION: Downloads specific files for the Phi 2 model from Hugging Face. It fetches tokenizer files and model safetensors into a local directory using `hf_hub_download`. This model does not require a Hugging Face token.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_11

LANGUAGE: Python
CODE:
```
def phi2_download():
  REPO_ID = "microsoft/phi-2"
  FILENAMES = ["tokenizer.json", "tokenizer_config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
  with out:
    for filename in FILENAMES:
      hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir="./phi-2")
```

----------------------------------------

TITLE: Converting Normalized Coordinates to Pixels in Python
DESCRIPTION: This Python function converts normalized (0-1) coordinates to pixel coordinates within an image. It takes normalized x and y values, image width, and image height as input, returning pixel (x, y) or None if coordinates are out of bounds. It's a utility for visualization.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/interactive_segmentation/python/interactive_segmenter.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int):
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px
```

----------------------------------------

TITLE: Installing MediaPipe Python Package
DESCRIPTION: This command installs the MediaPipe library for Python using pip, the package installer. The '-q' flag ensures a quiet installation, suppressing detailed output. This step is a prerequisite for utilizing any MediaPipe functionalities.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/python/[MediaPipe_Python_Tasks]_Pose_Landmarker.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
!pip install -q mediapipe
```

----------------------------------------

TITLE: Installing MediaPipe Library - Python
DESCRIPTION: This command installs the MediaPipe library and its related dependencies using pip. It is a prerequisite for running the face stylizer demo, ensuring all necessary packages are available in the environment.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_stylizer/python/face_stylizer.ipynb#_snippet_1

LANGUAGE: Python
CODE:
```
!pip install -q mediapipe
```

----------------------------------------

TITLE: Installing MediaPipe Library - Python
DESCRIPTION: This command installs the MediaPipe library using pip, which is a prerequisite for running face detection tasks. It ensures all necessary MediaPipe components are available in the Python environment.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_detector/python/face_detector.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
!pip install mediapipe
```

----------------------------------------

TITLE: Installing MediaPipe Library - Python
DESCRIPTION: This command installs the MediaPipe library using pip, the Python package installer. The `-q` flag ensures a quiet installation, suppressing verbose output. This is a prerequisite for using MediaPipe Tasks functionalities.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/audio_classifier/python/audio_classification.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
!pip install -q mediapipe
```

----------------------------------------

TITLE: Installing MediaPipe Library in Python
DESCRIPTION: This command installs the MediaPipe library using pip, the Python package installer. The -q flag ensures a quiet installation, suppressing verbose output. This is a prerequisite for using MediaPipe's functionalities.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/language_detector/python/[MediaPipe_Python_Tasks]_Language_Detector.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
!pip install -q mediapipe
```

----------------------------------------

TITLE: Loading SST-2 Dataset from TSV Files
DESCRIPTION: This snippet defines `CSVParams` to specify the text and label columns and the tab delimiter for TSV files. It then loads the training and validation datasets from their respective TSV files (`train.tsv` and `dev.tsv`) using `text_classifier.Dataset.from_csv`.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/text_classifier.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
csv_params = text_classifier.CSVParams(
    text_column='sentence', label_column='label', delimiter='\t')
train_data = text_classifier.Dataset.from_csv(
    filename=os.path.join(os.path.join(data_dir, 'train.tsv')),
    csv_params=csv_params)
validation_data = text_classifier.Dataset.from_csv(
    filename=os.path.join(os.path.join(data_dir, 'dev.tsv')),
    csv_params=csv_params)
```

----------------------------------------

TITLE: Training Average Word Embedding Text Classifier (Python)
DESCRIPTION: This snippet trains the text classifier using the `TextClassifier.create` function. It takes `train_data`, `validation_data`, and the previously defined `options` to create and train the model.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/text_classifier.ipynb#_snippet_6

LANGUAGE: Python
CODE:
```
model = text_classifier.TextClassifier.create(train_data, validation_data, options)
```

----------------------------------------

TITLE: Running MediaPipe Text Classification with Default Model in Python
DESCRIPTION: This command executes the `classify.py` Python script to perform text classification. It takes an input text string via the `--inputText` parameter and uses the default `classifier.tflite` model to classify the text's sentiment (positive or negative) and provide a confidence score.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/text_classification/raspberry_pi/README.md#_snippet_1

LANGUAGE: Python
CODE:
```
python3 classify.py --inputText "Your text goes here"
```

----------------------------------------

TITLE: Running MediaPipe Hand Landmarker Detection - Python
DESCRIPTION: This command executes the `detect.py` Python script, which initiates the real-time hand landmark detection using the MediaPipe Hand Landmarker. It uses default parameters for the model, number of hands, and confidence thresholds.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/raspberry_pi/README.md#_snippet_1

LANGUAGE: Python
CODE:
```
python3 detect.py
```

----------------------------------------

TITLE: Testing Base Model Inference - Hugging Face Transformers - Python
DESCRIPTION: This snippet demonstrates how to perform text generation inference using a pre-trained base model and the Hugging Face `pipeline` API. It initializes a text generation pipeline with a specified model and tokenizer, then generates a response to a given prompt with a maximum token limit.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_5

LANGUAGE: Python
CODE:
```
prompt = "What is the primary function of mitochondria within a cell?"
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
pipe(prompt, max_new_tokens=100)
```

----------------------------------------

TITLE: Initializing MediaPipe Interactive Segmenter Options
DESCRIPTION: This Python snippet initializes the `InteractiveSegmenter` options for MediaPipe. It sets the base model path to `model.tflite` and configures the segmenter to output a category mask, which is crucial for distinguishing foreground and background elements.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/interactive_segmentation/python/interactive_segmenter.ipynb#_snippet_7

LANGUAGE: python
CODE:
```
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers


BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white

RegionOfInterest = vision.InteractiveSegmenterRegionOfInterest
NormalizedKeypoint = containers.keypoint.NormalizedKeypoint

# Create the options that will be used for InteractiveSegmenter
base_options = python.BaseOptions(model_asset_path='model.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)
```

----------------------------------------

TITLE: Saving Fine-Tuned Model Weights - TRL Trainer - Python
DESCRIPTION: This snippet saves the weights of the fine-tuned model to a specified directory. The `save_model` method of the `SFTTrainer` instance persists the trained model's state, allowing it to be reloaded later for inference or further development.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_9

LANGUAGE: Python
CODE:
```
trainer.save_model("gemma3-1b-sft")
```

----------------------------------------

TITLE: Visualizing Face Blendshapes with MediaPipe in Python
DESCRIPTION: This snippet visualizes the detected face blendshapes from the MediaPipe FaceLandmarker result. It takes the first set of blendshapes from the `detection_result` and uses a helper function `plot_face_blendshapes_bar_graph` to display them, likely as a bar graph.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/python/[MediaPipe_Python_Tasks]_Face_Landmarker.ipynb#_snippet_7

LANGUAGE: python
CODE:
```
plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])
```

----------------------------------------

TITLE: Installing Dependencies with CocoaPods (Shell)
DESCRIPTION: This command initiates the installation of all required project dependencies using CocoaPods. It must be executed from the root directory of the project where the Podfile is located.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/ios/README.md#_snippet_0

LANGUAGE: Shell
CODE:
```
pod install
```

----------------------------------------

TITLE: Downloading MediaPipe Face Detector Model - Python
DESCRIPTION: This command downloads a pre-trained `blaze_face_short_range.tflite` model from Google's storage, which is required by MediaPipe's Face Detector. The model is saved locally as `detector.tflite` for subsequent use in the application.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_detector/python/face_detector.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
!wget -q -O detector.tflite -q https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
```

----------------------------------------

TITLE: Downloading EfficientDet Lite0 Model - Python
DESCRIPTION: This command downloads the 'efficientdet_lite0.tflite' model, a pre-trained TensorFlow Lite model optimized for object detection. It is fetched from a Google Cloud Storage bucket and saved locally as 'efficientdet.tflite', serving as the core model for MediaPipe's ObjectDetector.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/object_detection/python/object_detector.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
!wget -q -O efficientdet.tflite -q https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
```

----------------------------------------

TITLE: Configuring StableLM 3B Model Conversion in Python
DESCRIPTION: This line returns a `converter.ConversionConfig` object specifically for the StableLM 3B model. It defines the input checkpoint path, 'safetensors' format, model type, target backend, output directory, vocabulary file, and the final TFLite output file path, preparing the model for conversion.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_16

LANGUAGE: python
CODE:
```
return converter.ConversionConfig(input_ckpt=input_ckpt, ckpt_format='safetensors', model_type='STABLELM_4E1T_3B', backend=backend, output_dir=output_dir, combine_file_only=False, vocab_model_file=vocab_model_file, output_tflite_file=output_tflite_file)
```

----------------------------------------

TITLE: Downloading Exported MediaPipe TensorFlow Lite Model (Python)
DESCRIPTION: This snippet facilitates the download of the `gesture_recognizer.task` file, which is the exported TensorFlow Lite model, from the `exported_model` directory. This step is crucial for transferring the model to a local machine or an on-device application for deployment.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/gesture_recognizer.ipynb#_snippet_10

LANGUAGE: python
CODE:
```
files.download('exported_model/gesture_recognizer.task')
```

----------------------------------------

TITLE: Executing Face Stylizer Model Retraining in Python
DESCRIPTION: This snippet shows how to initiate the retraining process for the Face Stylizer model using the `create()` method. It takes the prepared `train_data` and the previously configured `face_stylizer_options` to fine-tune the model, requiring GPU resources for execution and potentially taking several minutes to hours.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/face_stylizer.ipynb#_snippet_7

LANGUAGE: Python
CODE:
```
face_stylizer_model = face_stylizer.FaceStylizer.create(
  train_data=data, options=face_stylizer_options
)
```

----------------------------------------

TITLE: Converting Model Checkpoints using Python Script
DESCRIPTION: This command executes the `convert.py` script to transform model checkpoints into a 'bins' folder. Users must specify the path to the input checkpoint file (`--ckpt_path`) and the desired output directory (`--output_path`).
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/tools/image_generator_converter/README.md#_snippet_1

LANGUAGE: Shell
CODE:
```
python3 convert.py --ckpt_path <ckpt_path> --output_path <output_path>
```

----------------------------------------

TITLE: Downloading Face Landmarker Model Bundle - Python
DESCRIPTION: This command downloads the face_landmarker_v2_with_blendshapes.task model bundle from Google Cloud Storage. This pre-trained model is essential for performing face landmark detection with MediaPipe Tasks. The -q flag ensures quiet download.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/python/[MediaPipe_Python_Tasks]_Face_Landmarker.ipynb#_snippet_2

LANGUAGE: Python
CODE:
```
!wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

----------------------------------------

TITLE: Running MediaPipe Face Landmarker with Custom Parameters - Python
DESCRIPTION: This command demonstrates how to run the `detect.py` script with optional parameters. It specifies the model file, sets the maximum number of faces to detect to 2, and adjusts the minimum confidence score for face detection to 0.5. This allows for customized behavior of the face landmarker.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/raspberry_pi/README.md#_snippet_2

LANGUAGE: Python
CODE:
```
python3 detect.py \
  --model face_landmarker.task \
  --numFaces 2 \
  --minFaceDetectionConfidence 0.5
```

----------------------------------------

TITLE: Running MediaPipe Hand Landmarker Detection with Custom Parameters - Python
DESCRIPTION: This command runs the `detect.py` Python script with specified optional parameters. It sets the `model` to `hand_landmarker.task`, limits `numHands` to 1, and adjusts the `minHandDetectionConfidence` to 0.5. These parameters allow for fine-tuning the detection process.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/raspberry_pi/README.md#_snippet_2

LANGUAGE: Python
CODE:
```
python3 detect.py \
  --model hand_landmarker.task \
  --numHands 1 \
  --minHandDetectionConfidence 0.5
```

----------------------------------------

TITLE: Generating Mask for LiteRT Model Input (Python)
DESCRIPTION: This function generates a mask for the input to a LiteRT model. It initializes a NumPy array with negative infinity and then sets elements below the k-th diagonal to zero, which is useful for attention mechanisms in models. It requires the `numpy` library.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_18

LANGUAGE: python
CODE:
```
def _get_mask(shape: Sequence[int], k: int):
  """Gets the mask for the input to the model.

  Args:
    shape: The shape of the mask input to the model.
    k: all elements below the k-th diagonal are set to 0.

  Returns:
    The mask for the input to the model. All the elements in the mask are set
    to -inf except that all the elements below the k-th diagonal are set to 0.
  """
  mask = np.ones(shape, dtype=np.float32) * float("-inf")
  mask = np.triu(mask, k=k)
  return mask
```

----------------------------------------

TITLE: Displaying Example Images per Class (Python)
DESCRIPTION: This snippet visualizes a specified number of example images from each identified class. It uses `matplotlib.pyplot` to display images, helping to visually confirm that the data is correctly classified and organized. The `NUM_EXAMPLES` constant controls how many images are shown per label.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb#_snippet_5

LANGUAGE: python
CODE:
```
NUM_EXAMPLES = 5

for label in labels:
  label_dir = os.path.join(image_path, label)
  example_filenames = os.listdir(label_dir)[:NUM_EXAMPLES]
  fig, axs = plt.subplots(1, NUM_EXAMPLES, figsize=(10,2))
  for i in range(NUM_EXAMPLES):
    axs[i].imshow(plt.imread(os.path.join(label_dir, example_filenames[i])))
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)
  fig.suptitle(f'Showing {NUM_EXAMPLES} examples for {label}')

plt.show()
```

----------------------------------------

TITLE: Downloading MediaPipe Text Classifier Model
DESCRIPTION: This command downloads an off-the-shelf BERT text classifier model (`bert_classifier.tflite`) from Google Cloud Storage. The `-O` flag specifies the output filename, and `-q` ensures quiet download. This TFLite model is essential for performing text classification.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/text_classification/python/text_classifier.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
!wget -O classifier.tflite -q https://storage.googleapis.com/mediapipe-models/text_classifier/bert_classifier/float32/1/bert_classifier.tflite
```

----------------------------------------

TITLE: Downloading YAMNet Audio Classifier Model - Python
DESCRIPTION: This command downloads the pre-trained YAMNet audio classification model (`.tflite` file) from Google Cloud Storage. The model is saved as `classifier.tflite` and is essential for performing audio classification with MediaPipe Tasks. The `-q` flag ensures quiet download.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/audio_classifier/python/audio_classification.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
!wget -O classifier.tflite -q https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite
```

----------------------------------------

TITLE: Running MediaPipe Face Detection with Custom Parameters (Python)
DESCRIPTION: This Python command runs the `detect.py` script, allowing users to customize the face detection behavior. It sets the TensorFlow Lite model to `detector.tflite`, adjusts the minimum detection confidence to `0.3`, and the minimum non-maximum suppression threshold to `0.5`.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_detector/raspberry_pi/README.md#_snippet_2

LANGUAGE: Python
CODE:
```
python3 detect.py \
  --model detector.tflite \
  --minDetectionConfidence 0.3 \
  --minSuppressionThreshold 0.5
```

----------------------------------------

TITLE: Running MediaPipe Pose Landmarker with Custom Parameters
DESCRIPTION: This command demonstrates how to run the `detect.py` script with several optional parameters to customize the pose detection process. It specifies a custom model file, limits the number of detected poses, adjusts the minimum confidence score for pose detection, and enables the visualization of segmentation masks.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/raspberry_pi/README.md#_snippet_2

LANGUAGE: Python
CODE:
```
python3 detect.py \
  --model pose_landmarker.task \
  --numPoses 1 \
  --minPoseDetectionConfidence 0.5\
  --outputSegmentationMasks
```

----------------------------------------

TITLE: Running MediaPipe Text Classification with Specified Model in Python
DESCRIPTION: This command runs the `classify.py` script, allowing the user to explicitly specify the TensorFlow Lite model to be used for classification via the `--model` parameter. It classifies the provided input text using the designated model, which must include metadata.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/text_classification/raspberry_pi/README.md#_snippet_2

LANGUAGE: Python
CODE:
```
python3 classify.py \
    --model classifier.tflite \
    --inputText "Your text goes here"
```

----------------------------------------

TITLE: Previewing Downloaded Images - Python
DESCRIPTION: This snippet defines a `resize_and_show` function to standardize image dimensions and displays the downloaded test image(s) using OpenCV. It helps verify that images are correctly loaded and prepared for subsequent processing by the MediaPipe model.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_stylizer/python/face_stylizer.ipynb#_snippet_4

LANGUAGE: Python
CODE:
```
import cv2
from google.colab.patches import cv2_imshow
import math

# Height and width that will be used by the model
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# Performs resizing and showing the image
def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2_imshow(img)


# Preview the image(s)
images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
for name, image in images.items():
  print(name)
  resize_and_show(image)
```

----------------------------------------

TITLE: Setting Up MediaPipe for Task Bundler in Colab
DESCRIPTION: This Python snippet sets up the environment by installing the `mediapipe` library using pip and imports necessary modules like `ipywidgets`, `IPython.display`, `google.colab.files`, and `mediapipe.tasks.python.genai.bundler`. It uses an `ipywidgets.Output` widget to display installation progress.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/bundling/llm_bundling.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
#@title Setup { display-mode: "form" }
import ipywidgets as widgets
from IPython.display import display
from google.colab import files
install_out = widgets.Output()
display(install_out)
with install_out:
  !pip install mediapipe
  from mediapipe.tasks.python.genai import bundler

install_out.clear_output()
with install_out:
  print("Setup done.")
```

----------------------------------------

TITLE: Installing MediaPipe Dependencies and Models (Shell)
DESCRIPTION: This shell script navigates to the project directory and executes the `setup.sh` script to install necessary dependencies and download pre-trained TensorFlow Lite models for the MediaPipe image classification example. This is a prerequisite for running the main classification script.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_classification/raspberry_pi/README.md#_snippet_0

LANGUAGE: Shell
CODE:
```
cd mediapipe/examples/image_classification/raspberry_pi
sh setup.sh
```

----------------------------------------

TITLE: Setting Up MediaPipe Text Classification on Raspberry Pi
DESCRIPTION: This snippet navigates into the specific project directory for text classification on Raspberry Pi and then executes the `setup.sh` script. This script is crucial for installing all necessary dependencies and downloading the required TensorFlow Lite models for the text classification demo.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/text_classification/raspberry_pi/README.md#_snippet_0

LANGUAGE: Bash
CODE:
```
cd mediapipe/examples/text_classification/raspberry_pi
sh setup.sh
```

----------------------------------------

TITLE: Evaluating BERT Text Classifier (Python)
DESCRIPTION: This snippet evaluates the performance of the trained BERT model on the `validation_data`. It calls `bert_model.evaluate()` to obtain the test loss and accuracy, which are then printed to demonstrate the model's performance.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/text_classifier.ipynb#_snippet_11

LANGUAGE: Python
CODE:
```
metrics = bert_model.evaluate(validation_data)
print(f'Test loss:{metrics[0]}, Test accuracy:{metrics[1]}')
```

----------------------------------------

TITLE: Installing MediaPipe Model Maker Package
DESCRIPTION: This snippet installs the necessary MediaPipe Model Maker package and upgrades pip. It is a prerequisite for customizing on-device machine learning models and ensures all required dependencies are available.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/gesture_recognizer.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
!pip install --upgrade pip
!pip install mediapipe-model-maker
```

----------------------------------------

TITLE: Installing Required Python Packages
DESCRIPTION: This snippet upgrades pip and installs the necessary Python packages for MediaPipe Model Maker, including 'keras<3.0.0' and 'mediapipe-model-maker', which are essential for training custom object detection models.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/tutorials/object_detection/Object_Detection_for_3_dogs.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
!pip install --upgrade pip
!pip install 'keras<3.0.0' mediapipe-model-maker
```

----------------------------------------

TITLE: Importing Required Libraries - Python
DESCRIPTION: This snippet imports essential Python libraries for the project, including `interpreter_lib` from `ai_edge_litert` for model interpretation, `AutoTokenizer` from `transformers` for text processing, and standard libraries like `numpy`, `collections.abc.Sequence`, and `sys`.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma2_tflite.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
from ai_edge_litert import interpreter as interpreter_lib
from transformers import AutoTokenizer
import numpy as np
from collections.abc import Sequence
import sys
```

----------------------------------------

TITLE: Importing MediaPipe Modules - Python
DESCRIPTION: This snippet imports the necessary MediaPipe modules for image classification, including `mediapipe` itself, `mediapipe.tasks.python` for task-specific functionalities, and `mediapipe.tasks.python.vision` for vision-related tasks. These modules are prerequisites for initializing the image classifier.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_classification/python/image_classifier.ipynb#_snippet_7

LANGUAGE: python
CODE:
```
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision
```

----------------------------------------

TITLE: Installing MediaPipe Model Maker Dependencies in Python
DESCRIPTION: This snippet installs necessary Python libraries for customizing MediaPipe models. It first checks the Python version, then upgrades pip to its latest version, and finally installs the `mediapipe-model-maker` package, which is essential for model customization tasks.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb#_snippet_1

LANGUAGE: Python
CODE:
```
!python --version
!pip install --upgrade pip
!pip install mediapipe-model-maker
```

----------------------------------------

TITLE: Installing MediaPipe Model Maker Libraries (Python)
DESCRIPTION: This snippet installs the necessary Python libraries for customizing MediaPipe models within a Colab environment. It first checks the Python version, then upgrades pip to its latest version, and finally installs the `mediapipe-model-maker` package, which is essential for retraining object detection models.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_1

LANGUAGE: Python
CODE:
```
!python --version
!pip install --upgrade pip
!pip install mediapipe-model-maker
```

----------------------------------------

TITLE: Running MediaPipe Audio Classifier (Python)
DESCRIPTION: This command executes the main Python script for the audio classification example. It runs the classifier with default parameters, processing audio streamed from the microphone to perform real-time classification.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/audio_classifier/raspberry_pi/README.md#_snippet_1

LANGUAGE: Python
CODE:
```
python3 classify.py
```

----------------------------------------

TITLE: Running MediaPipe Image Classifier (Python)
DESCRIPTION: This command executes the `classify.py` Python script, which runs the MediaPipe image classification example using default settings. It processes images streamed from the camera and performs real-time classification.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_classification/raspberry_pi/README.md#_snippet_1

LANGUAGE: Python
CODE:
```
python3 classify.py
```

----------------------------------------

TITLE: Running MediaPipe Pose Landmarker Detection Script
DESCRIPTION: This command executes the main Python script, `detect.py`, which performs real-time pose landmark detection using the camera stream. It initiates the MediaPipe pipeline with default settings to process video frames and identify human poses.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/raspberry_pi/README.md#_snippet_1

LANGUAGE: Python
CODE:
```
python3 detect.py
```

----------------------------------------

TITLE: Setting Up MediaPipe Pose Landmarker on Raspberry Pi
DESCRIPTION: This command sequence navigates into the specific project directory for the MediaPipe Pose Landmarker Raspberry Pi example and then executes the `setup.sh` script. The setup script is responsible for installing all required dependencies and downloading the necessary task file for the pose landmarker model.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/raspberry_pi/README.md#_snippet_0

LANGUAGE: Bash
CODE:
```
cd mediapipe/examples/pose_landmarker/raspberry_pi
sh setup.sh
```

----------------------------------------

TITLE: Downloading Image Embedder Model (Python)
DESCRIPTION: This snippet downloads a pre-trained `mobilenet_v3_small.tflite` model, which is an off-the-shelf image embedder. The `wget` command saves the model as `embedder.tflite` for subsequent use with MediaPipe's Image Embedder.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_embedder/python/image_embedder.ipynb#_snippet_2

LANGUAGE: Python
CODE:
```
!wget -O embedder.tflite -q https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite
```

----------------------------------------

TITLE: Downloading MediaPipe Text Embedder Model - Python
DESCRIPTION: This command downloads the `bert_embedder.tflite` model from Google Cloud Storage, which is essential for performing text embedding with MediaPipe. The model is saved as `embedder.tflite` in the current directory, and the `-q` flag suppresses output.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/text_embedder/python/text_embedder.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
#@title Start downloading here.
!wget -O embedder.tflite -q https://storage.googleapis.com/mediapipe-models/text_embedder/bert_embedder/float32/1/bert_embedder.tflite
```

----------------------------------------

TITLE: Downloading and Displaying Test Image - Python
DESCRIPTION: This snippet first downloads a test image named image.png from Google Cloud Storage using wget. It then uses OpenCV (cv2) to read the image and cv2_imshow (from google.colab.patches) to display it within the Colab environment. This image serves as input for subsequent face landmark detection.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/python/[MediaPipe_Python_Tasks]_Face_Landmarker.ipynb#_snippet_4

LANGUAGE: Python
CODE:
```
!wget -q -O image.png https://storage.googleapis.com/mediapipe-assets/business-person.png

import cv2
from google.colab.patches import cv2_imshow

img = cv2.imread("image.png")
cv2_imshow(img)
```

----------------------------------------

TITLE: Toggling Hugging Face Token Visibility
DESCRIPTION: A callback function `on_change_model` that controls the visibility of the Hugging Face token input field. If a 'gated' model (like Gemma 2B or Gemma 7B) is selected, the token input is displayed; otherwise, it is hidden, prompting the user for necessary authentication.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_5

LANGUAGE: Python
CODE:
```
def on_change_model(change):
  selected_values = ['Gemma 2B','Gemma 7B']

  if change['new'] in selected_values:
    token.layout.display = 'flex'
  else:
    token.layout.display = 'none'
```

----------------------------------------

TITLE: Downloading StableLM 3B Model Files
DESCRIPTION: Downloads specific files for the StableLM 3B model from Hugging Face. It fetches tokenizer files and the model safetensors into a local directory using `hf_hub_download`. This model does not require a Hugging Face token.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_10

LANGUAGE: Python
CODE:
```
def stablelm_download():
  REPO_ID = "stabilityai/stablelm-3b-4e1t"
  FILENAMES = ["tokenizer.json", "tokenizer_config.json", "model.safetensors"]
  with out:
    for filename in FILENAMES:
      hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir="./stablelm-3b-4e1t")
```

----------------------------------------

TITLE: Running MediaPipe Face Landmarker Detection - Python
DESCRIPTION: This command executes the main Python script `detect.py` to start the face landmarker detection. By default, it uses `face_landmarker.task` and detects one face with default confidence thresholds. This is the basic command to run the application.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/raspberry_pi/README.md#_snippet_1

LANGUAGE: Python
CODE:
```
python3 detect.py
```

----------------------------------------

TITLE: Downloading and Displaying Test Image - Python
DESCRIPTION: This snippet downloads a sample image (`image.jpg`) from a URL and then uses OpenCV (`cv2`) and `google.colab.patches.cv2_imshow` to load and display the image within a Colab environment. This image serves as input for demonstrating the face detection capabilities.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_detector/python/face_detector.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
!curl https://i.imgur.com/Vu2Nqwb.jpeg -s -o image.jpg

IMAGE_FILE = 'image.jpg'

import cv2
from google.colab.patches import cv2_imshow

img = cv2.imread(IMAGE_FILE)
cv2_imshow(img)
```

----------------------------------------

TITLE: Initializing Model, Backend, and Token Selection Widgets
DESCRIPTION: Initializes `ipywidgets.Dropdown` for model and backend selection, and `ipywidgets.Password` for Hugging Face token input. These widgets allow users to interactively choose an LLM, its target backend (CPU/GPU), and provide a necessary authentication token for gated models like Gemma.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_2

LANGUAGE: Python
CODE:
```
model = widgets.Dropdown(
    options=["Gemma 2B","Gemma 7B", "Falcon 1B", "StableLM 3B", "Phi 2"],
    value='Gemma 2B',
    description='model',
    disabled=False,
)

backend = widgets.Dropdown(
    options=["cpu", "gpu"],
    value='cpu',
    description='backend',
    disabled=False,
)

token = widgets.Password(
    value='',
    placeholder='huggingface token',
    description='HF token:',
    disabled=False
)
```

----------------------------------------

TITLE: Visualizing the Stylized Training Image
DESCRIPTION: This snippet loads the downloaded stylized image using `image_utils.load_image`, converts it from RGB to BGR format using OpenCV (`cv2`), and then displays it within the Google Colab environment using `cv2_imshow`. This step allows users to visually verify the input style image before proceeding with model training.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/face_stylizer.ipynb#_snippet_4

LANGUAGE: Python
CODE:
```
import cv2
from google.colab.patches import cv2_imshow

style_image_tensor = image_utils.load_image(style_image_path)
style_cv_image = cv2.cvtColor(style_image_tensor.numpy(), cv2.COLOR_RGB2BGR)
cv2_imshow(style_cv_image)
```

----------------------------------------

TITLE: Cloning MediaPipe Samples Repository - Bash
DESCRIPTION: This command downloads the MediaPipe samples demo code from the official GitHub repository. It is the initial step to obtain the source code for the LLM Inference Android demo app, enabling local setup and development.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/android/README.md#_snippet_0

LANGUAGE: Bash
CODE:
```
git clone https://github.com/google-ai-edge/mediapipe-samples
```

----------------------------------------

TITLE: Downloading Gemma2-2B-IT Model - Python
DESCRIPTION: This snippet downloads the Gemma2-2B-IT TFLite model file from HuggingFace Hub using `hf_hub_download`. It specifies the repository ID and the exact filename, storing the local path to the downloaded model in `model_path`.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma2_tflite.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="litert-community/Gemma2-2B-IT", filename="gemma2_q8_seq128_ekv1280.tflite")
```

----------------------------------------

TITLE: Downloading Gemma Tokenizer Model for MediaPipe (Python)
DESCRIPTION: This snippet downloads the `tokenizer.model` file for the Gemma 3.1B model from Hugging Face Hub. It uses `hf_hub_download` to save the tokenizer locally to the `/content` directory, which is a prerequisite for building the MediaPipe LLM task bundle.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_30

LANGUAGE: python
CODE:
```
from huggingface_hub import hf_hub_download
import joblib

REPO_ID = "google/gemma-3-1b"
FILENAME = "tokenizer.model"
tokenizer_model = (
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir="/content")
)
```

----------------------------------------

TITLE: Running Local HTTP Server with Python 3
DESCRIPTION: This command initiates a basic HTTP server using Python 3's `http.server` module. It hosts files from the current directory on port 8000, enabling local access to the MediaPipe LLM Inference web demo's HTML and JavaScript files. This is a prerequisite for running the demo locally.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/js/README.md#_snippet_0

LANGUAGE: Python
CODE:
```
python3 -m http.server 8000
```

----------------------------------------

TITLE: Setting Up Environment for MediaPipe LLM Conversion - Python
DESCRIPTION: This snippet sets up the Python environment by installing `mediapipe`, `torch`, and `huggingface_hub` using pip. It then imports necessary modules like `ipywidgets`, `IPython.display`, `os`, `hf_hub_download` from `huggingface_hub`, and `converter` from `mediapipe.tasks.python.genai`. This prepares the environment for LLM conversion tasks.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
# @title Setup { display-mode: "form" }
# @markdown import ipywidgets\
# @markdown import IPython.display\
# @markdown import os\
# @markdown import huggingface downloader\
# @markdown import mediapipe genai converter
import ipywidgets as widgets
from IPython.display import display
install_out = widgets.Output()
display(install_out)
with install_out:
  !pip install mediapipe
  !pip install torch
  !pip install huggingface_hub
  import os
  from huggingface_hub import hf_hub_download
  from mediapipe.tasks.python.genai import converter

install_out.clear_output()
with install_out:
  print("Setup done.")
```

----------------------------------------

TITLE: Initializing LiteRTLlmPipeline in Python
DESCRIPTION: This constructor initializes the `LiteRTLlmPipeline` class, setting up the interpreter and tokenizer. It also identifies and initializes the 'decode' signature runner, while the 'prefill' runner is initialized dynamically later.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma2_tflite.ipynb#_snippet_5

LANGUAGE: Python
CODE:
```
class LiteRTLlmPipeline:

  def __init__(self, interpreter, tokenizer):
    """Initializes the pipeline."""
    self._interpreter = interpreter
    self._tokenizer = tokenizer

    self._prefill_runner = None
    self._decode_runner = self._interpreter.get_signature_runner("decode")
```

----------------------------------------

TITLE: Downloading DeepLab v3 Image Segmenter Model
DESCRIPTION: This command downloads the pre-trained DeepLab v3 TFLite model from Google's storage. This model is specifically designed for image segmentation and is a prerequisite for initializing the MediaPipe ImageSegmenter.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_segmentation/python/image_segmentation.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
#@title Start downloading here.
!wget -O deeplabv3.tflite -q https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite
```

----------------------------------------

TITLE: Initializing LiteRT LLM Pipeline - Python
DESCRIPTION: This snippet demonstrates the instantiation of the LiteRTLlmPipeline class. It requires an 'interpreter' (likely for model execution) and a 'tokenizer' (for text processing) as arguments, setting up the necessary components for the LLM pipeline.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_28

LANGUAGE: python
CODE:
```
pipeline = LiteRTLlmPipeline(interpreter, tokenizer)
```

----------------------------------------

TITLE: Downloading Pre-trained Image Classification Model (Python)
DESCRIPTION: This command downloads a pre-trained efficientnet_lite0 image classification model in TFLite format from Google Cloud Storage. The model is saved as classifier.tflite and is essential for performing image classification inference.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_classification/python/image_classifier.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
!wget -O classifier.tflite -q https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite
```

----------------------------------------

TITLE: Plotting Example Images from Gesture Dataset
DESCRIPTION: This snippet visualizes a specified number of example images for each gesture label in the dataset. It helps in understanding the dataset content and verifying the image quality and label association before model training.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/gesture_recognizer.ipynb#_snippet_5

LANGUAGE: python
CODE:
```
NUM_EXAMPLES = 5

for label in labels:
  label_dir = os.path.join(dataset_path, label)
  example_filenames = os.listdir(label_dir)[:NUM_EXAMPLES]
  fig, axs = plt.subplots(1, NUM_EXAMPLES, figsize=(10,2))
  for i in range(NUM_EXAMPLES):
    axs[i].imshow(plt.imread(os.path.join(label_dir, example_filenames[i])))
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)
  fig.suptitle(f'Showing {NUM_EXAMPLES} examples for {label}')

plt.show()
```

----------------------------------------

TITLE: Installing MediaPipe Library - Python
DESCRIPTION: This command installs the MediaPipe library, a prerequisite for using MediaPipe Tasks for text embedding. The `-q` flag ensures a quiet installation, suppressing verbose output during the process.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/text_embedder/python/text_embedder.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
!pip install -q mediapipe
```

----------------------------------------

TITLE: Installing Python Dependencies with pip
DESCRIPTION: This command installs all necessary Python packages required for the MediaPipe samples. It includes libraries for deep learning (torch, pytorch_lightning), data manipulation (numpy, Pillow), and utilities (typing_extensions, requests, absl-py).
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/tools/image_generator_converter/README.md#_snippet_0

LANGUAGE: Shell
CODE:
```
pip install torch typing_extensions numpy Pillow requests pytorch_lightning absl-py
```

----------------------------------------

TITLE: Installing MediaPipe Python Package
DESCRIPTION: This command installs the MediaPipe Python library using pip. The `-q` flag ensures a quiet installation, suppressing verbose output. MediaPipe is a prerequisite for using its text classification capabilities.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/text_classification/python/text_classifier.ipynb#_snippet_1

LANGUAGE: python
CODE:
```
!pip install -q mediapipe
```

----------------------------------------

TITLE: Setting Up MediaPipe Object Detection Example on Raspberry Pi
DESCRIPTION: This shell script navigates to the project directory and executes `setup.sh` to install necessary dependencies and download TensorFlow Lite models for the MediaPipe object detection example on Raspberry Pi.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/object_detection/raspberry_pi/README.md#_snippet_0

LANGUAGE: Shell
CODE:
```
cd mediapipe/examples/object_detection/raspberry_pi
sh setup.sh
```

----------------------------------------

TITLE: Installing MediaPipe Dependencies and Models (Shell)
DESCRIPTION: This shell script navigates to the project directory and executes the `setup.sh` script to install required dependencies and download TensorFlow Lite models for the MediaPipe face detection example on Raspberry Pi.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_detector/raspberry_pi/README.md#_snippet_0

LANGUAGE: Shell
CODE:
```
cd mediapipe/examples/face_detection/raspberry_pi
sh setup.sh
```

----------------------------------------

TITLE: Installing LiteRT Pipeline Tools
DESCRIPTION: This command installs the LiteRT pipeline tools directly from the `ai-edge-apis` GitHub repository, specifically targeting the `litert_tools` subdirectory.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma3_1b_tflite.ipynb#_snippet_1

LANGUAGE: Python
CODE:
```
!pip install git+https://github.com/google-ai-edge/ai-edge-apis.git#subdirectory=litert_tools
```

----------------------------------------

TITLE: Downloading MediaPipe Language Detector Model
DESCRIPTION: This command downloads the 'language_detector.tflite' model from Google Cloud Storage using 'wget'. The -O flag specifies the output filename as 'detector.tflite', and -q ensures quiet execution. This model is essential for performing language detection with MediaPipe.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/language_detector/python/[MediaPipe_Python_Tasks]_Language_Detector.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
!wget -O detector.tflite -q https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/latest/language_detector.tflite
```

----------------------------------------

TITLE: Downloading Face Stylizer Model - Python
DESCRIPTION: This command downloads the pre-trained `face_stylizer.task` model from a Google Cloud Storage URL. This model contains the pre-determined style used for face stylization and is essential for the inference process.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_stylizer/python/face_stylizer.ipynb#_snippet_2

LANGUAGE: Python
CODE:
```
#@title Start downloading here.
!wget -O face_stylizer.task -q https://storage.googleapis.com/mediapipe-models/face_stylizer/blaze_face_stylizer/float32/latest/face_stylizer_color_sketch.task
```

----------------------------------------

TITLE: Downloading MediaPipe Pose Landmarker Model
DESCRIPTION: This command downloads the 'pose_landmarker_heavy.task' model bundle from a Google Cloud Storage URL. This pre-trained model is essential for performing pose landmark detection with MediaPipe Tasks. The '-q' flag suppresses output, and '-O' specifies the output filename.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/python/[MediaPipe_Python_Tasks]_Pose_Landmarker.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
!wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

----------------------------------------

TITLE: Downloading MediaPipe Hand Landmarker Model - Python
DESCRIPTION: This command downloads the pre-trained 'hand_landmarker.task' model bundle from Google Cloud Storage. This model is essential for the MediaPipe HandLandmarker API to perform hand landmark detection. The '-q' flag ensures a quiet download.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#_snippet_2

LANGUAGE: Python
CODE:
```
!wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

----------------------------------------

TITLE: Downloading Gesture Recognizer Model - Python
DESCRIPTION: This command downloads an off-the-shelf MediaPipe Gesture Recognizer model from Google Cloud Storage using `wget`. The `-q` flag ensures quiet download. This `.task` file contains the pre-trained model weights and metadata required for gesture recognition.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/gesture_recognizer/python/gesture_recognizer.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
!wget -q https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

----------------------------------------

TITLE: Downloading Interactive Segmenter Model
DESCRIPTION: This command downloads the `magic_touch.tflite` interactive segmentation model from Google Cloud Storage. The model is saved as `model.tflite` in the current directory, and the `-q` flag ensures quiet download progress.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/interactive_segmentation/python/interactive_segmenter.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
#@title Start downloading here.
!wget -O model.tflite -q https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite
```

----------------------------------------

TITLE: Updating Backend Options Based on Model Selection
DESCRIPTION: A callback function `on_use_gpu` that updates the `backend` dropdown options based on the currently selected model. It uses the `options_mapping` to filter available backends, ensuring only compatible options are presented to the user. If the model is not in the mapping, it clears token options.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_4

LANGUAGE: Python
CODE:
```
def on_use_gpu(change):
  selected_value = change['new']

  if selected_value in options_mapping:
    backend.options = options_mapping[selected_value]
    backend.value = options_mapping[selected_value][0]
  else:
    token.options = []
    token.value = None
```

----------------------------------------

TITLE: Setting up MediaPipe Gesture Recognizer on Raspberry Pi
DESCRIPTION: This command sequence navigates to the project directory and executes the setup script. The `setup.sh` script installs necessary dependencies and downloads the MediaPipe task file required for the gesture recognizer, preparing the environment for execution.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/gesture_recognizer/raspberry_pi/README.md#_snippet_0

LANGUAGE: Shell
CODE:
```
cd mediapipe/examples/gesture_recognizer/raspberry_pi
sh setup.sh
```

----------------------------------------

TITLE: Installing MediaPipe Audio Classifier Dependencies (Shell)
DESCRIPTION: This command sequence navigates to the project directory and executes the `setup.sh` script. The script is responsible for installing necessary dependencies and downloading the TensorFlow Lite models required for the audio classification example.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/audio_classifier/raspberry_pi/README.md#_snippet_0

LANGUAGE: Shell
CODE:
```
cd mediapipe/examples/audio_classifier/raspberry_pi
sh setup.sh
```

----------------------------------------

TITLE: Setting up MediaPipe Hand Landmarker Example on Raspberry Pi - Shell
DESCRIPTION: This shell script navigates to the MediaPipe hand landmarker example directory on the Raspberry Pi and then executes the `setup.sh` script. The `setup.sh` script is responsible for installing required dependencies and downloading the necessary task files for the hand landmarker.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/raspberry_pi/README.md#_snippet_0

LANGUAGE: Shell
CODE:
```
cd mediapipe/examples/hand_landmarker/raspberry_pi
sh setup.sh
```

----------------------------------------

TITLE: Handling Model Download and Conversion Button Click in Python
DESCRIPTION: This `on_button_clicked` function serves as a callback for an interactive button, orchestrating the download and conversion of selected AI models. It updates UI elements to reflect progress, calls specific download and conversion configuration functions based on the chosen model, and handles potential errors by resetting the UI and displaying messages.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_18

LANGUAGE: python
CODE:
```
def on_button_clicked(b):
  try:
    out.clear_output()
    with out:
      print("Downloading model ...")
    button.description = "Downloading ..."
    button.disabled = True
    model.disabled = True
    backend.disabled = True

    if model.value == 'Gemma 2B':
      gemma2b_download(token.value)
    elif model.value == 'Gemma 7B':
      gemma7b_download(token.value)
    elif model.value == 'Falcon 1B':
      falcon_download()
    elif model.value == 'StableLM 3B':
      stablelm_download()
    elif model.value == 'Phi 2':
      phi2_download()
    else:
      raise Exception("Invalid model")

    with out:
      print("Done")
      print("Converting model ...")

    button.description = "Converting ..."

    if model.value == 'Gemma 2B':
      config = gemma2b_convert_config(backend.value)
    elif model.value == 'Gemma 7B':
      config = gemma7b_convert_config(backend.value)
    elif model.value == 'Falcon 1B':
      config = falcon_convert_config(backend.value)
    elif model.value == 'StableLM 3B':
      config = stablelm_convert_config(backend.value)
    elif model.value == 'Phi 2':
      config = phi2_convert_config(backend.value)
    else:
      with out:
        raise Exception("Invalid model")
      return

    with out:
      converter.convert_checkpoint(config)
      print("Done")

    button.description = "Start Conversion"
    button.disabled = False
    model.disabled = False
    backend.disabled = False

  except Exception as e:
    button.description = "Start Conversion"
    button.disabled = False
    model.disabled = False
    backend.disabled = False
    with out:
      print(e)
```

----------------------------------------

TITLE: Importing Libraries for Face Stylizer Customization
DESCRIPTION: This snippet imports necessary Python libraries for the face stylizer customization process. It includes `google.colab.files` for Colab-specific file operations, `os` for interacting with the operating system, `tensorflow` for ML operations (asserting version 2.x), and specific modules from `mediapipe_model_maker` (`face_stylizer`, `image_utils`) for model customization and image utilities, along with `matplotlib.pyplot` for plotting.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/face_stylizer.ipynb#_snippet_2

LANGUAGE: Python
CODE:
```
from google.colab import files
import os
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import face_stylizer
from mediapipe_model_maker import image_utils

import matplotlib.pyplot as plt
```

----------------------------------------

TITLE: Downloading Test Image for Segmentation
DESCRIPTION: This Python script downloads a sample image from MediaPipe assets to be used as input for the segmentation demo. It uses `urllib.request.urlretrieve` to fetch the image and stores its filename in the `IMAGE_FILENAMES` array, which can be extended for multiple images.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_segmentation/python/image_segmentation.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
import urllib

IMAGE_FILENAMES = ['segmentation_input_rotation0.jpg']

for name in IMAGE_FILENAMES:
  url = f'https://storage.googleapis.com/mediapipe-assets/{name}'
  urllib.request.urlretrieve(url, name)
```

----------------------------------------

TITLE: Downloading Test Images for MediaPipe Gesture Recognition (Python)
DESCRIPTION: This snippet downloads a predefined set of test images from a Google Cloud Storage bucket. It uses the `urllib.request.urlretrieve` function to fetch each image and save it locally. The `IMAGE_FILENAMES` list specifies the names of the images to be downloaded.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/gesture_recognizer/python/gesture_recognizer.ipynb#_snippet_4

LANGUAGE: Python
CODE:
```
import urllib

IMAGE_FILENAMES = ['thumbs_down.jpg', 'victory.jpg', 'thumbs_up.jpg', 'pointing_up.jpg']

for name in IMAGE_FILENAMES:
  url = f'https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/{name}'
  urllib.request.urlretrieve(url, name)
```

----------------------------------------

TITLE: Defining Model-Backend Compatibility Mapping
DESCRIPTION: Defines a dictionary `options_mapping` that specifies which backends (CPU or GPU) are compatible with each large language model. This mapping is used to dynamically update the available backend options based on the selected model.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_3

LANGUAGE: Python
CODE:
```
options_mapping = {
              'Gemma 2B': ['cpu', 'gpu'],
              'Gemma 7B': ['gpu'],
              'Falcon 1B': ['cpu', 'gpu'],
              'StableLM 3B': ['cpu', 'gpu'],
              'Phi 2': ['cpu', 'gpu']
}
```

----------------------------------------

TITLE: Optional Image Upload for Testing - Python
DESCRIPTION: This commented-out Python snippet provides functionality to upload a custom image from the local system using google.colab.files.upload(). It iterates through uploaded files, saves their content, and identifies the first uploaded file's name. This allows users to test the face landmark detection with their own images instead of the default one.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/python/[MediaPipe_Python_Tasks]_Face_Landmarker.ipynb#_snippet_5

LANGUAGE: Python
CODE:
```
# from google.colab import files
# uploaded = files.upload()

# for filename in uploaded:
#   content = uploaded[filename]
#   with open(filename, 'wb') as f:
#     f.write(content)

# if len(uploaded.keys()):
#   IMAGE_FILE = next(iter(uploaded))
#   print('Uploaded file:', IMAGE_FILE)
```

----------------------------------------

TITLE: Installing MediaPipe Dependencies and Downloading Task File - Shell
DESCRIPTION: This shell script navigates to the project directory and executes the `setup.sh` script. The `setup.sh` script is responsible for installing required dependencies and downloading the necessary task file for the MediaPipe face landmarker example on Raspberry Pi.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/raspberry_pi/README.md#_snippet_0

LANGUAGE: Shell
CODE:
```
cd mediapipe/examples/face_landmarker/raspberry_pi
sh setup.sh
```

----------------------------------------

TITLE: Installing AI Edge Torch and MediaPipe (Python)
DESCRIPTION: This snippet installs `ai-edge-torch` directly from its GitHub repository, along with `ai-edge-litert` and `mediapipe`. These libraries are crucial for converting the fine-tuned model to a LiteRT format and for on-device deployment and inference using MediaPipe.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
! pip install git+https://github.com/google-ai-edge/ai-edge-torch
! pip install ai-edge-litert
! pip install mediapipe
```

----------------------------------------

TITLE: Downloading and Displaying Test Image - Python
DESCRIPTION: This snippet downloads a sample image from Google Cloud Storage and then displays it using OpenCV and `cv2_imshow`, which is typically used in environments like Google Colab. This image serves as the input for the hand landmark detection process.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#_snippet_4

LANGUAGE: Python
CODE:
```
!wget -q -O image.jpg https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg

import cv2
from google.colab.patches import cv2_imshow

img = cv2.imread("image.jpg")
cv2_imshow(img)
```

----------------------------------------

TITLE: Displaying and Resizing Images (Python)
DESCRIPTION: This snippet defines a utility function `resize_and_show` that resizes an image while maintaining its aspect ratio and then displays it using `cv2_imshow`. It then loads and displays the previously downloaded `burger.jpg` and `burger_crop.jpg` images to visually confirm their content.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_embedder/python/image_embedder.ipynb#_snippet_4

LANGUAGE: Python
CODE:
```
import cv2
from google.colab.patches import cv2_imshow
import math

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2_imshow(img)


# Preview the images.
images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
for name, image in images.items():
  print(name)
  resize_and_show(image)
```

----------------------------------------

TITLE: Downloading SST-2 Sentiment Analysis Dataset
DESCRIPTION: This code downloads the SST-2 (Stanford Sentiment Treebank) dataset, a ZIP file containing movie reviews for sentiment analysis, and extracts it to a local directory. This dataset is used for training and testing text classification models.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/text_classifier.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
data_path = tf.keras.utils.get_file(
    fname='SST-2.zip',
    origin='https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
    extract=True)
data_dir = os.path.join(os.path.dirname(data_path), 'SST-2')  # folder name
```

----------------------------------------

TITLE: Importing MediaPipe Model Maker Quantization Module (Python)
DESCRIPTION: Imports the `quantization` module from `mediapipe_model_maker`, which provides functionalities for post-training model quantization. This is a prerequisite for applying quantization to a retrained model.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb#_snippet_14

LANGUAGE: Python
CODE:
```
from mediapipe_model_maker import quantization
```

----------------------------------------

TITLE: Optional Image Upload for Object Detection - Python
DESCRIPTION: This commented-out code block provides an optional mechanism for users to upload their own image files directly into the Google Colab environment. It utilizes `google.colab.files` to handle the file upload, writes the content to a local file, and updates the `IMAGE_FILE` variable to point to the newly uploaded image.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/object_detection/python/object_detector.ipynb#_snippet_5

LANGUAGE: python
CODE:
```
# from google.colab import files
# uploaded = files.upload()

# for filename in uploaded:
#   content = uploaded[filename]
#   with open(filename, 'wb') as f:
#     f.write(content)

# if len(uploaded.keys()):
#   IMAGE_FILE = next(iter(uploaded))
#   print('Uploaded file:', IMAGE_FILE)
```

----------------------------------------

TITLE: Downloading and Displaying Test Image in Python
DESCRIPTION: This snippet first downloads a sample image named 'image.jpg' from Pixabay, which will be used as input for the pose detection model. Subsequently, it uses the OpenCV library (`cv2`) to read the downloaded image and then displays it within the Google Colab environment using `cv2_imshow`.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/python/[MediaPipe_Python_Tasks]_Pose_Landmarker.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
!wget -q -O image.jpg https://cdn.pixabay.com/photo/2019/03/12/20/39/girl-4051811_960_720.jpg

import cv2
from google.colab.patches import cv2_imshow

img = cv2.imread("image.jpg")
cv2_imshow(img)
```

----------------------------------------

TITLE: Downloading and Displaying Test Image - Python
DESCRIPTION: This snippet downloads a sample image, 'cat_and_dog.jpg', from a Google Cloud Storage URL, saving it locally. It then uses OpenCV (`cv2`) to read the downloaded image and displays it within the notebook environment using `cv2_imshow`, preparing it for subsequent object detection tasks.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/object_detection/python/object_detector.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
!wget -q -O image.jpg https://storage.googleapis.com/mediapipe-tasks/object_detector/cat_and_dog.jpg

IMAGE_FILE = 'image.jpg'

import cv2
from google.colab.patches import cv2_imshow

img = cv2.imread(IMAGE_FILE)
cv2_imshow(img)
```

----------------------------------------

TITLE: Importing Required Libraries for Gesture Recognition
DESCRIPTION: This snippet imports essential Python libraries for the gesture recognition task, including `files` and `os` for file operations, `tensorflow` for ML operations, `gesture_recognizer` from `mediapipe_model_maker`, and `matplotlib.pyplot` for plotting.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/gesture_recognizer.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
from google.colab import files
import os
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import gesture_recognizer

import matplotlib.pyplot as plt
```

----------------------------------------

TITLE: Importing Libraries for MediaPipe Object Detector Customization (Python)
DESCRIPTION: This snippet imports the required Python modules for object detection model customization, typically used in a Google Colab environment. It includes `google.colab.files` for file operations, `os` and `json` for general utilities, `tensorflow` for ML operations (asserting version 2.x), and `mediapipe_model_maker.object_detector` for the core model customization functionalities.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_2

LANGUAGE: Python
CODE:
```
from google.colab import files
import os
import json
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import object_detector
```

----------------------------------------

TITLE: Importing MediaPipe Model Maker and Dependencies
DESCRIPTION: This snippet imports essential Python libraries: 'os' for operating system interaction, 'tensorflow' (asserting version 2.x), 'google.colab.files' for Colab-specific file operations, and 'object_detector' from 'mediapipe_model_maker' for object detection functionalities.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/tutorials/object_detection/Object_Detection_for_3_dogs.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
import os
import tensorflow as tf
assert tf.__version__.startswith('2')
from google.colab import files

from mediapipe_model_maker import object_detector
```

----------------------------------------

TITLE: Downloading Sample Images (Python)
DESCRIPTION: This snippet downloads two sample image files, `burger.jpg` and `burger_crop.jpg`, from Google Cloud Storage. It uses the `urllib.request` module to retrieve the images, which are then used as input for the image embedding process.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_embedder/python/image_embedder.ipynb#_snippet_3

LANGUAGE: Python
CODE:
```
import urllib

IMAGE_FILENAMES = ['burger.jpg', 'burger_crop.jpg']

for name in IMAGE_FILENAMES:
  url = f'https://storage.googleapis.com/mediapipe-assets/{name}'
  urllib.request.urlretrieve(url, name)
```

----------------------------------------

TITLE: Importing Required Libraries for MediaPipe Image Classifier Customization in Python
DESCRIPTION: This snippet imports essential Python libraries for image classification model customization. It includes `google.colab.files` for Colab-specific file operations, `os` for operating system interactions, `tensorflow` for ML operations (asserting version 2.x), `mediapipe_model_maker.image_classifier` for the core model customization functionalities, and `matplotlib.pyplot` for plotting and visualization.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb#_snippet_2

LANGUAGE: Python
CODE:
```
from google.colab import files
import os
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import image_classifier

import matplotlib.pyplot as plt
```

----------------------------------------

TITLE: Downloading Stylized Face Image for Training
DESCRIPTION: This code downloads a sample stylized face image (`color_sketch.jpg`) from a Google Cloud Storage URL. This image serves as the target style for retraining the face stylizer model, demonstrating how to provide the single stylized face image required for the customization process.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/face_stylizer.ipynb#_snippet_3

LANGUAGE: Python
CODE:
```
style_image_path = 'color_sketch.jpg'
!wget -q -O {style_image_path} https://storage.googleapis.com/mediapipe-assets/face_stylizer_style_color_sketch.jpg
```

----------------------------------------

TITLE: Resizing and Displaying Test Images (Python)
DESCRIPTION: This Python code defines a `resize_and_show` function to scale images while maintaining aspect ratio, ensuring they fit within a `DESIRED_WIDTH` and `DESIRED_HEIGHT`. It then loads the previously downloaded test images using OpenCV (`cv2.imread`) and displays them using `cv2_imshow` (a Colab-specific patch for OpenCV display), providing a visual preview of the input data. It depends on `cv2`, `google.colab.patches`, and `math`.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_classification/python/image_classifier.ipynb#_snippet_6

LANGUAGE: python
CODE:
```
import cv2
from google.colab.patches import cv2_imshow
import math

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2_imshow(img)


# Preview the images.

images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
for name, image in images.items():
  print(name)
  resize_and_show(image)
```

----------------------------------------

TITLE: Previewing Downloaded Images with Resizing
DESCRIPTION: This snippet defines a `resize_and_show` function to scale images while maintaining aspect ratio and then displays the downloaded test image using OpenCV's `cv2_imshow`. It requires `cv2` and `google.colab.patches.cv2_imshow` for execution in a Colab environment.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_segmentation/python/image_segmentation.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
import cv2
from google.colab.patches import cv2_imshow
import math

# Height and width that will be used by the model
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# Performs resizing and showing the image
def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2_imshow(img)


# Preview the image(s)
images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
for name, image in images.items():
  print(name)
  resize_and_show(image)
```

----------------------------------------

TITLE: Resizing and Displaying Images with OpenCV in Colab (Python)
DESCRIPTION: This snippet defines a utility function `resize_and_show` to resize an image while maintaining its aspect ratio and then displays it using `cv2_imshow` (specific to Google Colab). It then loads all images specified in `IMAGE_FILENAMES` using OpenCV and previews them. Dependencies include `opencv-python` and `google.colab.patches`.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/gesture_recognizer/python/gesture_recognizer.ipynb#_snippet_6

LANGUAGE: Python
CODE:
```
import cv2

from google.colab.patches import cv2_imshow
import math

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2_imshow(img)


# Preview the images.
images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
for name, image in images.items():
  print(name)
  resize_and_show(image)
```

----------------------------------------

TITLE: Importing Transformers Pipeline (Python)
DESCRIPTION: This snippet imports the `pipeline` function from the `transformers` library, along with the `torch` library. The `pipeline` function provides a high-level API for various NLP tasks, simplifying the process of using pre-trained models for inference, such as text generation or question answering.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
import torch

from transformers import pipeline
```

----------------------------------------

TITLE: Resizing and Displaying Images with OpenCV in Python
DESCRIPTION: This Python code defines a `resize_and_show` function to resize an image while maintaining its aspect ratio and displays it using `cv2_imshow`. It then iterates through downloaded images, printing their names and displaying them, preparing them for segmentation.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/interactive_segmentation/python/interactive_segmenter.ipynb#_snippet_5

LANGUAGE: python
CODE:
```
import cv2
from google.colab.patches import cv2_imshow
import math

# Height and width that will be used by the model
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# Performs resizing and showing the image
def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2_imshow(img)


# Preview the image(s)
images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
for name, image in images.items():
  print(name)
  resize_and_show(image)
```

----------------------------------------

TITLE: Downloading Sample Image for Interactive Segmentation
DESCRIPTION: This Python snippet downloads a sample image, `cats_and_dogs.jpg`, from Google Cloud Storage using `urllib.request`. The image is used as input for demonstrating the interactive segmenter, and the code can be extended to download multiple images.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/interactive_segmentation/python/interactive_segmenter.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
import urllib
IMAGE_FILENAMES = ['cats_and_dogs.jpg']

for name in IMAGE_FILENAMES:
  url = f'https://storage.googleapis.com/mediapipe-assets/{name}'
  urllib.request.urlretrieve(url, name)
```

----------------------------------------

TITLE: Printing COCO Dataset Categories in Python
DESCRIPTION: This Python snippet reads the `labels.json` file from a COCO dataset path, loads its content, and iterates through the 'categories' array to print each category's ID and name. It helps verify the dataset's category definitions, expecting a 'background' class at index 0 and two non-background categories.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_6

LANGUAGE: Python
CODE:
```
with open(os.path.join(train_dataset_path, "labels.json"), "r") as f:
  labels_json = json.load(f)
for category_item in labels_json["categories"]:
  print(f"{category_item['id']}: {category_item['name']}")
```

----------------------------------------

TITLE: Downloading Test Image - Python
DESCRIPTION: This Python snippet downloads a test image (`business-person.png`) from Google Cloud Storage using `urllib.request.urlretrieve`. The image is used as input for the face stylization process, demonstrating how to prepare input data for the model.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_stylizer/python/face_stylizer.ipynb#_snippet_3

LANGUAGE: Python
CODE:
```
import urllib
IMAGE_FILENAMES = ['business-person.png']

for name in IMAGE_FILENAMES:
  url = f'https://storage.googleapis.com/mediapipe-assets/{name}'
  urllib.request.urlretrieve(url, name)
```

----------------------------------------

TITLE: Downloading Test Images (Python)
DESCRIPTION: This Python script downloads two sample images, 'burger.jpg' and 'cat.jpg', from a Google Cloud Storage URL. It uses the `urllib` module to retrieve the images and save them locally, which are then used for testing the image classification model.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_classification/python/image_classifier.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
import urllib

IMAGE_FILENAMES = ['burger.jpg', 'cat.jpg']

for name in IMAGE_FILENAMES:
  url = f'https://storage.googleapis.com/mediapipe-tasks/image_classifier/{name}'
  urllib.request.urlretrieve(url, name)
```

----------------------------------------

TITLE: Importing Libraries for Text Classification
DESCRIPTION: This snippet imports necessary Python libraries: `os` for operating system interactions, `tensorflow` for machine learning operations (asserting version 2.x), and `text_classifier` from `mediapipe_model_maker` for text classification functionalities.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/text_classifier.ipynb#_snippet_2

LANGUAGE: python
CODE:
```
import os
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import text_classifier
```

----------------------------------------

TITLE: Downloading Sample Audio File - Python
DESCRIPTION: This Python snippet uses the `urllib` module to download a sample `.wav` audio file named `speech_16000_hz_mono.wav` from a Google Cloud Storage URL. This audio file will be used as input for the audio classification task.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/audio_classifier/python/audio_classification.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
import urllib

audio_file_name = 'speech_16000_hz_mono.wav'
url = f'https://storage.googleapis.com/mediapipe-assets/{audio_file_name}'
urllib.request.urlretrieve(url, audio_file_name)
```

----------------------------------------

TITLE: Downloading the Generated MediaPipe Task Bundle
DESCRIPTION: This Python snippet facilitates downloading the `.task` file generated by the bundler. It uses `google.colab.files.download` to prompt the user to download the output bundle file, which is specified by the `output_filename` variable from the previous step.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/bundling/llm_bundling.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
#@title Download the Bundle { display-mode: "form" }
#@markdown Run this cell to download the generated `.task` file.
files.download(output_filename)
```

----------------------------------------

TITLE: Listing and Downloading Exported Model in Google Colab (Python)
DESCRIPTION: These commands are specific to a Google Colab environment. The first command lists the contents of the `exported_model` directory, and the second command downloads the `model.tflite` file to the local development environment. This facilitates access to the exported model for further use.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb#_snippet_11

LANGUAGE: python
CODE:
```
!ls exported_model
files.download('exported_model/model.tflite')
```

----------------------------------------

TITLE: Verifying Dataset Labels for Gesture Recognition
DESCRIPTION: This snippet verifies the downloaded dataset by listing and printing the detected labels. It iterates through the dataset directory to identify subdirectories, which represent the different gesture labels, ensuring the 'none' label is present.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/gesture_recognizer.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
print(dataset_path)
labels = []
for i in os.listdir(dataset_path):
  if os.path.isdir(os.path.join(dataset_path, i)):
    labels.append(i)
print(labels)
```

----------------------------------------

TITLE: Listing Exported Model Files (Shell)
DESCRIPTION: Executes a shell command to list the contents of the `exported_model` directory in a long format, showing file sizes in human-readable units. This command is used to verify the creation and compare the size of the newly exported quantized model (`model_int8.tflite`) against the original.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb#_snippet_17

LANGUAGE: Shell
CODE:
```
!ls -lh exported_model
```

----------------------------------------

TITLE: Instantiating AI Edge LiteRT Interpreter (Python)
DESCRIPTION: This snippet creates an instance of the `InterpreterWithCustomOps` from the `ai_edge_litert` library. It configures the interpreter to use custom GenAI operations, loads the quantized Gemma LiteRT model from the specified path, sets the number of inference threads, and enables experimental delegate features for optimal performance.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_17

LANGUAGE: Python
CODE:
```
interpreter = interpreter_lib.InterpreterWithCustomOps(
    custom_op_registerers=["pywrap_genai_ops.GenAIOpsRegisterer"],
    model_path="/content/gemma3_1b_finetune_q8_ekv1024.tflite",
    num_threads=2,
    experimental_default_delegate_latest_features=True)
```

----------------------------------------

TITLE: Importing AI Edge LiteRT Interpreter Dependencies (Python)
DESCRIPTION: This snippet imports essential Python libraries required for working with AI Edge LiteRT models. It includes `interpreter_lib` for the LiteRT interpreter, `AutoTokenizer` from `transformers` for tokenization, `numpy` for numerical operations, and `collections.abc.Sequence` and `sys` for general utility.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_15

LANGUAGE: Python
CODE:
```
from ai_edge_litert import interpreter as interpreter_lib
from transformers import AutoTokenizer
import numpy as np
from collections.abc import Sequence
import sys
```

----------------------------------------

TITLE: Uploading Custom Images in Google Colab (Python)
DESCRIPTION: This commented-out snippet provides an optional way to upload custom images when running in a Google Colab environment. It uses `google.colab.files.upload()` to prompt the user for file selection, then saves the uploaded content to local files and updates the `IMAGE_FILENAMES` list.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/gesture_recognizer/python/gesture_recognizer.ipynb#_snippet_5

LANGUAGE: Python
CODE:
```
# from google.colab import files
# uploaded = files.upload()

# for filename in uploaded:
#   content = uploaded[filename]
#   with open(filename, 'wb') as f:
#     f.write(content)
# IMAGE_FILENAMES = list(uploaded.keys())

# print('Uploaded files:', IMAGE_FILENAMES)
```

----------------------------------------

TITLE: Uploading Custom Image (Commented) - Python
DESCRIPTION: This commented-out Python snippet provides an alternative method for users to upload their own image files from their local machine to the Colab environment. It uses `google.colab.files.upload()` to handle the file upload and sets `IMAGE_FILE` to the name of the uploaded file.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_detector/python/face_detector.ipynb#_snippet_5

LANGUAGE: python
CODE:
```
# from google.colab import files
# uploaded = files.upload()

# for filename in uploaded:
#   content = uploaded[filename]
#   with open(filename, 'wb') as f:
#     f.write(content)

# if len(uploaded.keys()):
#   IMAGE_FILE = next(iter(uploaded))
#   print('Uploaded file:', IMAGE_FILE)
```

----------------------------------------

TITLE: Saving Converted Models to Google Drive in Python
DESCRIPTION: This Python snippet, designed for Google Colab, mounts Google Drive and copies the converted models from the local `/content/converted_models` directory to a specified folder within 'My Drive'. It includes shell commands for directory creation and file copying, providing persistent storage for the generated model files.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_20

LANGUAGE: python
CODE:
```
# @title Save to google drive { display-mode: "form"}
google_drive_directory = "converted_models" #@param {type:"string"}
from google.colab import drive
drive.mount('/content/drive')
print("Copying models ...")
!mkdir -p /content/drive/MyDrive/$google_drive_directory
!cp -r -f /content/converted_models/* /content/drive/MyDrive/$google_drive_directory
print("Done")
```

----------------------------------------

TITLE: Installing Dependencies for LiteRT
DESCRIPTION: This snippet downloads and extracts the `protoc` compiler, a prerequisite for some LiteRT tools, ensuring it's available in the system's local directory.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma3_1b_tflite.ipynb#_snippet_0

LANGUAGE: Python
CODE:
```
! wget -q https://github.com/protocolbuffers/protobuf/releases/download/v3.19.0/protoc-3.19.0-linux-x86_64.zip
! unzip -o protoc-3.19.0-linux-x86_64.zip -d /usr/local/
```

----------------------------------------

TITLE: Importing MediaPipe Model Maker Quantization Module (Python)
DESCRIPTION: This snippet shows the necessary import statement to access the quantization functionalities within the MediaPipe Model Maker library, which is a prerequisite for applying post-training quantization.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_16

LANGUAGE: Python
CODE:
```
from mediapipe_model_maker import quantization
```

----------------------------------------

TITLE: Printing Facial Transformation Matrix with MediaPipe in Python
DESCRIPTION: This snippet prints the facial transformation matrix obtained from the MediaPipe FaceLandmarker detection result. The `facial_transformation_matrixes` property contains the 3D transformation data for the detected face(s).
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/python/[MediaPipe_Python_Tasks]_Face_Landmarker.ipynb#_snippet_8

LANGUAGE: python
CODE:
```
print(detection_result.facial_transformation_matrixes)
```

----------------------------------------

TITLE: Attaching Observers and Displaying UI Widgets
DESCRIPTION: Attaches the `on_change_model` and `on_use_gpu` functions as observers to the `model` dropdown, ensuring dynamic updates. It then displays all interactive widgets (`model`, `backend`, `token_description`, `token`) in the UI, along with a helpful message about the Hugging Face token.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_6

LANGUAGE: Python
CODE:
```
model.observe(on_change_model, names='value')
model.observe(on_use_gpu, names='value')


display(model)
display(backend)

token_description = widgets.Output()
with token_description:
  print("Huggingface token needed for gated model (e.g. Gemma)")
  print("You can get it from https://huggingface.co/settings/tokens")

display(token_description)
display(token)
```

----------------------------------------

TITLE: Optional: Uploading Custom Images (Python)
DESCRIPTION: This commented-out Python code provides an optional method for users to upload their own images in a Google Colab environment. It uses `google.colab.files` to handle file uploads and then saves the content locally, updating the `IMAGE_FILENAMES` list for subsequent processing.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_classification/python/image_classifier.ipynb#_snippet_5

LANGUAGE: python
CODE:
```
# from google.colab import files
# uploaded = files.upload()

# for filename in uploaded:
#   content = uploaded[filename]
#   with open(filename, 'wb') as f:
#     f.write(content)
# IMAGE_FILENAMES = list(uploaded.keys())

# print('Uploaded files:', IMAGE_FILENAMES)
```

----------------------------------------

TITLE: Defining Input Text for Language Detection in Python
DESCRIPTION: This snippet defines a Python string variable 'INPUT_TEXT' containing a Chinese phrase. This variable serves as the input for the MediaPipe language detector, demonstrating how to prepare text for analysis. The #@param comment indicates it's a parameter in a notebook environment.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/language_detector/python/[MediaPipe_Python_Tasks]_Language_Detector.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
INPUT_TEXT = "\u5206\u4E45\u5FC5\u5408\u5408\u5FC5\u5206" #@param {type:"string"}
```

----------------------------------------

TITLE: Defining Input Text for Classification in Python
DESCRIPTION: This snippet defines a string variable `INPUT_TEXT` which holds the text that will be classified by the MediaPipe model. This serves as the input for the text classification inference process.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/text_classification/python/text_classifier.ipynb#_snippet_3

LANGUAGE: python
CODE:
```
INPUT_TEXT = "I'm looking forward to what will come next."
```

----------------------------------------

TITLE: Running Decode Operation for LLM in Python
DESCRIPTION: This method is responsible for the iterative decoding phase of text generation. It takes the starting position, initial token ID, the current KV cache, and a maximum number of decode steps to generate subsequent tokens.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma2_tflite.ipynb#_snippet_11

LANGUAGE: Python
CODE:
```
  def _run_decode(
      self,
      start_pos: int,
      start_token_id: int,
      kv_cache: dict[str, np.ndarray],
      max_decode_steps: int,
  ) -> str:
```

----------------------------------------

TITLE: Running Prefill Operation for LLM in Python
DESCRIPTION: This method executes the prefill operation, preparing the input tokens and their positions for the model. It initializes the KV cache and then runs the prefill runner, returning the updated KV cache for subsequent decode steps.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma2_tflite.ipynb#_snippet_9

LANGUAGE: Python
CODE:
```
  def _run_prefill(
      self, prefill_token_ids: Sequence[int],
  ) -> dict[str, np.ndarray]:
    """Runs prefill and returns the kv cache.

    Args:
      prefill_token_ids: The token ids of the prefill input.

    Returns:
      The updated kv cache.
    """
    if not self._prefill_runner:
      raise ValueError("Prefill runner is not initialized.")
    prefill_token_length = len(prefill_token_ids)
    if prefill_token_length == 0:
      return self._init_kv_cache()

    # Prepare the input to be [1, max_seq_len].
    input_token_ids = [0] * self._max_seq_len
    input_token_ids[:prefill_token_length] = prefill_token_ids
    input_token_ids = np.asarray(input_token_ids, dtype=np.int32)
    input_token_ids = np.expand_dims(input_token_ids, axis=0)

    # Prepare the input position to be [max_seq_len].
    input_pos = [0] * self._max_seq_len
    input_pos[:prefill_token_length] = range(prefill_token_length)
    input_pos = np.asarray(input_pos, dtype=np.int32)

    # Initialize kv cache.
    prefill_inputs = self._init_kv_cache()
    prefill_inputs.update({
        "tokens": input_token_ids,
        "input_pos": input_pos,
    })
    prefill_outputs = self._prefill_runner(**prefill_inputs)
    if "logits" in prefill_outputs:
      # Prefill outputs includes logits and kv cache. We only output kv cache.
      prefill_outputs.pop("logits")

    return prefill_outputs
```

----------------------------------------

TITLE: Running Decode Stage - LiteRT LLM Pipeline - Python
DESCRIPTION: This method performs the iterative decoding process, generating tokens one by one. It updates input positions, manages the KV cache, dynamically creates attention masks for each step, and accumulates the generated text until an end-of-sequence token is encountered or the maximum decode steps are reached.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_26

LANGUAGE: python
CODE:
```
  def _run_decode(
      self,
      start_pos: int,
      start_token_id: int,
      kv_cache: dict[str, np.ndarray],
      max_decode_steps: int,
  ) -> str:
    """Runs decode and outputs the token ids from greedy sampler.

    Args:
      start_pos: The position of the first token of the decode input.
      start_token_id: The token id of the first token of the decode input.
      kv_cache: The kv cache from the prefill.
      max_decode_steps: The max decode steps.

    Returns:
      The token ids from the greedy sampler.
    """
    next_pos = start_pos
    next_token = start_token_id
    decode_text = []
    decode_inputs = kv_cache

    for _ in range(max_decode_steps):
      decode_inputs.update({
          "tokens": np.array([[next_token]], dtype=np.int32),
          "input_pos": np.array([next_pos], dtype=np.int32),
      })
      if "mask" in self._decode_runner.get_input_details().keys():
        # For decode, mask has shape [batch=1, 1, 1, kv_cache_size].
        # We want mask[0, 0, 0, j] = 0 for j<=next_pos and -inf otherwise.
        decode_inputs["mask"] = _get_mask(
            shape=self._decode_runner.get_input_details()["mask"]["shape"],
            k=next_pos + 1,
        )
      decode_outputs = self._decode_runner(**decode_inputs)
      # Output logits has shape (batch=1, 1, vocab_size). We only take the first
      # element.
      logits = decode_outputs.pop("logits")[0][0]
      next_token = self._greedy_sampler(logits)
      if next_token == self._tokenizer.eos_token_id:
        break
      decode_text.append(self._tokenizer.decode(next_token, skip_special_tokens=True))
      if len(decode_text[-1]) == 0:
        # Break out the loop if we hit the special token.
        break

      print(decode_text[-1], end='', flush=True)
      # Decode outputs includes logits and kv cache. We already poped out
      # logits, so the rest is kv cache. We pass the updated kv cache as input
      # to the next decode step.
      decode_inputs = decode_outputs
      next_pos += 1

    print() # print a new line at the end.
    return ''.join(decode_text)
```

----------------------------------------

TITLE: Initializing and Displaying UI Widgets in Python
DESCRIPTION: This snippet initializes a `Button` widget with 'Start Conversion' text and attaches the `on_button_clicked` function as its event handler. It also creates an `Output` widget for displaying messages and both widgets are displayed, providing an interactive interface for model operations and status updates.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_19

LANGUAGE: python
CODE:
```
button = widgets.Button(description="Start Conversion")

button.on_click(on_button_clicked)
display(button)

out = widgets.Output(layout={'border': '1px solid black'})
display(out)

print("\nNotice: Converted models are saved under ./converted_models")
```

----------------------------------------

TITLE: Listing and Downloading Exported Model in Google Colab (Python)
DESCRIPTION: This snippet provides commands for use within Google Colab to list the contents of the 'exported_model' directory and download the 'face_stylizer.task' file. The '.task' file is a crucial bundle containing three TFLite models required to run the MediaPipe face stylizer task library in a development environment.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/face_stylizer.ipynb#_snippet_10

LANGUAGE: python
CODE:
```
!ls exported_model
files.download('exported_model/face_stylizer.task')
```

----------------------------------------

TITLE: Initializing KV Cache in Python
DESCRIPTION: This method initializes the Key-Value (KV) cache, which is crucial for efficient sequence generation in LLMs. It creates a dictionary of zero-filled NumPy arrays for each 'kv_cache' input key identified by the prefill runner.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma2_tflite.ipynb#_snippet_7

LANGUAGE: Python
CODE:
```
  def _init_kv_cache(self) -> dict[str, np.ndarray]:
    if self._prefill_runner is None:
      raise ValueError("Prefill runner is not initialized.")
    kv_cache = {}
    for input_key in self._prefill_runner.get_input_details().keys():
      if "kv_cache" in input_key:
        kv_cache[input_key] = np.zeros(
            self._prefill_runner.get_input_details()[input_key]["shape"],
            dtype=np.float32,
        )
        kv_cache[input_key] = np.zeros(
            self._prefill_runner.get_input_details()[input_key]["shape"],
            dtype=np.float32,
        )
    return kv_cache
```

----------------------------------------

TITLE: Initializing KV Cache for LiteRTLlmPipeline (Python)
DESCRIPTION: This method initializes the Key-Value (KV) cache, which is crucial for efficient LLM inference. It creates a dictionary of NumPy arrays, where each array corresponds to a KV cache input of the prefill runner and is initialized with zeros based on the expected shape.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_21

LANGUAGE: python
CODE:
```
  def _init_kv_cache(self) -> dict[str, np.ndarray]:
    if self._prefill_runner is None:
      raise ValueError("Prefill runner is not initialized.")
    kv_cache = {}
    for input_key in self._prefill_runner.get_input_details().keys():
      if "kv_cache" in input_key:
        kv_cache[input_key] = np.zeros(
            self._prefill_runner.get_input_details()[input_key]["shape"],
            dtype=np.float32,
        )
        kv_cache[input_key] = np.zeros(
            self._prefill_runner.get_input_details()[input_key]["shape"],
            dtype=np.float32,
        )
    return kv_cache
```

----------------------------------------

TITLE: Initializing Prefill Runner for LLM in Python
DESCRIPTION: This method initializes the prefill runner and determines the maximum sequence lengths for both input and KV cache based on the provided number of input tokens. It selects the most suitable prefill runner from available signatures.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma2_tflite.ipynb#_snippet_6

LANGUAGE: Python
CODE:
```
  def _init_prefill_runner(self, num_input_tokens: int):
    """Initializes all the variables related to the prefill runner.

    This method initializes the following variables:
      - self._prefill_runner: The prefill runner based on the input size.
      - self._max_seq_len: The maximum sequence length supported by the model.
      - self._max_kv_cache_seq_len: The maximum sequence length supported by the
        KV cache.

    Args:
      num_input_tokens: The number of input tokens.
    """
    if not self._interpreter:
      raise ValueError("Interpreter is not initialized.")

    # Prefill runner related variables will be initialized in `predict_text` and
    # `compute_log_likelihood`.
    self._prefill_runner = self._get_prefill_runner(num_input_tokens)
    # input_token_shape has shape (batch, max_seq_len)
    input_token_shape = self._prefill_runner.get_input_details()["tokens"][
        "shape"
    ]
    if len(input_token_shape) == 1:
      self._max_seq_len = input_token_shape[0]
    else:
      self._max_seq_len = input_token_shape[1]

    # kv cache input has shape [batch=1, seq_len, num_heads, dim].
    kv_cache_shape = self._prefill_runner.get_input_details()["kv_cache_k_0"][
        "shape"
    ]
    self._max_kv_cache_seq_len = kv_cache_shape[1]
```

----------------------------------------

TITLE: Initializing Prefill Runner in LiteRTLlmPipeline (Python)
DESCRIPTION: This method initializes the prefill runner and related variables, such as `_max_seq_len` and `_max_kv_cache_seq_len`, based on the number of input tokens. It dynamically selects the appropriate prefill runner from the interpreter's signatures and extracts sequence length information from its input details.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/Gemma3_1b_fine_tune.ipynb#_snippet_20

LANGUAGE: python
CODE:
```
  def _init_prefill_runner(self, num_input_tokens: int):
    """Initializes all the variables related to the prefill runner.

    This method initializes the following variables:
      - self._prefill_runner: The prefill runner based on the input size.
      - self._max_seq_len: The maximum sequence length supported by the model.

    Args:
      num_input_tokens: The number of input tokens.
    """
    if not self._interpreter:
      raise ValueError("Interpreter is not initialized.")

    # Prefill runner related variables will be initialized in `predict_text` and
    # `compute_log_likelihood`.
    self._prefill_runner = self._get_prefill_runner(num_input_tokens)
    # input_token_shape has shape (batch, max_seq_len)
    input_token_shape = self._prefill_runner.get_input_details()["tokens"][
        "shape"
    ]
    if len(input_token_shape) == 1:
      self._max_seq_len = input_token_shape[0]
    else:
      self._max_seq_len = input_token_shape[1]

    # kv cache input has shape [batch=1, num_kv_heads, cache_size, head_dim].
    kv_cache_shape = self._prefill_runner.get_input_details()["kv_cache_k_0"][
        "shape"
    ]
    self._max_kv_cache_seq_len = kv_cache_shape[2]
```

----------------------------------------

TITLE: Selecting Best Prefill Runner in Python
DESCRIPTION: This method dynamically selects the most suitable prefill runner from the interpreter's available signatures. It prioritizes runners that can accommodate the `num_input_tokens` with the smallest possible sequence length, optimizing for efficiency.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/codelabs/litert_inference/gemma2_tflite.ipynb#_snippet_8

LANGUAGE: Python
CODE:
```
  def _get_prefill_runner(self, num_input_tokens: int) :
    """Gets the prefill runner with the best suitable input size.

    Args:
      num_input_tokens: The number of input tokens.

    Returns:
      The prefill runner with the smallest input size.
    """
    best_signature = None
    delta = sys.maxsize
    max_prefill_len = -1
    for key in self._interpreter.get_signature_list().keys():
      if "prefill" not in key:
        continue
      input_pos = self._interpreter.get_signature_runner(key).get_input_details()[
          "input_pos"
      ]
      # input_pos["shape"] has shape (max_seq_len, )
      seq_size = input_pos["shape"][0]
      max_prefill_len = max(max_prefill_len, seq_size)
      if num_input_tokens <= seq_size and seq_size - num_input_tokens < delta:
        delta = seq_size - num_input_tokens
        best_signature = key
    if best_signature is None:
      raise ValueError(
          "The largest prefill length supported is %d, but we have %d number of input tokens"
          %(max_prefill_len, num_input_tokens)
      )
    return self._interpreter.get_signature_runner(best_signature)
```

----------------------------------------

TITLE: Running Local HTTP Server with Python 2
DESCRIPTION: This command starts a simple HTTP server using Python 2's `SimpleHTTPServer` module. It serves files from the current directory on port 8000, providing local access to the MediaPipe LLM Inference web demo. This option is intended for environments with older Python versions.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/js/README.md#_snippet_1

LANGUAGE: Python
CODE:
```
python -m SimpleHTTPServer 8000
```

----------------------------------------

TITLE: Defining Interactive Slider Parameters for ROI
DESCRIPTION: This Python snippet defines `x` and `y` variables, marked as interactive sliders for a Colab environment. These parameters represent normalized coordinates (0 to 1) and are intended to specify the Region of Interest (ROI) for the interactive segmenter.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/interactive_segmentation/python/interactive_segmenter.ipynb#_snippet_6

LANGUAGE: python
CODE:
```
x = 0.68 #@param {type:"slider", min:0, max:1, step:0.01}
y = 0.68 #@param {type:"slider", min:0, max:1, step:0.01}
```

----------------------------------------

TITLE: Playing Back Downloaded Audio in IPython - Python
DESCRIPTION: This snippet uses `IPython.display.Audio` to create an audio playback widget within an IPython or Jupyter environment. It allows users to verify the downloaded audio file (`speech_16000_hz_mono.wav`) by playing it back directly in the notebook, with `autoplay` set to `False`.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/audio_classifier/python/audio_classification.ipynb#_snippet_4

LANGUAGE: python
CODE:
```
from IPython.display import Audio, display

file_name = 'speech_16000_hz_mono.wav'
display(Audio(file_name, autoplay=False))
```

----------------------------------------

TITLE: Uploading Custom Image (Optional) - Python
DESCRIPTION: This commented-out snippet provides an optional way for users to upload their own image files from their local machine. Once uploaded, the `IMAGE_FILE` variable would store the name of the first uploaded file, allowing it to be used as input for the hand landmark detection.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#_snippet_5

LANGUAGE: Python
CODE:
```
# from google.colab import files
# uploaded = files.upload()

# for filename in uploaded:
#   content = uploaded[filename]
#   with open(filename, 'wb') as f:
#     f.write(content)

# if len(uploaded.keys()):
#   IMAGE_FILE = next(iter(uploaded))
#   print('Uploaded file:', IMAGE_FILE)
```

----------------------------------------

TITLE: License Information for MediaPipe Model Maker Python
DESCRIPTION: This snippet provides the Apache License, Version 2.0, under which the MediaPipe Authors distribute their code. It specifies the terms and conditions for use, reproduction, and distribution of the software, ensuring compliance with open-source guidelines.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/image_classifier.ipynb#_snippet_0

LANGUAGE: Python
CODE:
```
#@title License information
# Copyright 2023 The MediaPipe Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: Apache 2.0 License Header - Python
DESCRIPTION: This snippet contains the Apache License, Version 2.0 header, indicating the terms under which the code is distributed and used. It specifies permissions, limitations, and disclaimers for the software.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_stylizer/python/face_stylizer.ipynb#_snippet_0

LANGUAGE: Python
CODE:
```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: Apache 2.0 License Header - Python
DESCRIPTION: This snippet contains the standard Apache License, Version 2.0 header, outlining the terms under which the code is provided and may be used. It specifies the conditions for distribution, modification, and limitations of liability.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#_snippet_0

LANGUAGE: Python
CODE:
```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: Apache License 2.0 for MediaPipe Samples - Python
DESCRIPTION: This snippet provides the Apache License, Version 2.0, under which the MediaPipe samples are distributed. It outlines the terms and conditions for use, reproduction, and distribution of the software, emphasizing the disclaimer of warranties and limitations of liability.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/audio_classifier/python/audio_classification.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: Apache License Header for MediaPipe Samples
DESCRIPTION: This snippet contains the standard Apache License 2.0 header, outlining the terms under which the MediaPipe samples are distributed and used, ensuring compliance with open-source licensing requirements.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/interactive_segmentation/python/interactive_segmenter.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: Apache 2.0 License Header
DESCRIPTION: This snippet contains the Apache License, Version 2.0 header, indicating the licensing terms under which the MediaPipe code is distributed. It specifies the conditions for use, reproduction, and distribution.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/text_classification/python/text_classifier.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: Apache License Header - Python
DESCRIPTION: This snippet contains the standard Apache License 2.0 header, indicating the licensing terms under which the code is distributed. It specifies the conditions for use, reproduction, and distribution.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_detector/python/face_detector.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: License Header for MediaPipe Samples - Python
DESCRIPTION: This code block contains the standard Apache License, Version 2.0 header. It specifies the terms and conditions under which the MediaPipe code is licensed, including permissions for use, reproduction, and distribution, along with disclaimers of warranty.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/object_detection/python/object_detector.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: License Information for MediaPipe LLM Conversion - Python
DESCRIPTION: This snippet displays the Apache License, Version 2.0, under which the MediaPipe LLM conversion code is distributed. It outlines the terms and conditions for use, reproduction, and distribution of the software.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/conversion/llm_conversion.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
#@title License information { display-mode: "form" }
#@markdown Copyright 2024 The MediaPipe Authors.
#@markdown Licensed under the Apache License, Version 2.0 (the "License");
#@markdown
#@markdown you may not use this file except in compliance with the License.
#@markdown You may obtain a copy of the License at
#@markdown
#@markdown https://www.apache.org/licenses/LICENSE-2.0
#@markdown
#@markdown Unless required by applicable law or agreed to in writing, software
#@markdown distributed under the License is distributed on an "AS IS" BASIS,
#@markdown WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#@markdown See the License for the specific language governing permissions and
#@markdown limitations under the License.
```

----------------------------------------

TITLE: License Information for MediaPipe Samples
DESCRIPTION: This snippet displays the Apache License, Version 2.0, under which the MediaPipe samples are distributed. It outlines the terms and conditions for use, reproduction, and distribution of the software.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/llm_inference/bundling/llm_bundling.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
#@title License information { display-mode: "form" }
#@markdown Copyright 2024 The MediaPipe Authors.
#@markdown Licensed under the Apache License, Version 2.0 (the "License");
#@markdown
#@markdown you may not use this file except in compliance with the License.
#@markdown You may obtain a copy of the License at
#@markdown
#@markdown https://www.apache.org/licenses/LICENSE-2.0
#@markdown
#@markdown Unless required by applicable law or agreed to in writing, software
#@markdown distributed under the License is distributed on an "AS IS" BASIS,
#@markdown WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#@markdown See the License for the specific language governing permissions and
#@markdown limitations under the License.
```

----------------------------------------

TITLE: Apache 2.0 License Header (Python)
DESCRIPTION: This snippet provides the standard Apache License, Version 2.0 header, outlining the terms under which the code is distributed and used. It specifies the permissions, limitations, and conditions for using the software.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_classification/python/image_classifier.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: License Header for MediaPipe Samples - Python
DESCRIPTION: This snippet contains the Apache License, Version 2.0 header, indicating the licensing terms under which the MediaPipe samples are distributed. It specifies the conditions for use, reproduction, and distribution of the software.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/python/[MediaPipe_Python_Tasks]_Face_Landmarker.ipynb#_snippet_0

LANGUAGE: Python
CODE:
```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: License Header for MediaPipe Samples - Python
DESCRIPTION: This snippet contains the Apache License, Version 2.0, which governs the use and distribution of the MediaPipe samples. It specifies the terms under which the code can be used, modified, and shared, ensuring compliance with open-source principles.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/text_embedder/python/text_embedder.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: License Information for MediaPipe Object Detector (Python)
DESCRIPTION: This snippet provides the Apache 2.0 license information for the MediaPipe Object Detector, outlining the terms under which the software can be used, modified, and distributed. It ensures compliance with open-source guidelines for the project.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/object_detector.ipynb#_snippet_0

LANGUAGE: Python
CODE:
```
#@title License information
# Copyright 2023 The MediaPipe Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: Apache 2.0 License Header
DESCRIPTION: This snippet contains the standard Apache License, Version 2.0 boilerplate, outlining the terms under which the MediaPipe code samples are distributed and used.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_segmentation/python/image_segmentation.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: License Information for MediaPipe Model Maker
DESCRIPTION: This snippet provides the Apache License, Version 2.0, for the MediaPipe Model Maker project, outlining the terms under which the software is distributed and used.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/text_classifier.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
#@title License information
# Copyright 2023 The MediaPipe Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: License Information for MediaPipe Model Maker
DESCRIPTION: This snippet provides the Apache License, Version 2.0, under which the MediaPipe Authors distribute their work. It outlines the terms and conditions for using, reproducing, and distributing the software components.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/gesture_recognizer.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
#@title License information
# Copyright 2023 The MediaPipe Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: Apache License Header for MediaPipe Samples
DESCRIPTION: This snippet contains the standard Apache License, Version 2.0 header, outlining the terms under which the MediaPipe code is licensed. It specifies permissions, limitations, and disclaimers for use, reproduction, and distribution.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/language_detector/python/[MediaPipe_Python_Tasks]_Language_Detector.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: Apache License Header
DESCRIPTION: This snippet contains the standard Apache License 2.0 header, indicating the licensing terms for the code. It specifies that the code is licensed under Apache License, Version 2.0, and outlines the conditions for use, reproduction, and distribution.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/image_embedder/python/image_embedder.ipynb#_snippet_0

LANGUAGE: Python
CODE:
```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: Apache 2.0 License Header - Python
DESCRIPTION: This snippet contains the standard Apache License, Version 2.0 header, indicating the licensing terms under which the MediaPipe code is distributed. It specifies the conditions for use, reproduction, and distribution of the software.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/gesture_recognizer/python/gesture_recognizer.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: Apache License Header for MediaPipe
DESCRIPTION: This snippet provides the standard Apache License, Version 2.0 header, outlining the terms under which the code is licensed. It specifies the permissions and limitations for use, reproduction, and distribution of the software.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/tutorials/object_detection/Object_Detection_for_3_dogs.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: License Information for MediaPipe Face Stylizer
DESCRIPTION: This snippet provides the Apache License, Version 2.0, for the MediaPipe project, outlining the terms under which the code can be used, modified, and distributed. It's a standard boilerplate for open-source projects.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/customization/face_stylizer.ipynb#_snippet_0

LANGUAGE: Python
CODE:
```
#@title License information
# Copyright 2023 The MediaPipe Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

----------------------------------------

TITLE: Apache License Header for MediaPipe
DESCRIPTION: This code block contains the standard Apache 2.0 license header, which specifies the terms and conditions under which the MediaPipe code is distributed. It outlines the permissions, limitations, and disclaimers associated with using and modifying the software.
SOURCE: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/python/[MediaPipe_Python_Tasks]_Pose_Landmarker.ipynb#_snippet_0

LANGUAGE: python
CODE:
```
#@title Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```