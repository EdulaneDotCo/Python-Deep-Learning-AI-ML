# VISION WHISPER
### (An AI-Powered Image and Speech Recognition System)

</br>


# Project Overview
1. This project involves developing an image and speech recognition system using Python.

2. The system uses OpenCV and TensorFlow to identify and categorize objects in images (e.g., humans, cars, fruits, and others).
3. It integrates with ChatGPT to provide detailed information about the identified objects and allows users to download this information as a PDF. Users can also edit the prompt before generating the PDF.

#  Import The Neccessary Libraries

<table>
    <tr>
        <th>Library</th>
        <th>Description</th>
        <th>Import Statement</th>
    </tr>
    <tr>
        <td>OpenCV</td>
        <td>Used for image processing and recognition</td>
        <td><code>import cv2</code></td>
    </tr>
    <tr>
        <td>TensorFlow</td>
        <td>Used for building and deploying machine learning models</td>
        <td><code>import tensorflow as tf</code></td>
    </tr>
    <tr>
        <td>ChatGPT API</td>
        <td>Used for getting detailed information about the objects</td>
        <td><code>import openai</code></td>
    </tr>
    <tr>
        <td>SpeechRecognition</td>
        <td>Used for converting speech input to text</td>
        <td><code>import speech_recognition as sr</code></td>
    </tr>
    <tr>
        <td>PyPDF2</td>
        <td>Used for creating and manipulating PDF files</td>
        <td><code>from PyPDF2 import PdfFileWriter, PdfFileReader</code></td>
    </tr>
</table>
 </code></td>
    </tr>
</table>
</code></td>
    </tr>
</table>

# Functional Requirement

 # 1. Image Recognition
 1. **Capture Image** : Caputure an image using a webcam or upload an image file in the format  .Png ,  .jpg
 2.  **Preproceesing** : Apply preprocessing techniques such as resizing,normalization,and data augmentation
   <table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Description</th>
      <th>Libraries Used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Normalize</strong></td>
      <td>Adjusts the range of pixel intensity values to a standard scale (e.g., 0 to 1 or -1 to 1) for consistent processing and to improve model convergence.</td>
      <td>OpenCV, TensorFlow</td>
    </tr>
    <tr>
      <td><strong>Resize</strong></td>
      <td>Resizes images to a specified dimension (e.g., to fit model input requirements or for consistency in processing).</td>
      <td>OpenCV, TensorFlow</td>
    </tr>
    <tr>
      <td><strong>Data Augmentation</strong></td>
      <td>Generates new training samples by applying transformations like rotations, translations, flips, zooms, etc., to increase model robustness and performance.</td>
      <td>OpenCVenst


  </tbody>
</table>


   </tbody>
</table>


 
3. **Object Detection** : To Train  model to detect and categories.
4. **Object Classification** : Classify detected objects into  categories like human,car,fruits,and others





  
# 2.  Speech Recongnition

1. **Capture Audio** : Capture an image using a webcam or upload an image file.
<table>
  <thead>
    <tr>
      <th>Function</th>
      <th>Description</th>
      <th>Libraries Used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>capture_and_recognize_speech()</td>
      <td>Captures audio input from the default microphone and performs speech recognition using Google's Web Speech API.</td>
      <td>speech_recognition</td>
    </tr>
    <tr>
      <td>Requirements</td>
      <td>Requires the Speech Recognition library for speech recognition and PyAudio for accessing the microphone.</td>
      <td>speech_recognition, pyaudio</td>
    </tr>
  </tbody>
</table>
 </tbody>
</table>

2. **Preprocessing** : Convert audio input to text using speech recognition techniques.
<table>
  <thead>
    <tr>
      <th>Technique</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Audio Normalization</td>
      <td>Adjusts the amplitude of audio signals to a standard level, ensuring consistent volume levels.</td>
    </tr>
    <tr>
      <td>Noise Reduction</td>
      <td>Filters out background noise from audio signals, improving the clarity of speech recognition.</td>
    </tr>
    <tr>
      <td>Feature Extraction</td>
      <td>Converts raw audio signals into representative features like Mel-Frequency Cepstral Coefficients (MFCCs) or spectrograms, suitable for model training.</td>
    </tr>
  </tbody>
</table>
  </tbody>
</table>

3. **Command Processing** : Process commands related to image capture,object identification,and pdf generation.
# 3. Integration with ChatGpt

1. **Query Generation** : Generat queries based on identified objects.
 <table>
  <thead>
    <tr>
      <th>Function Name</th>
      <th>Description</th>
      <th>Args</th>
      <th>Returns</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>generate_query()</td>
      <td>Generates a query string based on a list of identified objects.</td>
      <td>objects (list of str)</td>
      <td>str</td>
    </tr>
  </tbody>
</table>

2. **Chatgpt Response** : Get detailed information about the objects from ChatGpt.
<table>
  <thead>
    <tr>
      <th>Function Name</th>
      <th>Description</th>
      <th>Args</th>
      <th>Returns</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>get_chatgpt_response()</td>
      <td>Gets detailed information about the objects from ChatGPT.</td>
      <td>query (str)</td>
      <td>str</td>
    </tr>
  </tbody>
</table>

3. **Promt Editing** : Allow users to edit the prompt's before downloading.
<table>
  <thead>
    <tr>
      <th>Function/Method</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>edit_prompt(prompt)</td>
      <td>Allows users to edit the generated prompt before downloading or further processing.</td>
    </tr>
  </tbody>
</table>

# 4. Pdf Generation


1. **Content Compilation** : Compile the detailed information into a structured format.
  <table>
  <thead>
    <tr>
      <th>Method/Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>compile_information(info)</td>
      <td>Compiles detailed information into a structured format for further processing or display.</td>
    </tr>
  </tbody>
</table>

2. **PDF Creation** : Create a PDF file with the compiled information
 <table>
  <thead>
    <tr>
      <th>Method/Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>create_pdf(content)</td>
      <td>Creates a PDF file with the compiled information.</td>
    </tr>
  </tbody>
</table>

3. **Download Option** : Provide an option for users to download the generated Pdf.
  <table>
  <thead>
    <tr>
      <th>Function/Method</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>download_pdf(file_path)</td>
      <td>Provides an option for users to download the generated PDF.</td>
    </tr>
  </tbody>
</table>

# 5. Technical Requirements

## Hardware Requirements
1. **Camera** : For capturing images.
2. **Microphone** : For capturing audio input.
3. **Computer/Server** : To run the application and process the data.


# 6. Project Phases


## Phase 1 : Project Initialization
1. Define project scope and objectives.
2. Set up develop environment
  <table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Set up development environment</td>
      <td>Prepare the system and install required software tools and libraries for project development.</td>
    </tr>
  </tbody>
</table>

3. Install necessary libraries and tools
## Phase 2 : Image Recognition Module
1. implement image capture functionality
  <table>
  <thead>
    <tr>
      <th>Function/Method</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>capture_image()</td>
      <td>Captures an image from a webcam or loads an image file (supports .png, .jpg).</td>
    </tr>
    <tr>
      <td>upload_image()</td>
      <td>Allows the user to upload an image file (.png, .jpg) for processing.</td>
    </tr>
  </tbody>
</table>

2. Apply preprocessing techniques.
3. Develop object detection and classification models.
  <table>
  <thead>
    <tr>
      <th>Techniques and Approaches</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Deep Learning Models</td>
      <td>Utilize pre-trained convolutional neural networks (CNNs) like VGG, ResNet, or MobileNet as feature extractors, followed by additional layers for object detection and classification.</td>
    </tr>
  </tbody
</table>
 
4. Integrate the models with the application
  <table>
  <thead>
    <tr>
      <th>Function/Tool</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>TensorFlow</td>
      <td>Framework used to build and train machine learning models, including object detection and classification models.</td>
    </tr>
    <tr>
      <td>OpenCV</td>
      <td>Library used for image processing tasks, such as capturing images, preprocessing, and integrating with deep learning models.</td>
    </tr>
  </tbody>
</table>

# Phase 3 : Speech Recognition Module
1. Implement audio capture functionality
2. Convert speech to text using speech recognition
3. Develop command processing logic
  <table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Function/Method</th>
      <th>Tools</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Implement audio capture functionality</td>
      <td>capture_audio()</td>
      <td>SpeechRecognition, PyAudio</td>
      <td>Captures audio input from the default microphone.</td>
    </tr>
    <tr>
      <td>Convert speech to text using speech recognition</td>
      <td>convert_speech_to_text(audio)</td>
      <td>SpeechRecognition</td>
      <td>Converts captured audio to text using Google's Web Speech API.</td>
    </tr>
    <tr>
      <td>Develop command processing logic</td>
      <td>process_command(command)</td>
      <td>Custom Logic</td>
      <td>Processes user commands for image capture, object identification, PDF generation, etc.</td>
    </tr>
  </tbody>
</table>

# Phase 4 : Integration with ChatGpt
1. Set up the ChatGpt API.
2. Implement query generation based on identified objects.
3. Integrate ChatGpt response into the application.
4. Implement promt editing functionality.
<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Function/Method</th>
      <th>Tools</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Set up the ChatGPT API</td>
      <td>setup_chatgpt_api()</td>
      <td>OpenAI API</td>
      <td>Initializes and configures the ChatGPT API for use in the application.</td>
    </tr>
    <tr>
      <td>Implement query generation based on identified objects</td>
      <td>generate_query(objects)</td>
      <td>Custom Logic</td>
      <td>Generates a query string based on a list of identified objects.</td>
    </tr>
    <tr>
      <td>Integrate ChatGPT response into the application</td>
      <td>get_chatgpt_response(query)</td>
      <td>OpenAI API</td>
      <td>Gets detailed information about the objects from ChatGPT using the generated query.</td>
    </tr>
    <tr>
      <td>Implement prompt editing functionality</td>
      <td>edit_prompt(prompt)</td>
      <td>Tkinter</td>
      <td>Allows users to edit the prompt before sending it to ChatGPT.</td>
    </tr>
  </tbody>
</table>

# Phase 5: PDF Generation
1. Develop functionality to compile information into a structured format.
2. Implement Pdf creation and downloads options.
3. Ensures the Pdf contains all necessary information if the user edit the promt allow the option.

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Function/Method</th>
      <th>Tools</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Develop functionality to compile information into a structured format</td>
      <td>compile_information(info)</td>
      <td>Custom Logic</td>
      <td>Compiles detailed information into a structured format for inclusion in the PDF.</td>
    </tr>
    <tr>
      <td>Implement PDF creation and download options</td>
      <td>create_pdf(content)</td>
      <td>PyPDF2, FPDF</td>
      <td>Creates a PDF file with the compiled information.</td>
    </tr>
    <tr>
      <td>Implement PDF download option</td>
      <td>download_pdf(file_path)</td>
      <td>Custom Logic</td>
      <td>Provides an option for users to download the generated PDF.</td>
    </tr>
    <tr>
      <td>Ensure the PDF contains all necessary information if the user edits the prompt</td>
      <td>edit_prompt(prompt)</td>
      <td>Tkinter</td>
      <td>Allows users to edit the prompt before generating the PDF, ensuring all necessary information is included.</td>
    </tr>
  </tbody>
</table>

# Phase 6: User Interface Development
1. Develop a simple GUI using Tkinter.
2. Ensure the interface is user-friendly and intuitive.
  <table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Function/Method</th>
      <th>Tools</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Develop a simple GUI using Tkinter</td>
      <td>create_gui()</td>
      <td>Tkinter</td>
      <td>Creates a simple graphical user interface for user interactions.</td>
    </tr>
    <tr>
      <td>Ensure the interface is user-friendly and intuitive</td>
      <td>setup_user_friendly_interface()</td>
      <td>Tkinter</td>
      <td>Sets up the interface layout, buttons, and labels to be user-friendly and intuitive.</td>
    </tr>
  </tbody>
</table>

# Phase 7: Testing and Debugging
1. Test each module individually and as a whole.
2. Debug any issues and ensure the application works as intended.
3. Conduct user acceptance testing to gather feedback.<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Function/Method</th>
      <th>Tools</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Test each module individually and as a whole</td>
      <td>test_modules()</td>
      <td>unittest, pytest</td>
      <td>Tests each module separately and then tests the integrated system to ensure they work together correctly.</td>
    </tr>
    <tr>
      <td>Debug any issues and ensure the application works as intended</td>
      <td>debug_application()</td>
      <td>Logging, Debugging Tools</td>
      <td>Identifies and fixes any issues or bugs in the application to ensure it functions as intended.</td>
    </tr>
    <tr>
      <td>Conduct user acceptance testing to gather feedback</td>
      <td>user_acceptance_testing()</td>
      <td>Surveys, User Feedback Tools</td>
      <td>Conducts testing with actual users to gather feedback and ensure the application meets user needs.</td>
    </tr>
  </tbody>
</table>


  
# Phase 8: Deployment
1. Deploy the application to a suitable environment.
2. Ensure all dependencies are properly configured.
3. Provide documentation and user guides.
# 7. User Workflow
1. Image Capture: User captures or uploads an image.
2. Object Detection: The system detects and categorizes objects in the image.
3. Speech Command: User gives a speech command for detailed information.
4. ChatGPT Integration: The system generates a query and retrieves detailed information from ChatGPT.
5. Prompt Editing: User edits the prompt if necessary.
6. PDF Generation: The system compiles the information and generates a PDF.
7. PDF Download: User downloads the PDF.

# 8. Documentation and User Guides

1. **Installation Guide** : Steps to set up the development environment and install necessary libraries

2. **User Manual** : Instructions on how to use the application, including image capture, speech commands, and PDF generation.

3. **Developer Guide** : Detailed documentation of the codebase, including explanations of key functions and modules.

# 9. Best Practices for the Project
1. Modular Development: Develop each module (image recognition, speech recognition, ChatGPT integration, PDF generation) independently to ensure modularity and ease of testing.
2. Robust Error Handling: Implement comprehensive error handling to manage issues such as failed image uploads, recognition errors, and API call failures.
3. User-Friendly Interface: Ensure the GUI is intuitive and provides clear instructions for each step of the process.
4. Extensive Testing: Conduct thorough testing of each module and the integrated system to identify and fix any bugs.
5. Scalability Considerations: Design the system to handle a variety of image and audio inputs, and ensure it can be easily extended with new features in the future.
