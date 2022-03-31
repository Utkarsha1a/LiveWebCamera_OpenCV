# Video Streaming in Web Browsers with OpenCV & Flask

![Python 3.6](https://img.shields.io/badge/Python-3.6-brightgreen.svg) ![OpenCV](https://img.shields.io/badge/ComputerVision-opencv-blue.svg) ![Flask](https://img.shields.io/badge/Web_Framework-Flask-yellow.svg)

# Introduction

Image filters are very common in social media these days.Everyone using Social Media must have tried out filters at some point or other..Be it on Instagram, Snapchat, Facebook or even on Picsart. With just a few clicks and adjustments you can modify your images.People use filters to give the desired effects they want, to their photos

#### What is an Image Filter?

An Image Filter is a method or process through which colours, shade, hue, saturation, texture and other characteristics of an image are modified. Filters are used to visually modify images for commercial, artistic or aesthetic needs.

In this project, I have create a camera app using flask framework wherein we can click pictures, record videos, apply filters like greyscale, negative and ‘face only’ filter like the one that appears on Snapchat or Instagram 

## So how do we use web browser to view the live streaming ?

For implementing the computer vision part we will use the OpenCV module in Python and to display the live stream in the web browser we will use the Flask web framework.

Front-end:

Firstly, the front-end is a basic HTML file with buttons to take inputs and image source tag to display output frames after pre-processing in the back-end.
1 app.py — This is the Flask application we created above
2 templates — This folder contains our ‘index.html’ file. This is mandatory in Flask while rendering


Back-end:

Flask is a micro web-framework which works like a bridge between front and back-end. From flask, we import ‘Response’ and ‘request’ modules to handle HTTP response-requests. ‘render_template’ is used to render the HTML file shown before. OpenCV is the module used to perform all the computer vision tasks. We then declare all the global variables that act like a ‘toggle switch’ to perform different tasks like capture image, start/stop recording and applying filters. Initialize the variables to 0 to set everything to false.

When we run ‘app.py’, you will get one URL as shown in video
On clicking on the provided URL, our web browser opens up with the live feed.


https://user-images.githubusercontent.com/89068470/161105991-382b395d-b30b-453c-8285-19377143159a.mp4


