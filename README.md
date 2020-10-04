# Experiment using TensorFlow lite with Kivy

This is an experimental project, no support will be done.



It uses:
- Kivy android camera to grab RGBA pixels
- Pillow to convert the image to the tensor input format
- TensorFlow lite Android libraries to detect

This example is mostly based on the Image classification from TensorFlow
lite example repository.

## Create a model

- Train your model https://teachablemachine.withgoogle.com/
- Export as a Quantized model for TensorFlow lite
- Copy it into the app, ajust the paths.
