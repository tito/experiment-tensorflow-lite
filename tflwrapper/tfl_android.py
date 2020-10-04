from jnius import autoclass, cast
from PIL import Image
import os
import threading


File = autoclass("java.io.File")
Interpreter = autoclass("org.tensorflow.lite.Interpreter")
InterpreterOptions = autoclass("org.tensorflow.lite.Interpreter$Options")
TensorImage = autoclass("org.tensorflow.lite.support.image.TensorImage")
TensorBuffer = autoclass("org.tensorflow.lite.support.tensorbuffer.TensorBuffer")
ByteBuffer = autoclass("java.nio.ByteBuffer")


class TFLWrapper(threading.Thread):
    def __init__(self, model, labels, on_detect=None):
        super().__init__()
        self.event = threading.Event()
        self.quit = False
        self.on_detect = on_detect
        self.async_running = False
        self.init_from_model(model, labels)

    def init_from_model(self, model, labels):
        raise NotImplementedError()

    def async_start(self):
        if self.async_running:
            return
        self.next_frame = None
        self.daemon = True
        self.async_running = True
        self.start()

    def async_stop(self):
        self.quit = True

    def run(self):
        try:
            while not self.quit:
                if self.event.wait(0.5) is None:
                    continue
                next_frame, self.next_frame = self.next_frame, None
                self.event.clear()
                if next_frame is None:
                    continue
                result = self.detect(*next_frame)
                if self.on_detect:
                    self.on_detect(result)
        except Exception as e:
            print("Exception in TFLWrapper:", e)
            import traceback; traceback.print_exc()


class TFLWrapperAndroid(TFLWrapper):
    def init_from_model(self, model, labels):
        options = InterpreterOptions()
        options.setNumThreads(4)
        model_filename = os.path.realpath(model)
        label_filename = os.path.realpath(labels)
        model = File(model_filename)

        with open(label_filename, encoding="utf-8") as fd:
            self.labels = {}
            for line in fd.read().strip().splitlines():
                if not line.strip():
                    return
                index, label = line.split(" ", 1)
                self.labels[int(index)] = label

        tflite = Interpreter(model, options)
        shape = tflite.getInputTensor(0).shape()
        imgwidth, imgheight = shape[1:3]

        # Creates the input tensor.
        imageDataType = tflite.getInputTensor(0).dataType()
        inputImageBuffer = TensorImage(imageDataType)

        # Create output tensor
        probabilityShape = tflite.getOutputTensor(0).shape()
        probabilityDataType = tflite.getOutputTensor(0).dataType()
        outputProbabilityBuffer = TensorBuffer.createFixedSize(
            probabilityShape, probabilityDataType)

        self.tflite = tflite
        self.inputImageBuffer = inputImageBuffer
        self.outputProbabilityBuffer = outputProbabilityBuffer
        self.imgwidth = imgwidth
        self.imgheight = imgheight

    def async_detect(self, frame, cw, ch):
        self.async_start()
        self.next_frame = (frame, cw, ch)
        self.event.set()

    def detect(self, frame, cw, ch):
        cropsize = min(cw, ch)

        # use pillow
        image = Image.frombytes("RGBA", (cw, ch), frame, "raw")
        left = (cw - cropsize) / 2
        top = (ch - cropsize) / 2
        image = image.crop((left, top, left + cropsize, top + cropsize))
        image = image.resize((self.imgwidth, self.imgheight))
        image = image.convert("RGB")
        pixels = image.tobytes()

        buffer = ByteBuffer.wrap(pixels)
        self.tflite.run(buffer, self.outputProbabilityBuffer.getBuffer().rewind())
        result = self.outputProbabilityBuffer.getFloatArray()
        return result

    def get_labels_with_value(self, result):
        unsorted_labels = [(
            self.labels[index], value / 255
        ) for index, value in enumerate(result)]
        return list(sorted(unsorted_labels, key=lambda x: x[1], reverse=True))