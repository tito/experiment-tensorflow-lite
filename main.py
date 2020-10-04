from kivy.app import App
from kivy.factory import Factory as F
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty
from kivy.clock import Clock, mainthread
from time import time

Builder.load_string('''
<CameraClassifier@RelativeLayout>:
    orientation: 'vertical'
    Camera:
        id: camera
        resolution: (1920, 1080)
        play: False
    Label:
        text: app.normalized_result
        text_size: self.width - dp(48), self.height - dp(48)
        y: dp(24)
        markup: True
''')


class TestCamera(App):

    normalized_result = StringProperty("")

    def build(self):
        Clock.schedule_once(self.start_camera, .5)
        return F.CameraClassifier()

    def start_camera(self, *largs):
        self.root.ids.camera.play = True
        self.root.ids.camera._camera.bind(on_texture=self.on_camera_texture)
        from tflwrapper.tfl_android import TFLWrapperAndroid
        self.tflite = TFLWrapperAndroid(
            "data/pba_quantized/model.tflite",
            "data/pba_quantized/labels.txt",
            on_detect=self.on_tflite_detect)

    def on_camera_texture(self, camera):
        pixels = camera._fbo.pixels
        w, h = camera.resolution
        self.tflite.async_detect(pixels, w, h)

    @mainthread
    def on_tflite_detect(self, result):
        self.result = result
        labels = self.tflite.get_labels_with_value(result)
        textresult = []
        for label, value in labels:
            text = f"{value:.05f}: {label}"
            if value > .8:
                text = f"[color=44FF44]{text}[/color]"
            else:
                text = f"[color=FF4444]{text}[/color]"
            textresult.append(text)
        self.normalized_result = "\n".join(textresult)

    def detect(self, *largs):
        timer = time()
        camera = self.root.ids.camera._camera
        w, h = camera.resolution
        pixels = camera._fbo.pixels
        self.result = self.tfl.detect(pixels, w, h)

TestCamera().run()