"""Live camera analysis tool (forged).

Repaired 2026-08-08: this tool used to open the webcam at IMPORT time —

    camera_analysis = LiveCameraAnalysis()      # module level
    ...
    def __init__(self):
        self.capture = cv2.VideoCapture(0)

Because the forge loader imports every persisted tool during ORION's startup,
that one line made booting ORION open the camera. On Windows the probe blocks
or fails hard ("cv::obsensor ... Camera index out of range" — the line in every
orion_startup*.log), and ORION never reached the Command Deck at all.

The camera is now opened lazily inside run(), used, and always released. The
loader also bounds tool imports now, so a future tool doing this cannot stall
startup again — but a tool should not need that safety net.
"""

class LiveCameraAnalysis:
    """Opens the camera only when actually asked to look at something."""

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.capture = None

    def open(self) -> bool:
        import cv2
        if self.capture is not None and self.capture.isOpened():
            return True
        self.capture = cv2.VideoCapture(self.index)
        return bool(self.capture and self.capture.isOpened())

    def analyze_frame(self, frame) -> str:
        """Report what can honestly be measured from the frame.

        The original returned the constant string "Analyzed frame content"
        regardless of the image — a placeholder presented as a result. These
        are real measurements of the captured frame.
        """
        height, width = frame.shape[:2]
        mean = frame.mean()
        brightness = "dark" if mean < 60 else "dim" if mean < 120 else "well lit"
        return (f"Captured a {width}x{height} frame; it is {brightness} "
                f"(mean intensity {mean:.0f} of 255).")

    def run_analysis(self) -> str:
        if not self.open():
            return (f"No camera available at index {self.index} — nothing was "
                    "captured.")
        ret, frame = self.capture.read()
        if not ret or frame is None:
            return "The camera opened but returned no frame."
        return self.analyze_frame(frame)

    def release(self) -> None:
        if self.capture is not None:
            try:
                self.capture.release()
            finally:
                self.capture = None


def get_tool_schema():
    return {
        "name": "live_camera_analysis",
        "description": "Capture one frame from the live camera and report what "
                       "is measurable about it (size and lighting). Opens the "
                       "camera on demand and releases it immediately.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }


def run():
    # Constructed per call: nothing holds the camera open between uses, and
    # importing this module touches no hardware at all.
    analyser = LiveCameraAnalysis()
    try:
        return analyser.run_analysis()
    except Exception as exc:
        return f"Error: {exc}"
    finally:
        analyser.release()
