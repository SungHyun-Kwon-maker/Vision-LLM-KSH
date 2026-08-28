"""Model-independent tests for the educational vehicle assistant."""

from dataclasses import dataclass
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from project.vehicle_assistant import (
    GestureVolumeFilter,
    InteractionState,
    RateGate,
    RoadScene,
    RoadVisionWorker,
    SignalObservation,
    TemporalSignalFilter,
    build_gemma_messages,
    build_volume_get_command,
    build_volume_set_command,
    compact_vision_context,
    detection_to_context_object,
    hand_pose_measurements,
    is_road_class,
    parse_volume_output,
    volume_from_pinch_ratio,
)


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class FakeFrame:
    shape = (100, 200, 3)

    def copy(self):
        return self


class RecordingTTS:
    def __init__(self):
        self.messages = []
        self.spoken = threading.Event()

    def speak(self, text):
        self.messages.append(text)
        self.spoken.set()
        return True


def valid_control_hand():
    points = [Point() for _ in range(21)]
    points[4] = Point(0.0, 0.0)
    points[5] = Point(0.0, 0.0)
    points[6] = Point(1.0, 0.0)
    points[7] = Point(2.0, 0.0)  # index angle: 180 degrees
    points[8] = Point(1.0, 0.0)
    points[9] = Point(0.0, 0.0)
    points[10] = Point(1.0, 0.0)
    points[11] = Point(1.0, 1.0)  # folded: 90 degrees
    points[13] = Point(0.0, 0.0)
    points[14] = Point(1.0, 0.0)
    points[15] = Point(1.0, 1.0)
    points[17] = Point(0.0, 1.0)
    points[18] = Point(1.0, 1.0)
    points[19] = Point(1.0, 2.0)
    return points


class FramePolicyTest(unittest.TestCase):
    def test_signal_requires_five_votes(self):
        signal_filter = TemporalSignalFilter(window_size=7, required_votes=5)
        for _ in range(4):
            state, changed, _ = signal_filter.update("red")
            self.assertEqual(state, "unknown")
            self.assertFalse(changed)

        state, changed, votes = signal_filter.update("red")
        self.assertEqual((state, changed, votes), ("red", True, 5))

        for _ in range(4):
            state, _, _ = signal_filter.update("green")
            self.assertEqual(state, "red")
        state, changed, votes = signal_filter.update("green")
        self.assertEqual((state, changed, votes), ("green", True, 5))

    def test_rate_gate_drops_intermediate_times(self):
        gate = RateGate(10.0)
        self.assertTrue(gate.due(0.0))
        self.assertFalse(gate.due(0.05))
        self.assertTrue(gate.due(0.10))

    def test_road_queue_keeps_only_latest_frame(self):
        args = SimpleNamespace(
            stability_window=7,
            stability_votes=5,
            announcement_cooldown=5.0,
        )
        worker = RoadVisionWorker(
            detector=None,
            args=args,
            gpu_lock=threading.Lock(),
            interaction=InteractionState(),
            tts=RecordingTTS(),
        )
        worker.submit(1, 1.0, FakeFrame())
        worker.submit(2, 2.0, FakeFrame())
        sequence, captured_at, _ = worker._queue.get_nowait()
        worker._queue.task_done()
        self.assertEqual((sequence, captured_at), (2, 2.0))

    def test_stable_signal_event_uses_rule_tts_without_gemma(self):
        args = SimpleNamespace(
            stability_window=1,
            stability_votes=1,
            announcement_cooldown=5.0,
        )
        tts = RecordingTTS()
        worker = RoadVisionWorker(
            detector=None,
            args=args,
            gpu_lock=threading.Lock(),
            interaction=InteractionState(),
            tts=tts,
        )
        scene = RoadScene(
            signal=SignalObservation(
                box=None,
                confidence=0.0,
                raw_state="red",
                scores={"red": 1.0, "yellow": 0.0, "green": 0.0},
            ),
            objects=tuple(),
        )
        with patch(
            "project.vehicle_assistant.detect_road_scene",
            return_value=scene,
        ), patch("builtins.print"):
            worker.start()
            worker.submit(1, 1.0, FakeFrame())
            self.assertTrue(tts.spoken.wait(1.0))
            worker.close()

        self.assertEqual(len(tts.messages), 1)
        self.assertIn("빨간불", tts.messages[0])
        self.assertEqual(worker.latest().stable_state, "red")


class GestureTest(unittest.TestCase):
    def test_pinch_ratio_maps_to_five_percent_steps(self):
        self.assertEqual(volume_from_pinch_ratio(0.0), 0)
        self.assertEqual(volume_from_pinch_ratio(0.20), 0)
        self.assertEqual(volume_from_pinch_ratio(0.85), 50)
        self.assertEqual(volume_from_pinch_ratio(1.50), 100)
        self.assertEqual(volume_from_pinch_ratio(3.0), 100)

    def test_pose_requires_extended_index_and_folded_other_fingers(self):
        points = valid_control_hand()
        valid, ratio = hand_pose_measurements(points)
        self.assertTrue(valid)
        self.assertAlmostEqual(ratio, 1.0)

        points[11] = Point(2.0, 0.0)  # middle finger becomes straight
        valid, _ = hand_pose_measurements(points)
        self.assertFalse(valid)

    def test_activation_smoothing_and_release(self):
        gesture_filter = GestureVolumeFilter(initial_volume=50)
        for frame_index in range(5):
            decision = gesture_filter.update(True, 1.50, frame_index * 0.1)
            self.assertFalse(decision.active)
            self.assertIsNone(decision.target_volume)

        decision = gesture_filter.update(True, 1.50, 0.5)
        self.assertTrue(decision.active)
        self.assertTrue(decision.just_activated)
        self.assertEqual(decision.target_volume, 100)

        decision = gesture_filter.update(True, 0.20, 0.8)
        self.assertEqual(decision.target_volume, 75)

        for frame_index in range(7):
            decision = gesture_filter.update(False, None, 1.0 + frame_index * 0.1)
            self.assertTrue(decision.active)
        decision = gesture_filter.update(False, None, 1.7)
        self.assertFalse(decision.active)
        self.assertTrue(decision.just_deactivated)


class VisionContextTest(unittest.TestCase):
    def test_only_planned_coco_road_classes_are_allowed(self):
        self.assertTrue(is_road_class("person"))
        self.assertTrue(is_road_class("traffic light"))
        self.assertFalse(is_road_class("chair"))

    def test_detection_context_has_position_but_no_physical_distance(self):
        detected = detection_to_context_object(
            class_id=2,
            class_name="car",
            confidence=0.9123,
            bbox=(700, 200, 1100, 650),
            image_width=1200,
            image_height=700,
        )
        self.assertEqual(detected["class_ko"], "자동차")
        self.assertEqual(detected["horizontal_position"], "오른쪽")
        self.assertEqual(detected["confidence"], 0.912)
        self.assertIn("화면", detected["size_hint"])

    def test_gemma_prompt_contains_exactly_one_road_image(self):
        vision_data = {
            "objects": [],
            "selected_traffic_light": None,
        }
        messages = build_gemma_messages(
            "앞에 무엇이 있어?",
            "data:image/jpeg;base64,ROAD_FRAME",
            vision_data,
        )
        user_content = messages[1]["content"]
        images = [item for item in user_content if item["type"] == "image_url"]
        self.assertEqual(len(images), 1)
        self.assertEqual(
            images[0]["image_url"]["url"],
            "data:image/jpeg;base64,ROAD_FRAME",
        )

    def test_gemma_structured_context_is_limited_to_twelve_objects(self):
        vision_data = {
            "objects": [
                {"class": "car", "sequence": index} for index in range(15)
            ],
            "selected_traffic_light": None,
        }
        compact = compact_vision_context(vision_data)
        self.assertEqual(len(compact["objects"]), 12)
        self.assertEqual(compact["omitted_object_count"], 3)


class VolumeCommandTest(unittest.TestCase):
    def test_backend_commands_are_argument_lists_and_clamped(self):
        self.assertEqual(
            build_volume_get_command("wpctl"),
            ["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"],
        )
        self.assertEqual(
            build_volume_set_command("wpctl", 150)[-1],
            "1.00",
        )
        self.assertEqual(
            build_volume_set_command("pactl", -10)[-1],
            "0%",
        )
        command = build_volume_set_command(
            "amixer",
            55,
            alsa_card="hw:2",
            alsa_control="Speaker Volume",
        )
        self.assertEqual(
            command,
            [
                "amixer",
                "-D",
                "hw:2",
                "sset",
                "Speaker Volume",
                "55%",
            ],
        )

    def test_volume_output_parsing(self):
        self.assertEqual(parse_volume_output("wpctl", "Volume: 0.42"), 42)
        self.assertEqual(
            parse_volume_output(
                "pactl",
                "Volume: front-left: 32768 / 50% / -18.06 dB",
            ),
            50,
        )
        self.assertEqual(
            parse_volume_output(
                "amixer",
                "Mono: Playback 32 [50%] [-16.50dB] [on]",
            ),
            50,
        )


if __name__ == "__main__":
    unittest.main()
