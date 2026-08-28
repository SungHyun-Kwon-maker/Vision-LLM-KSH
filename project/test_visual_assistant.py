"""Tests for model-independent visual-assistant logic."""

import unittest

from project.visual_assistant import (
    detection_to_object,
    detections_to_text,
    estimate_proximity,
    horizontal_position,
    vertical_position,
)


class PositionTest(unittest.TestCase):
    def test_horizontal_thirds(self):
        self.assertEqual(horizontal_position(100, 900), "왼쪽")
        self.assertEqual(horizontal_position(450, 900), "정면")
        self.assertEqual(horizontal_position(800, 900), "오른쪽")

    def test_vertical_thirds(self):
        self.assertEqual(vertical_position(50, 600), "위쪽")
        self.assertEqual(vertical_position(300, 600), "중앙 높이")
        self.assertEqual(vertical_position(550, 600), "아래쪽")

    def test_proximity_uses_bbox_ratio(self):
        self.assertEqual(estimate_proximity(0.30), "매우 가까워 보입니다")
        self.assertEqual(estimate_proximity(0.15), "가까워 보입니다")
        self.assertEqual(estimate_proximity(0.05), "중간 거리로 보입니다")
        self.assertEqual(estimate_proximity(0.01), "멀리 있어 보입니다")


class StructuredOutputTest(unittest.TestCase):
    def test_detection_has_korean_name_position_and_clamped_bbox(self):
        detected = detection_to_object(
            class_name="chair",
            confidence=0.91234,
            bbox=(-10, 100, 950, 590),
            image_width=900,
            image_height=600,
        )

        self.assertEqual(detected["class_ko"], "의자")
        self.assertEqual(detected["confidence"], 0.912)
        self.assertEqual(
            detected["bbox"],
            {"x1": 0, "y1": 100, "x2": 900, "y2": 590},
        )
        self.assertEqual(detected["position"]["horizontal"], "정면")
        self.assertEqual(detected["position"]["vertical"], "중앙 높이")
        self.assertEqual(detected["proximity_hint"], "매우 가까워 보입니다")

    def test_empty_detection_text_does_not_claim_scene_is_empty(self):
        text = detections_to_text({"objects": []})
        self.assertIn("탐지 결과에는 객체가 없습니다", text)
        self.assertIn("탐지 실패 가능성", text)

    def test_context_contains_position_and_distance_caveat(self):
        detected = detection_to_object(
            class_name="person",
            confidence=0.8,
            bbox=(10, 20, 110, 220),
            image_width=600,
            image_height=400,
        )
        text = detections_to_text({"objects": [detected]})

        self.assertIn("탐지 개수 요약: 사람 1개", text)
        self.assertIn("사람(person)", text)
        self.assertIn("카메라 화면의 왼쪽", text)
        self.assertIn("실제 깊이가 아니라", text)


if __name__ == "__main__":
    unittest.main()
