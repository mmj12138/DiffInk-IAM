import copy
import unittest
import numpy as np
from scripts.convert_iam_json import convert_line


class IAMConversionTest(unittest.TestCase):
    def setUp(self):
        self.data = {'text': 'a b', 'writer_id': '12', 'points': [
            {'x': i, 'y': i % 3, 'stroke': i // 3, 'index': i % 3}
            for i in range(6)], 'gt_segmentation': {'segments': [
                {'substring': 'a', 'inkRanges': [{'startStroke': 0, 'startPoint': 0, 'endStroke': 0, 'endPoint': 2}]},
                {'substring': ' '},
                {'substring': 'b', 'inkRanges': [{'startStroke': 1, 'startPoint': 0, 'endStroke': 1, 'endPoint': 2}]}]}}

    def test_spaces_and_exclusive_endpoints(self):
        seq, ends, text, writer, _ = convert_line(self.data)
        np.testing.assert_array_equal(ends, [3, 3, 6])
        np.testing.assert_array_equal(seq[[2, 5], 2:], [[0, 0, 1], [0, 0, 1]])
        self.assertEqual(text, 'a b')
        self.assertEqual(writer, '12')
        self.assertAlmostEqual(float(seq[:, 1].std()), 1, places=6)
        self.assertGreater(seq[0, 1], seq[1, 1])

    def test_multi_stroke_character(self):
        data = copy.deepcopy(self.data)
        data['text'] = 'a'
        data['gt_segmentation']['segments'] = [{'substring': 'a', 'inkRanges': [
            {'startStroke': 0, 'startPoint': 0, 'endStroke': 1, 'endPoint': 2}]}]
        seq, ends, *_ = convert_line(data)
        np.testing.assert_array_equal(seq[2, 2:], [0, 1, 0])
        np.testing.assert_array_equal(ends, [6])

    def test_reject_uncovered_points(self):
        self.data['gt_segmentation']['segments'][-1]['inkRanges'][0]['endPoint'] = 1
        with self.assertRaisesRegex(ValueError, 'outside segmentation'):
            convert_line(self.data)

    def test_reject_overlapping_ranges(self):
        self.data['gt_segmentation']['segments'][-1]['inkRanges'][0]['startStroke'] = 0
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            convert_line(self.data)


if __name__ == '__main__':
    unittest.main()
