import unittest
import numpy as np
from check.iam_prefix import prefix_info


class PrefixAuditTest(unittest.TestCase):
    def test_character_boundary_and_latent_rounding(self):
        seq=np.zeros((200,5),dtype=np.float32)
        seq[:,2]=1
        ends=[40,91,120,150,180,200]
        seq[np.array(ends)-1,2:]=[0,0,1]
        info=prefix_info(seq,ends,'photos')
        self.assertEqual(info['prefix_text'],'ph')
        self.assertEqual(info['prefix_end_exclusive'],91)
        self.assertEqual(info['retained_latent_tokens'],12)
        self.assertEqual(info['effective_prefix_end_exclusive'],96)
        self.assertEqual(info['actual_latent_generation_mask'],[0.]*12+[1.]*13)

    def test_spaces_consume_no_points(self):
        seq=np.zeros((24,5),dtype=np.float32)
        seq[:,2]=1
        seq[[7,23],2:]=[0,0,1]
        info=prefix_info(seq,[8,8,24],'a b')
        self.assertEqual(info['prefix_text'],'a')
        self.assertEqual(info['prefix_end_exclusive'],8)
        self.assertEqual(info['extra_points_in_boundary_block'],0)

    def test_rejects_incorrect_endpoint_state(self):
        seq=np.zeros((16,5),dtype=np.float32)
        seq[:,2]=1
        with self.assertRaisesRegex(ValueError,'end-of-character'):
            prefix_info(seq,[8,16],'ab')


if __name__=='__main__':unittest.main()
