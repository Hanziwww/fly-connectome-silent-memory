"""Viewer contracts: model-specific heads, catalog isolation and multi-box validation."""
import unittest
from unittest.mock import patch

import numpy as np

from serve_joint import JointPlayer, MODELS


class JointViewerTests(unittest.TestCase):
    def test_catalog_is_verified_and_contains_no_saved_action_sequences(self):
        player=JointPlayer()
        data=player.catalog()
        self.assertIn('P03',player.levels)
        self.assertIn(data['default_suite'],{'joint_linear','joint_p05'})
        self.assertTrue({'pilot','repair','joint_linear'}<=set(data['suites']))
        self.assertEqual(data['results']['linear_joint']['joint_linear']['test']['records'][0]['steps'],22)
        self.assertEqual(data['joint_summary']['updates'],216)
        def inspect(value):
            if isinstance(value,dict):
                self.assertFalse({'actions','reference_actions','decisions'} & value.keys())
                for child in value.values(): inspect(child)
            elif isinstance(value,list):
                for child in value: inspect(child)
        inspect(data)

    def test_new_task_rejects_cross_suite_models_and_invalid_boxes(self):
        player=JointPlayer()
        level=player.levels['P03']
        message=dict(level_id='P03',model='linear_joint',state=level['start'])
        self.assertEqual(player.validate(message)[-1],'linear_joint')
        with self.assertRaises(ValueError): player.validate(dict(message,model='brain'))
        with self.assertRaises(ValueError): player.validate(dict(message,state=[22,[36,36]]))

    def test_each_verified_task_keeps_its_models_and_reference_separate(self):
        player=JointPlayer()
        catalog=player.catalog()
        for level_id,suite_key in player.joint_levels.items():
            level=player.levels[level_id]
            suite=catalog['suites'][suite_key]
            self.assertEqual(catalog['joint_experiments'][suite_key]['summary']['task_id'],level_id)
            for kind in suite['models']:
                record=catalog['results'][kind][suite_key]['test']['records'][0]
                self.assertEqual(record['level_id'],level_id)
                self.assertEqual(player.validate(dict(level_id=level_id,model=kind,state=level['start']))[-1],kind)
            for other_suite in catalog['suites'].values():
                if other_suite is suite:
                    continue
                with self.assertRaises(ValueError):
                    player.validate(dict(level_id=level_id,model=other_suite['default_model'],state=level['start']))

    def test_current_and_reference_signals_use_their_own_readouts(self):
        class FakeRuntime:
            def __init__(self):
                self.heads={kind:object() for kind in MODELS}
                self.ids={kind:kind for kind in MODELS}
            def response(self,board,kind):
                return np.zeros(512),np.zeros((2,1)),dict(checkpoint_id=kind)
        player=JointPlayer()
        player.runtime=FakeRuntime()
        seen=[]
        def fake_decide(head,features,mask,checkpoint):
            seen.append(head)
            return dict(action=3,final_probabilities=[0.,0.,0.,1.],brain_logits=[1.,2.,3.,4.],
                        correction_logits=[0.,0.,0.,0.],checkpoint_id=checkpoint)
        with patch('serve_joint.decide',side_effect=fake_decide):
            response=player.respond(dict(level_id='P03',model='linear_joint',state=player.levels['P03']['start']),True)
        self.assertIs(seen[0],player.runtime.heads['linear_joint'])
        self.assertIs(seen[1],player.runtime.heads['linear_source'])
        self.assertIsNot(seen[0],seen[1])
        self.assertEqual(response['reference_kind'],'before-joint-learning')
        self.assertEqual(response['reference_label'],'before P03 training')
        self.assertEqual(response['state'][0],23)


if __name__=='__main__':
    unittest.main()
