"""Required supplied interfaces must survive ordinary motion-group lowering."""
import copy
import unittest

from pydantic import ValidationError

from rfd3_mosaic.schema.design import UserDesignSpec


def payload():
    return {
        "name": "mixed-interface", "input": "seed.pdb", "symmetry": "C3",
        "components": {
            "a": {"selectors": ["A1-3"], "pose": {
                "mode": "bounded_mobile", "max_translation": 2,
                "max_rotation_deg": 10}},
            "b": {"selectors": ["B1-3"]},
        },
        "interfaces": [{"id": "keep", "between": ["a", "b"],
                        "relation": {"mode": "preserve_input"}}],
    }


class RequiredPreserveMotionTests(unittest.TestCase):
    def test_required_preserve_without_task_rejects_independent_motion(self):
        p = payload()
        original = copy.deepcopy(p)
        with self.assertRaisesRegex(ValidationError, "keep.*joint-rigid"):
            UserDesignSpec.model_validate(p)
        self.assertEqual(p, original)

    def test_required_preserve_resolves_ports_before_checking_motion(self):
        p = payload()
        p["ports"] = {"left": {"component": "a", "selectors": ["A1-3"]},
                      "right": {"component": "b", "selectors": ["B1-3"]}}
        p["interfaces"][0]["between"] = ["left", "right"]
        with self.assertRaisesRegex(ValidationError, "keep.*joint-rigid"):
            UserDesignSpec.model_validate(p)

    def test_locked_supplied_components_remain_supported(self):
        p = payload()
        p["components"]["a"].pop("pose")
        self.assertIsNone(UserDesignSpec.model_validate(p).task)

    def test_advisory_preserve_does_not_promote_to_a_hard_contract(self):
        p = payload()
        p["interfaces"][0]["required"] = False
        self.assertFalse(UserDesignSpec.model_validate(p).interfaces[0].required)

    def test_contact_between_mobile_components_remains_supported(self):
        p = payload()
        p["interfaces"][0]["relation"] = {"mode": "contact"}
        self.assertEqual(UserDesignSpec.model_validate(p).interfaces[0].relation.mode,
                         "contact")

    def test_joint_seed_can_preserve_interface_and_request_new_contact(self):
        p = payload()
        pose = p["components"]["a"]["pose"]
        p["components"] = {"seed": {"selectors": ["A1-3", "B1-3"],
                                    "geometry": "joint_rigid", "pose": pose},
                           "other": {"selectors": ["C1-3"]}}
        p["ports"] = {"left": {"component": "seed", "selectors": ["A1-3"]},
                      "right": {"component": "seed", "selectors": ["B1-3"]},
                      "third": {"component": "other", "selectors": ["C1-3"]}}
        p["interfaces"] = [{"id": "keep", "between": ["left", "right"],
                            "relation": {"mode": "preserve_input"}},
                           {"id": "new", "between": ["right", "third"],
                            "relation": {"mode": "contact"}}]
        self.assertEqual(len(UserDesignSpec.model_validate(p).interfaces), 2)


if __name__ == "__main__":
    unittest.main()
