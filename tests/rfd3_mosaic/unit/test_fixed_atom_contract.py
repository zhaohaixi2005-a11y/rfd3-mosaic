"""Dependency-free contract tests; these do not substitute for native integration."""
import unittest
from rfd3_mosaic import fixed_atom_contract as contract

class FixedAtomContractTest(unittest.TestCase):
    def resolve(self, atoms="all", explicit=False, conditioned=True, redesign=True):
        return contract.resolve_fixed_atom_scope(atoms, explicit=explicit,
            sequence_conditioned=conditioned, redesign=redesign)

    def test_omitted_default_and_explicit_redesign(self):
        self.assertEqual(self.resolve(), "backbone")

    def test_explicit_backbone_with_redesign(self):
        self.assertEqual(self.resolve("backbone", True), "backbone")

    def test_explicit_all_conflict(self):
        with self.assertRaisesRegex(ValueError, "Explicit atoms=all"):
            self.resolve(explicit=True)

    def test_sequence_clause_alone_preserves_all(self):
        for explicit in (False, True):
            self.assertEqual(self.resolve(explicit=explicit, redesign=False), "all")

    def test_allatom_mode_preserved(self):
        self.assertEqual(self.resolve(explicit=True, conditioned=False, redesign=False), "all")

    def test_physical_backbone_excludes_sidechain_and_virtual(self):
        example = {"select_fixed_atoms": {"A1-30": "BKBN", "B1-31": "BKBN"}}
        physical = [(c, r, a) for c, n in (("A",30),("B",31))
                    for r in range(1,n+1) for a in ("N","CA","C","O")]
        self.assertEqual(sum(contract.source_atom_is_fixed(example,*x) for x in physical)*3, 732)
        for name in ("CB","CG","ND1","VX","VX1"):
            self.assertFalse(contract.source_atom_is_fixed(example,"A",1,name))

    def test_selector_gaps_are_preserved(self):
        example = {"select_fixed_atoms": {"A1,A3": "BKBN"}}
        self.assertTrue(contract.source_atom_is_fixed(example,"A",3,"CA"))
        self.assertFalse(contract.source_atom_is_fixed(example,"A",2,"CA"))

    def test_allatom_membership_keeps_sidechains(self):
        self.assertTrue(contract.source_atom_is_fixed({"select_fixed_atoms":{"A1":"ALL"}},"A",1,"CB"))

if __name__ == "__main__":
    unittest.main()
