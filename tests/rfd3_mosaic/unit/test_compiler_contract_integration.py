import tempfile
import json
import yaml
from rfd3_mosaic.output.rfd3_adapter import compile_assembly_rfd3_input
import unittest
from pathlib import Path
from rfd3_mosaic.constraint_plan import compile_constraint_plan
from rfd3_mosaic.design_compiler import lower_user_design
from rfd3_mosaic.schema.design import UserDesignSpec

class CompilerContractIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "motif.pdb"
        lines=[]
        for serial, atom in enumerate(("N", "CA", "C", "O", "CB"), 1):
            lines.append(f"ATOM  {serial:5d} {atom:>4s} ALA A   1    {30.0+serial:8.3f}{0.:8.3f}{0.:8.3f}  1.00 20.00          {atom[0]:>2s}\n")
        self.path.write_text("".join(lines)+"END\n")

    def tearDown(self):
        self.tmp.cleanup()

    def design(self, atoms=None, redesign=True):
        clause={"kind":"fixed_xyz","selector":"A1"}
        if atoms is not None: clause["atoms"]=atoms
        return UserDesignSpec.model_validate({"name":"contract-test","input":str(self.path),"symmetry":"C3",
            "generation":[{"kind":"terminal","anchor":"A1","terminus":"c","length":20}],"constraints":[clause],"conditioning":{"sequence":[{"selector":"A1","mode":"masked"}],
                "redesign_motif_sidechains":redesign}})

    def adapter_payload(self, design):
        lowered=lower_user_design(design)
        config=Path(self.tmp.name)/"assembly.yaml"
        config.write_text(yaml.safe_dump({"assembly":lowered.specification.model_dump(mode="json")}))
        outputs=compile_assembly_rfd3_input(config,Path(self.tmp.name)/"adapter",
            base_directory=self.tmp.name,extra_metadata={**lowered.runtime_constraint_metadata,
                "constraint_plan":lowered.constraint_plan.model_dump(mode="json")})
        return lowered,next(iter(json.loads(outputs.input_path.read_text()).values()))

    def test_scope_reaches_real_adapter_unchanged(self):
        for atoms,redesign in ((None,True),("backbone",True),("all",False)):
            with self.subTest(atoms=atoms,redesign=redesign):
                design=self.design(atoms,redesign)
                if not redesign:
                    payload=design.model_dump(mode="json");payload.pop("conditioning");design=UserDesignSpec.model_validate(payload)
                lowered,native=self.adapter_payload(design)
                expected="BKBN" if redesign else "ALL"
                self.assertEqual(set(native["select_fixed_atoms"].values()),{expected})
                self.assertEqual(lowered.constraint_plan.operators[0].atoms.value,"backbone" if redesign else "all")
                self.assertEqual(native["extra"]["constraint_plan"]["operators"][0]["atoms"],"backbone" if redesign else "all")
                if redesign:self.assertTrue(native["select_unfixed_sequence"])
                else:self.assertFalse(native.get("select_unfixed_sequence"))
        with self.assertRaisesRegex(ValueError,"Explicit atoms=all"):
            self.adapter_payload(self.design("all",True))

    def graph_design(self, redesign):
        text=self.path.read_text().replace("END\n", "")
        second="".join(line[:30]+f"{float(line[30:38])+12.:8.3f}"+line[38:] for line in text.replace("ALA A   1", "ALA A   2").splitlines(True))
        text+=second
        self.path.write_text(text)
        return UserDesignSpec.model_validate({"name":"graph-test","input":str(self.path),"symmetry":"C3",
            "components":{"core":{"selectors":["A1"]},"tail":{"selectors":["A2"]}},
            "connections":[{"id":"extension","from":{"component":"core","selector":"A1","terminus":"c"},
                "to":{"component":"tail","selector":"A2","terminus":"n"},"length":20}],
            "conditioning":{"redesign_motif_sidechains":redesign}})

    def test_graph_with_generation_redesign_rejected_early(self):
        with self.assertRaisesRegex(ValueError,"Graph components with sidechain redesign"):
            lower_user_design(self.graph_design(True))

    def test_graph_without_redesign_retains_all(self):
        design=self.graph_design(False)
        self.assertTrue(all(op.atoms.value=="all" for op in compile_constraint_plan(design).operators))
        lowered,native=self.adapter_payload(design)
        self.assertEqual(set(native["select_fixed_atoms"].values()),{"ALL"})

    def test_omitted_atoms_redesign_agrees_plan_fragment(self):
        design=self.design()
        self.assertEqual(compile_constraint_plan(design).operators[0].atoms.value,"backbone")
        lowered=lower_user_design(design)
        self.assertTrue(all(f.fixed_atoms=="backbone" for f in lowered.specification.fragments.values()))

    def test_explicit_backbone_agrees_plan_fragment(self):
        design=self.design("backbone")
        self.assertEqual(compile_constraint_plan(design).operators[0].atoms.value,"backbone")
        self.assertTrue(all(f.fixed_atoms=="backbone" for f in lower_user_design(design).specification.fragments.values()))

    def test_explicit_all_redesign_conflict(self):
        with self.assertRaisesRegex(ValueError,"Explicit atoms=all"):
            lower_user_design(self.design("all"))

    def test_allatom_without_sequence_redesign_preserved(self):
        payload=self.design("all",False).model_dump(mode="json")
        payload.pop("conditioning")
        design=UserDesignSpec.model_validate(payload)
        self.assertEqual(compile_constraint_plan(design).operators[0].atoms.value,"all")
        self.assertTrue(all(f.fixed_atoms=="all" for f in lower_user_design(design).specification.fragments.values()))
    def test_masked_clause_alone_keeps_all(self):
        design=self.design(redesign=False)
        self.assertEqual(compile_constraint_plan(design).operators[0].atoms.value,"all")
        with self.assertRaisesRegex(ValueError, "requires fixed_atoms=backbone"):
            lower_user_design(design)

if __name__ == "__main__": unittest.main()
