import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import yaml
from pydantic import ValidationError

from rfd3_mosaic.cli import main
from rfd3_mosaic.compile import expand_symmetry_instances
from rfd3_mosaic.design_compiler import lower_user_design
from rfd3_mosaic.rfd3_prevalidate import prevalidate_rfd3_input
from rfd3_mosaic.schema import SimpleCageIntentSpec, UserDesignTask
from rfd3_mosaic.schema.simple_intent import load_simple_cage_intent
from rfd3_mosaic.simple_resolver import (
    enumerate_simple_design_candidates,
    resolve_simple_intent,
)


class SimpleIntentResolverTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.structure = self.root / "interface_seed.pdb"
        # Enumeration is deliberately structure-independent, but the emitted
        # public designs must retain a real, replayable input path.
        self.structure.write_text("END\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _intent(
        self,
        *,
        use: object = "auto",
        symmetries: tuple[str, ...] = ("C3",),
    ) -> SimpleCageIntentSpec:
        return SimpleCageIntentSpec.model_validate(
            {
                "name": "ordinary-interface-ring",
                "input": self.structure,
                "goal": {
                    "architecture": "ring",
                    "composition": "homomer",
                    "symmetry": list(symmetries),
                },
                "interface_seeds": {
                    "supplied_interface": {
                        "participants": ["A", "B"],
                        "selectors": {
                            "A": "A/1-2/*",
                            "B": "B/1-2/*",
                        },
                        "use": use,
                        "geometry": "preserve_exact",
                    }
                },
                "generation": {
                    "length": {"minimum": 20, "maximum": 60}
                },
            }
        )

    def test_ordinary_supplied_interface_cannot_be_deformed(self) -> None:
        payload = self._intent().model_dump(mode="python")
        payload["interface_seeds"]["supplied_interface"]["geometry"] = (
            "bounded"
        )

        with self.assertRaisesRegex(ValidationError, "preserve_exact"):
            SimpleCageIntentSpec.model_validate(payload)

    @staticmethod
    def _candidate_signature(candidate) -> tuple[object, ...]:
        return (
            candidate.candidate_id,
            candidate.symmetry,
            candidate.topology_id,
            tuple(candidate.connection_order),
            tuple(candidate.unresolved_variables),
            candidate.design.model_dump(mode="json"),
        )

    def test_c3_enumerates_four_deterministic_direction_offset_candidates(
        self,
    ) -> None:
        intent = self._intent(symmetries=("C3",))

        first = enumerate_simple_design_candidates(
            intent,
            symmetry_ids=("C3",),
            seed_start=9300,
        )
        second = enumerate_simple_design_candidates(
            intent,
            symmetry_ids=("C3",),
            seed_start=9300,
        )

        # Two possible polymer directions multiplied by the two distinct C3
        # nearest-neighbour offsets.  The resolver must expose all four rather
        # than silently choosing the first topology it happens to enumerate.
        self.assertEqual(len(first), 4)
        self.assertEqual(
            tuple(map(self._candidate_signature, first)),
            tuple(map(self._candidate_signature, second)),
        )
        self.assertEqual(
            len({candidate.candidate_id for candidate in first}),
            4,
        )
        self.assertEqual(
            len({candidate.topology_id for candidate in first}),
            4,
        )
        self.assertEqual(
            len({tuple(candidate.connection_order) for candidate in first}),
            2,
        )

    def test_c2_deduplicates_inverse_neighbour_offset(self) -> None:
        candidates = enumerate_simple_design_candidates(
            self._intent(symmetries=("C2",)),
            symmetry_ids=("C2",),
        )

        # In C2, +1 and -1 identify the same non-identity group element.  The
        # two polymer directions remain distinct, but the inverse offset must
        # not be emitted twice.
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            len({candidate.topology_id for candidate in candidates}),
            2,
        )

    def test_size_goal_is_carried_into_executable_candidates(self) -> None:
        payload = self._intent(symmetries=("C3",)).model_dump(
            mode="json"
        )
        payload["goal"]["diameter_angstrom"] = {
            "minimum": 40.0,
            "maximum": 100.0,
        }
        payload["goal"]["cavity_diameter_angstrom"] = {
            "minimum": 5.0,
            "maximum": 60.0,
        }
        intent = SimpleCageIntentSpec.model_validate(payload)

        candidates = enumerate_simple_design_candidates(
            intent,
            symmetry_ids=("C3",),
        )

        self.assertTrue(candidates)
        for candidate in candidates:
            shape = candidate.design.assembly_shape
            self.assertIsNotNone(shape)
            assert shape is not None
            self.assertEqual(
                (
                    shape.diameter_angstrom.minimum,
                    shape.diameter_angstrom.maximum,
                ),
                (40.0, 100.0),
            )
            self.assertEqual(
                (
                    shape.cavity_diameter_angstrom.minimum,
                    shape.cavity_diameter_angstrom.maximum,
                ),
                (5.0, 60.0),
            )

    def test_candidates_are_standard_expert_designs_on_the_shared_path(
        self,
    ) -> None:
        candidates = enumerate_simple_design_candidates(
            self._intent(),
            symmetry_ids=("C3",),
        )

        for candidate in candidates:
            design = candidate.design
            self.assertEqual(candidate.symmetry, "C3")
            self.assertEqual(design.symmetry, "C3")
            self.assertEqual(design.input, self.structure)
            self.assertEqual(design.user_mode, "expert")
            self.assertEqual(
                design.task,
                UserDesignTask.PRESERVE_SUPPLIED_GEOMETRY,
            )
            self.assertEqual(len(design.components), 1)
            component = next(iter(design.components.values()))
            self.assertEqual(component.geometry, "joint_rigid")
            self.assertEqual(
                set(component.selectors),
                {"A/1-2/*", "B/1-2/*"},
            )
            self.assertEqual(len(design.ports), 2)
            self.assertEqual(len(design.interfaces), 1)
            self.assertEqual(
                design.interfaces[0].relation.mode,
                "preserve_input",
            )
            self.assertEqual(len(design.connections), 1)
            self.assertFalse(design.generation)
            self.assertFalse(design.constraints)

    def test_exact_and_range_usage_filter_incompatible_cyclic_orders(
        self,
    ) -> None:
        exact = enumerate_simple_design_candidates(
            self._intent(
                use={"exact": 3},
                symmetries=("C2", "C3", "C4"),
            ),
        )
        ranged = enumerate_simple_design_candidates(
            self._intent(
                use={"minimum": 2, "maximum": 3},
                symmetries=("C2", "C3", "C4"),
            ),
        )

        self.assertEqual({candidate.symmetry for candidate in exact}, {"C3"})
        self.assertEqual(
            {candidate.symmetry for candidate in ranged},
            {"C2", "C3"},
        )

    def test_exact_partial_c4_orbit_is_lowered_after_c2_seed_validation(
        self,
    ) -> None:
        lines = []
        serial = 1
        base = (
            (1, "N", (1.0, 0.0, 0.0)),
            (1, "CA", (1.3, 0.4, 0.2)),
            (1, "C", (1.6, 0.5, 0.7)),
            (1, "O", (1.8, 0.7, 1.0)),
            (2, "N", (2.0, 0.2, 1.2)),
            (2, "CA", (2.3, 0.5, 1.5)),
            (2, "C", (2.6, 0.7, 2.0)),
            (2, "O", (2.8, 0.9, 2.2)),
        )
        for chain, sign in (("A", 1.0), ("B", -1.0)):
            for residue, atom_name, (x, y, z) in base:
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:^4s} ALA "
                    f"{chain}{residue:4d}    "
                    f"{sign*x:8.3f}{sign*y:8.3f}{z:8.3f}"
                    f"{1.0:6.2f}{20.0:6.2f}"
                    f"          {atom_name[0]:>2s}\n"
                )
                serial += 1
        self.structure.write_text(
            "".join(lines) + "END\n",
            encoding="utf-8",
        )
        intent = self._intent(use={"exact": 2}, symmetries=("C4",))

        candidates = enumerate_simple_design_candidates(intent)

        self.assertEqual(len(candidates), 2)
        for candidate in candidates:
            self.assertIsNotNone(candidate.design.finite_orbit_action)
            lowered = lower_user_design(candidate.design)
            instances = expand_symmetry_instances(lowered.specification)
            orbit = instances.constraint_orbits["motif_orbit"]
            self.assertEqual(len(orbit.transform_ids), 2)
            self.assertEqual(len(orbit.group_instance_ids), 2)

    def test_multi_participant_interface_fails_closed(self) -> None:
        intent = SimpleCageIntentSpec.model_validate(
            {
                "name": "three-participant-site",
                "input": self.structure,
                "goal": {"architecture": "ring", "symmetry": ["C3"]},
                "interface_seeds": {
                    "cooperative_site": {
                        "participants": ["A", "B", "C"],
                        "selectors": {
                            "A": "A/1-2/*",
                            "B": "B/1-2/*",
                            "C": "C/1-2/*",
                        },
                    }
                },
                "generation": {"length": 40},
            }
        )

        with self.assertRaises((NotImplementedError, ValueError)) as context:
            enumerate_simple_design_candidates(intent)

        message = str(context.exception).lower()
        self.assertTrue(
            any(
                word in message
                for word in ("multi-participant", "hyperedge", "binary")
            ),
            message,
        )

    def test_single_three_participant_interface_rebuilds_declared_paths(
        self,
    ) -> None:
        lines = []
        serial = 1
        chain_offsets = {
            "A": (0.0, 0.0),
            "B": (3.2, 0.0),
            "C": (1.6, 2.7),
        }
        for chain, (x_offset, y_offset) in chain_offsets.items():
            for residue, z_offset in ((1, 0.0), (2, 1.5), (10, 10.0), (11, 11.5)):
                for atom_name, atom_offset in (
                    ("N", (0.0, 0.0, 0.0)),
                    ("CA", (0.4, 0.2, 0.3)),
                    ("C", (0.8, 0.0, 0.6)),
                    ("O", (1.1, -0.2, 0.8)),
                ):
                    x = x_offset + atom_offset[0]
                    y = y_offset + atom_offset[1]
                    z = z_offset + atom_offset[2]
                    lines.append(
                        f"ATOM  {serial:5d} {atom_name:^4s} ALA "
                        f"{chain}{residue:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}"
                        f"{1.0:6.2f}{20.0:6.2f}"
                        f"          {atom_name[0]:>2s}\n"
                    )
                    serial += 1
        self.structure.write_text(
            "".join(lines) + "END\n",
            encoding="utf-8",
        )
        intent = SimpleCageIntentSpec.model_validate(
            {
                "name": "three-participant-explicit-paths",
                "input": self.structure,
                "goal": {
                    "architecture": "cage",
                    "composition": "auto",
                    "symmetry": ["C3"],
                },
                "interface_seeds": {
                    "cooperative_site": {
                        "participants": ["A", "B", "C"],
                        "selectors": {
                            "A": "A/1-2/*,A/10-11/*",
                            "B": "B/1-2/*,B/10-11/*",
                            "C": "C/1-2/*,C/10-11/*",
                        },
                        "use": {"exact": 3},
                        "geometry": "preserve_exact",
                    }
                },
                "generation": {"length": 7},
                "inspection": {
                    "contact_cutoff": 4.5,
                    "minimum_atom_contacts": 4,
                    "minimum_contact_residues_per_side": 2,
                },
            }
        )

        candidates = enumerate_simple_design_candidates(intent)

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(
            candidate.resolution_frontend,
            "single_supplied_hyperedge_explicit_paths_v1",
        )
        self.assertEqual(candidate.polymer_units_per_copy, 3)
        self.assertEqual(candidate.physical_polymer_unit_count, 9)
        self.assertEqual(len(candidate.design.components), 1)
        self.assertEqual(len(candidate.design.connections), 3)
        interface = candidate.design.interfaces[0]
        self.assertEqual(len(interface.between), 3)
        self.assertEqual(len(interface.contact_pairs), 2)

        lowered = lower_user_design(candidate.design)
        self.assertEqual(len(lowered.specification.interfaces), 2)
        self.assertEqual(len(lowered.interface_usage), 1)
        self.assertEqual(
            lowered.interface_usage[0].physical_instance_count,
            3,
        )
        self.assertEqual(
            {
                edge.hyperedge_id
                for edge in lowered.specification.interfaces.values()
            },
            {"cooperative_site"},
        )
        instances = expand_symmetry_instances(lowered.specification)
        self.assertEqual(len(instances.interfaces), 6)

    def test_real_pi25_three_participant_seed_resolves_one_c3_quotient(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[3]
        intent = load_simple_cage_intent(
            repository
            / "experiments"
            / "lrz_simple_three_participant_c3_quotient_v100_50step_intent.yaml"
        )

        candidates = enumerate_simple_design_candidates(
            intent,
            timesteps=50,
            seed_start=950,
        )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.symmetry, "C3")
        self.assertIsNotNone(candidate.design.finite_orbit_action)
        self.assertEqual(
            set(
                candidate.design.finite_orbit_action
                .stabilizer_path_transform_ids
            ),
            {
                "participant__A__link_01",
                "participant__B__link_01",
                "participant__C__link_01",
            },
        )
        self.assertEqual(
            set(
                candidate.design.finite_orbit_action
                .stabilizer_path_transform_ids.values()
            ),
            {"C3:e", "C3:r1", "C3:r2"},
        )
        self.assertEqual(candidate.physical_polymer_unit_count, 3)
        self.assertEqual(len(candidate.design.interfaces), 1)
        self.assertEqual(len(candidate.design.interfaces[0].between), 3)
        self.assertEqual(len(candidate.design.connections), 3)
        lowered = lower_user_design(candidate.design)
        self.assertEqual(len(lowered.interface_usage), 1)
        self.assertEqual(
            lowered.interface_usage[0].physical_instance_count,
            1,
        )
        instances = expand_symmetry_instances(lowered.specification)
        orbit = instances.constraint_orbits["motif_orbit"]
        self.assertEqual(len(orbit.transform_ids), 1)
        self.assertEqual(len(instances.interfaces), 2)

        progress_messages: list[str] = []
        manifest = resolve_simple_intent(
            intent,
            self.root / "pi25_three_way_resolution",
            timesteps=50,
            seed_start=950,
            top_count=1,
            progress=progress_messages.append,
        )
        self.assertTrue(
            any("materializing" in message for message in progress_messages)
        )
        self.assertTrue(
            any("enumerated 1" in message for message in progress_messages)
        )
        self.assertTrue(
            any("strict replay finished" in message for message in progress_messages)
        )
        self.assertEqual(
            manifest["resolver"],
            "rfd3_mosaic.single_supplied_hyperedge_explicit_paths_v1",
        )
        self.assertEqual(manifest["candidate_count"], 1)
        self.assertEqual(
            manifest["accepted_count"],
            1,
            manifest["ranking"],
        )
        self.assertEqual(
            manifest["selected_count"],
            1,
            manifest["ranking"],
        )
        self.assertTrue(manifest["recommended_design"])
        self.assertTrue(
            manifest["ranking"][0]["rfd3_adapter_validated"]
        )
        self.assertTrue(
            manifest["ranking"][0]["rfd3_adapter_prevalidated"]
        )
        adapter_input = Path(
            manifest["ranking"][0]["rfd3_adapter_input"]
        )
        prevalidation = prevalidate_rfd3_input(adapter_input)
        self.assertEqual(prevalidation["expected_multiplicity"], 3)
        self.assertEqual(prevalidation["chain_count"], 3)
        self.assertEqual(
            prevalidation["symmetry_action_kind"],
            "preexpanded_stabilized_asu",
        )

    def test_multi_seed_homomer_requires_path_equivalence_proof(
        self,
    ) -> None:
        payload = self._intent().model_dump(mode="json")
        payload["interface_seeds"]["second_interface"] = {
            "participants": ["C", "D"],
            "selectors": {
                "C": "C/1-2/*",
                "D": "D/1-2/*",
            },
            "use": "auto",
            "geometry": "preserve_exact",
        }
        intent = SimpleCageIntentSpec.model_validate(payload)

        with self.assertRaises((NotImplementedError, ValueError)) as context:
            enumerate_simple_design_candidates(intent)

        message = str(context.exception).lower()
        self.assertIn("homomer", message)
        self.assertTrue(
            any(word in message for word in ("equivalent", "equivalence")),
            message,
        )

    def test_resolve_cli_dispatches_simple_intent_and_prints_manifest(
        self,
    ) -> None:
        intent_path = self.root / "simple_intent.yaml"
        intent_path.write_text(
            yaml.safe_dump(
                self._intent().model_dump(mode="json", exclude_none=True),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        output_directory = self.root / "resolution"
        expected = {
            "resolver": "mock-simple-resolver",
            "candidate_count": 4,
            "accepted_count": 4,
            "selected_count": 4,
            "ranking": [],
            "manifest_path": str(
                output_directory / "resolution_manifest.json"
            ),
        }
        stdout = StringIO()

        with patch(
            "rfd3_mosaic.cli.resolve_simple_intent",
            return_value=expected,
        ) as resolver, redirect_stdout(stdout):
            main(
                [
                    "resolve",
                    str(intent_path),
                    "--output-dir",
                    str(output_directory),
                    "--format",
                    "json",
                ]
            )

        self.assertEqual(json.loads(stdout.getvalue()), expected)
        resolver.assert_called_once()
        positional, keywords = resolver.call_args
        self.assertIsInstance(positional[0], SimpleCageIntentSpec)
        self.assertEqual(positional[0].input, self.structure.resolve())
        self.assertEqual(positional[1], output_directory)
        self.assertEqual(keywords["source_path"], intent_path)
        self.assertIsNone(keywords["symmetry_ids"])
        # Omission is distinct from an explicit ``--pose-samples 1``:
        # the resolver uses the ordinary diversity preset only when this is
        # None, while an explicit one-start fast gate must remain one start.
        self.assertIsNone(keywords["pose_samples"])
        self.assertEqual(keywords["seed_start"], 0)
        self.assertEqual(keywords["top_count"], 20)
        self.assertEqual(keywords["max_candidates"], 4096)
        self.assertTrue(keywords["optimize_poses"])
        self.assertEqual(keywords["pose_optimize_top"], 4)
        self.assertEqual(keywords["pose_optimization_levels"], 3)
        self.assertEqual(keywords["pose_maximum_translation"], 12.0)
        self.assertEqual(keywords["pose_maximum_rotation_deg"], 25.0)

    def test_resolve_cli_forwards_explicit_single_pose_sample(self) -> None:
        intent_path = self.root / "explicit_pose_sample_intent.yaml"
        intent_path.write_text(
            yaml.safe_dump(
                self._intent().model_dump(mode="json", exclude_none=True),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        output_directory = self.root / "explicit_pose_sample_resolution"
        expected = {
            "resolver": "mock-simple-resolver",
            "candidate_count": 1,
            "accepted_count": 1,
            "selected_count": 1,
            "ranking": [],
            "manifest_path": str(
                output_directory / "resolution_manifest.json"
            ),
        }

        with patch(
            "rfd3_mosaic.cli.resolve_simple_intent",
            return_value=expected,
        ) as resolver, redirect_stdout(StringIO()):
            main(
                [
                    "resolve",
                    str(intent_path),
                    "--output-dir",
                    str(output_directory),
                    "--pose-samples",
                    "1",
                    "--format",
                    "json",
                ]
            )

        resolver.assert_called_once()
        self.assertEqual(resolver.call_args.kwargs["pose_samples"], 1)

    def test_resolve_cli_reports_restored_linker_lengths(self) -> None:
        intent_path = self.root / "restored_linker_intent.yaml"
        intent_path.write_text(
            yaml.safe_dump(
                self._intent().model_dump(mode="json", exclude_none=True),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        output_directory = self.root / "restored_linker_resolution"
        selected = output_directory / "selected" / "rank_0001.yaml"
        expected = {
            "resolver": "mock-simple-resolver",
            "candidate_count": 1,
            "accepted_count": 1,
            "selected_count": 1,
            "recommended_design": str(selected),
            "continuous_pose_optimization": {"enabled": True},
            "ranking": [
                {
                    "candidate_id": "candidate_000000",
                    "symmetry": "T",
                    "topology_id": "t-test",
                    "physical_polymer_unit_count": 24,
                    "rank": 1,
                    "resolved_design": str(selected),
                    "feasibility_restoration": {
                        "changed": True,
                        "linker_length_bindings": [
                            {
                                "source_link_id": "polymer_link_002",
                                "tie_group": "unit_length",
                                "selected_length": 34,
                            }
                        ],
                    },
                }
            ],
            "manifest_path": str(
                output_directory / "resolution_manifest.json"
            ),
        }
        stdout = StringIO()

        with patch(
            "rfd3_mosaic.cli.resolve_simple_intent",
            return_value=expected,
        ), redirect_stdout(stdout):
            main(
                [
                    "resolve",
                    str(intent_path),
                    "--output-dir",
                    str(output_directory),
                    "--top",
                    "1",
                ]
            )

        self.assertIn(
            "restored linker lengths: polymer_link_002=34 "
            "(tie=unit_length)",
            stdout.getvalue(),
        )

    def test_run_simple_intent_resolves_then_dispatches_strict_replay(
        self,
    ) -> None:
        intent_path = self.root / "one_command_intent.yaml"
        payload = self._intent().model_dump(mode="json", exclude_none=True)
        payload["output"] = {
            "root": str(self.root / "runs"),
            "campaign": "one-command-test",
        }
        intent_path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        resolution_directory = self.root / "one-command-resolution"
        selected = resolution_directory / "selected" / "rank_0001.yaml"
        selected.parent.mkdir(parents=True)
        selected.write_text("schema_version: 1\n", encoding="utf-8")
        result = {
            "recommended_design": str(selected),
            "manifest_path": str(
                resolution_directory / "resolution_manifest.json"
            ),
        }
        stdout = StringIO()

        with patch(
            "rfd3_mosaic.cli.resolve_simple_intent",
            return_value=result,
        ) as resolver, patch(
            "rfd3_mosaic.cli._dispatch_replayed_design"
        ) as dispatch, redirect_stdout(stdout):
            main(
                [
                    "run",
                    str(intent_path),
                    "--resolution-dir",
                    str(resolution_directory),
                    "--resolve-pose-samples",
                    "1",
                    "--resolve-timesteps",
                    "50",
                    "--profile",
                    "v100",
                    "--run-root",
                    str(self.root / "portable-runs"),
                    "--campaign",
                    "portable-campaign",
                    "--dry-run",
                ]
            )

        resolver.assert_called_once()
        self.assertEqual(resolver.call_args.args[1], resolution_directory)
        self.assertEqual(resolver.call_args.kwargs["pose_samples"], 1)
        self.assertEqual(resolver.call_args.kwargs["timesteps"], 50)
        self.assertEqual(resolver.call_args.kwargs["top_count"], 1)
        dispatch.assert_called_once_with(
            command="run",
            design_path=selected.resolve(),
            profile="v100",
            output_directory=None,
            run_root=self.root / "portable-runs",
            campaign="portable-campaign",
            dry_run=True,
        )
        self.assertIn("ordinary one-command execution", stdout.getvalue())

    def test_run_simple_intent_fails_closed_without_replay_candidate(
        self,
    ) -> None:
        intent_path = self.root / "unresolved_one_command_intent.yaml"
        payload = self._intent().model_dump(mode="json", exclude_none=True)
        payload["output"] = {
            "root": str(self.root / "runs"),
            "campaign": "one-command-test",
        }
        intent_path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        resolution_directory = self.root / "unresolved-resolution"
        manifest = resolution_directory / "resolution_manifest.json"
        stderr = StringIO()

        with patch(
            "rfd3_mosaic.cli.resolve_simple_intent",
            return_value={
                "recommended_design": None,
                "manifest_path": str(manifest),
            },
        ), patch(
            "rfd3_mosaic.cli._dispatch_replayed_design"
        ) as dispatch, redirect_stderr(stderr), self.assertRaises(SystemExit):
            main(
                [
                    "run",
                    str(intent_path),
                    "--resolution-dir",
                    str(resolution_directory),
                    "--dry-run",
                ]
            )

        dispatch.assert_not_called()
        self.assertIn("No strictly replayable design", stderr.getvalue())
        self.assertIn(str(manifest), stderr.getvalue())

    def test_resolve_cli_rejects_already_standard_public_design(self) -> None:
        design_path = self.root / "standard_design.yaml"
        design_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "name": "already-standard",
                    "input": str(self.structure),
                    "symmetry": "C3",
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        stderr = StringIO()

        with patch(
            "rfd3_mosaic.cli.resolve_simple_intent"
        ) as resolver, redirect_stderr(stderr), self.assertRaises(SystemExit):
            main(
                [
                    "resolve",
                    str(design_path),
                    "--output-dir",
                    str(self.root / "unused"),
                ]
            )

        resolver.assert_not_called()
        self.assertIn(
            "resolve expects kind: simple_cage_intent",
            stderr.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
