import unittest

from rfd3_mosaic.rfd3_scaffold_audit import _effective_chain_rg_limit


class ScaffoldAuditPolicyTestCase(unittest.TestCase):
    def test_default_limit_is_preserved_for_compact_fixed_geometry(self) -> None:
        self.assertEqual(
            _effective_chain_rg_limit(
                explicit_limit=None,
                fixed_geometry_floor=18.0,
            ),
            25.0,
        )

    def test_fixed_geometry_sets_an_automatic_lower_bound(self) -> None:
        self.assertEqual(
            _effective_chain_rg_limit(
                explicit_limit=None,
                fixed_geometry_floor=25.327,
            ),
            27.327,
        )

    def test_explicit_limit_remains_authoritative(self) -> None:
        self.assertEqual(
            _effective_chain_rg_limit(
                explicit_limit=24.0,
                fixed_geometry_floor=40.0,
            ),
            24.0,
        )

    def test_invalid_limits_fail_closed(self) -> None:
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _effective_chain_rg_limit(
                        explicit_limit=None,
                        fixed_geometry_floor=value,
                    )


if __name__ == "__main__":
    unittest.main()
