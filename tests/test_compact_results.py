"""Compact publication preserves original measurements and source evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class CompactResultsTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('glval.compact'), 'Compact result export is missing')
        from glval import compact
        self.compact = compact

    def test_runtime_summary_keeps_binding_and_exact_progression_without_changing_source(self):
        raw = {'status': 'completed', 'hook_sha256': 'abc', 'last_bucket_calls': 122720,
               'refresh_coverage': 'not established', 'timing_refresh_steps': list(range(1001, 122721))}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'runtime-rank-0.json'
            source.write_text(json.dumps(raw))
            before = source.read_bytes()
            target = self.compact.export_file(source)
            self.assertEqual(before, source.read_bytes())
            saved = json.loads(target.read_text())
        self.assertEqual(saved['measurements']['hook_sha256'], 'abc')
        self.assertEqual(saved['measurements']['refresh_coverage'], 'not established')
        self.assertNotIn('timing_refresh_steps', saved['measurements'])
        steps = saved['omitted_sequences']['timing_refresh_steps']
        self.assertEqual((steps['count'], steps['first'], steps['last'], steps['step']), (121720, 1001, 122720, 1))
        self.assertEqual(len(saved['source_sha256']), 64)
        self.assertEqual(len(steps['sha256']), 64)
        self.assertLess(len(json.dumps(saved)), 2000)

    def test_timing_retains_recorded_mean_budget_and_per_rank_statistics(self):
        raw = {'status': 'completed', 'mean_iteration_seconds': .99, 'measured_steps': 100,
               'rank_elapsed_seconds': [99, 98], 'first_measured_update': 102,
               'actual_parameters': {'example_configuration': list(range(100))},
               'rank_iteration_seconds': [[.2] * 100, [.3] * 100], 'rank_losses': [[4.] * 100, [5.] * 100]}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'all_results.json'
            source.write_text(json.dumps(raw))
            target = self.compact.export_file(source)
            saved = json.loads(target.read_text())
        self.assertEqual(target.name, 'timing-summary.json')
        self.assertEqual(saved['measurements']['mean_iteration_seconds'], .99)
        self.assertEqual(saved['measurements']['measured_steps'], 100)
        self.assertIn('actual_parameters', saved['measurements'])
        self.assertEqual(saved['measurements']['actual_parameters'], raw['actual_parameters'])
        self.assertEqual(saved['measurements']['first_measured_update'], 102)
        stats = saved['omitted_sequences']['rank_iteration_seconds[0]']
        self.assertEqual((stats['count'], stats['mean'], stats['median'], stats['p95']), (100, .2, .2, .2))
        self.assertNotIn('rank_iteration_seconds', saved['measurements'])
        self.assertIn('not a new verification', saved['limitations'][0])

    def test_explicit_new_run_exports_ignored_evidence_without_staging_raw_files(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            output = project / 'runs/new/jobs/method/output'
            output.mkdir(parents=True)
            source = output / 'runtime-rank-0.json'
            source.write_text(json.dumps({'method_id': 'dense', 'timing_refresh_steps': None}))
            with patch('glval.artifacts.ROOT', project):
                try:
                    exported = self.compact.export_tracked(root=project, runs=['runs/new'])
                except TypeError:
                    self.fail('Explicit run selection must support ignored new evidence')
            self.assertEqual(len(exported['records']), 1)
            self.assertTrue(source.exists())
            self.assertTrue((output / 'runtime-summary-rank-0.json').exists())

    def test_short_timing_windows_also_summarize_step_arrays_but_preserve_unknowns(self):
        fields = ('rank_iteration_seconds', 'rank_losses', 'rank_bucket_counts',
                  'rank_bucket_bytes', 'rank_hook_step_seconds')
        raw = {'mean_iteration_seconds': .5, 'rank_elapsed_seconds': [1.5, 3.],
               **{key: [[1., 2., 3.], [2., 3., 4.]] for key in fields}}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'all_results.json'
            source.write_text(json.dumps(raw))
            saved = json.loads(self.compact.export_file(source).read_text())
            for key in fields:
                self.assertNotIn(key, saved['measurements'])
                self.assertEqual(saved['omitted_sequences'][key + '[0]']['count'], 3)
            self.assertEqual(saved['measurements']['rank_elapsed_seconds'], [1.5, 3.])
            raw['rank_iteration_seconds'] = None
            source.write_text(json.dumps(raw))
            saved = json.loads(self.compact.export_file(source).read_text())
            self.assertIsNone(saved['measurements']['rank_iteration_seconds'])

    def test_unknown_refresh_information_stays_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'runtime-rank-0.json'
            source.write_text(json.dumps({'method_id': 'dense', 'timing_refresh_steps': None}))
            try:
                target = self.compact.export_file(source)
            except TypeError:
                self.fail('Null refresh evidence must remain unknown')
            saved = json.loads(target.read_text())
            self.assertIsNone(saved['measurements']['timing_refresh_steps'])
            self.assertNotIn('timing_refresh_steps', saved['omitted_sequences'])

    def test_saved_artifacts_reads_published_timing_without_raw_observations(self):
        from unittest.mock import patch
        from glval import results
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            run = project / 'runs/example'
            output = run / 'jobs/method/output'
            output.mkdir(parents=True)
            source = output / 'all_results.json'
            source.write_text(json.dumps({'status': 'completed', 'mean_iteration_seconds': .5,
                                          'rank_iteration_seconds': [[.5] * 100]}))
            self.compact.export_file(source)
            source.unlink()
            with patch('glval.artifacts.ROOT', project):
                saved = results.saved_results(run)
            self.assertEqual(len(saved['files']), 1)
            self.assertEqual(saved['files'][0]['metrics']['measurements.mean_iteration_seconds'], .5)
            self.assertEqual(saved['status'], 'saved_results')
            self.assertFalse((run / 'report.json').exists())

    def test_quality_results_are_not_replaced_and_irregular_steps_are_not_claimed_regular(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'all_results.json'
            source.write_text('{"final_eval_loss":3.0922343272218806,"status":"completed"}')
            before = source.read_bytes()
            self.assertIsNone(self.compact.export_file(source))
            self.assertEqual(source.read_bytes(), before)
        summary = self.compact.sequence_summary([1, 3, 8])
        self.assertNotIn('step', summary)


if __name__ == '__main__':
    unittest.main()
