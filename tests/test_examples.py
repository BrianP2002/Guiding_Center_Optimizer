"""The examples run (with --fast), the code blocks of docs/usage.md run, and the published text carries no private identifiers."""
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, 'examples')


def _run(args, env_extra=None, timeout=170):
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu', **(env_extra or {})}
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, env=env, timeout=timeout)
    assert r.returncode == 0, f'{args} failed\n--- stdout\n{r.stdout[-2500:]}\n--- stderr\n{r.stderr[-3500:]}'
    return r.stdout


CASES = [
    ('01_quickstart_lorenz.py', ['--fast'], 'd<z>/d(rho)'),
    ('02_ensemble_and_reliability.py', ['--fast'], 'nus consistency'),
    ('03_custom_system_lorenz96.py', ['--fast'], 'combined standard errors'),
    ('04_guiding_center_sea.py', ['--fast'], 'results of the studies for draw 835'),
    ('04_guiding_center_sea.py', ['--fast', '--draw', '581', '--fd'], 'finite differences in eps_t'),
    ('05_optimize_lorenz.py', ['--fast'], 'finite-difference gradients'),
]


@pytest.mark.parametrize('name,extra,phrase', CASES, ids=[f'{c[0]} {" ".join(c[1])}' for c in CASES])
def test_example_runs(name, extra, phrase):
    out = _run([os.path.join('examples', name), *extra])
    assert phrase in out


def test_cluster_ensemble_template(tmp_path):
    d = os.path.join('examples', '06_cluster_ensemble')
    _run([os.path.join(d, 'member.py'), '--outdir', str(tmp_path), '--index', '0', '--fast'])
    _run([os.path.join(d, 'member.py'), '--outdir', str(tmp_path), '--fast'], env_extra={'SLURM_ARRAY_TASK_ID': '1'})   # index from the environment
    assert sorted(os.listdir(tmp_path)) == ['run_0000.npz', 'run_0001.npz']
    out = _run([os.path.join(d, 'collect.py'), str(tmp_path), '--min-lyap-time', '3'])
    assert 'NILSS ensemble: 2 usable runs of 2' in out and 'reliability report' in out


def _python_blocks(path):
    text = open(path, encoding='utf-8').read()
    blocks = re.findall(r'^```python\n(.*?)^```', text, flags=re.S | re.M)
    return [b for b in blocks if '# not tested' not in b]


def test_usage_snippets_run(tmp_path):
    """Every ```python block of docs/usage.md (except those marked '# not tested') runs, in order, in one namespace."""
    blocks = _python_blocks(os.path.join(ROOT, 'docs', 'usage.md'))
    assert len(blocks) >= 8
    code = '\n\n'.join(blocks)
    script = tmp_path / 'usage_snippets.py'
    script.write_text(code, encoding='utf-8')
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu'}
    r = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, env=env, timeout=170)
    assert r.returncode == 0, f'--- stdout\n{r.stdout[-2500:]}\n--- stderr\n{r.stderr[-3500:]}'


PRIVATE = ['e32695', 'zwm4367', '/gpfs', '/projects/p52907', 'p52907', 'wisc.edu', '163.com', 'gmail.com']


def _published_files():
    for sub in ('examples', 'docs'):
        for dirpath, _, files in os.walk(os.path.join(ROOT, sub)):
            for f in files:
                if f.endswith(('.py', '.md', '.sbatch', '.sh', '.txt', '.cff', '.json')):
                    yield os.path.join(dirpath, f)
    for f in ('CITATION.cff', 'CHANGELOG.md'):
        if os.path.exists(os.path.join(ROOT, f)):
            yield os.path.join(ROOT, f)


def test_no_private_identifiers_in_examples_and_docs():
    bad = []
    for path in _published_files():
        text = open(path, encoding='utf-8', errors='ignore').read()
        bad += [f'{os.path.relpath(path, ROOT)}: {s}' for s in PRIVATE if s in text]
    assert not bad, bad
