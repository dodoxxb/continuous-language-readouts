"""Run portable CPU checks on retained measurements; no neural model is run."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'evidence'
CASES = {
    'stability': ('calibration/scripts/verify.py', None),
    'fixed-mdp': ('calibration/scripts/verify_fixed_mdp.py', None),
    'fixed-test': ('confirmatory/verify.py', None),
    'langflow-budget': ('confirmatory/verify_followup.py', None),
    'prospective': ('prospective/verify.py', None),
    'aurora': ('@check_aurora.py', None),
    'new-interfaces': ('@check_new_interfaces.py', None),
}


def run(case: str, evidence: Path, env: dict[str, str]) -> None:
    script, root_arg = CASES[case]
    target = ROOT / 'tools' / script[1:] if script.startswith('@') else evidence / script
    command = [sys.executable, '-B', str(target)]
    if root_arg:
        command += ['--root', str(evidence / root_arg)]
    print(f'\n[{case}]', flush=True)
    subprocess.run(command, cwd=evidence, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', choices=list(CASES), help='Run only this case; may be repeated.')
    parser.add_argument('--list', action='store_true', help='List the available cases and exit.')
    args = parser.parse_args()
    if args.list:
        print('\n'.join(CASES))
        return
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    for case in args.case or CASES:
        if case in {'stability', 'fixed-mdp'}:
            # Historical verifier(s) write a result file. Keep the checkout clean.
            with tempfile.TemporaryDirectory(prefix='readout_calibration_') as tmp:
                temp_e = Path(tmp)
                shutil.copytree(E / 'calibration', temp_e / 'calibration')
                run(case, temp_e, env)
        else:
            run(case, E, env)
    print('\nRequested evidence checks passed. These checks do not rerun GPU inference or model training.')


if __name__ == '__main__':
    main()
