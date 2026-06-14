"""Regenerate the consolidated REPORT.md from whatever artifacts exist.

  python -m revision.report --exp-dir experiments_revision
"""

import argparse

from .runlog import write_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="experiments_revision")
    a = ap.parse_args()
    write_report(a.exp_dir)


if __name__ == "__main__":
    main()
