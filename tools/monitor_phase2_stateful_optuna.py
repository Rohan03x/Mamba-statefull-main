#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path


def _default_pid_file(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "optuna_studies" / "phase2_stateful_default_no_flags.pid"


def _latest_glob(repo_root: Path, pattern: str) -> Path | None:
    candidates = sorted((repo_root).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cmdline_for_pid(pid: int) -> str:
    try:
        out = subprocess.check_output(["ps", "-p", str(pid), "-o", "cmd="], text=True)
        return (out or "").strip()
    except Exception:
        return ""


def _resolve_runner_pid(pid: int) -> int:
    """Best-effort: map a wrapper shell PID to the actual Python runner PID."""
    cmd = _cmdline_for_pid(pid)
    if "run_stage_b_stateful_phase2.py" in cmd:
        return pid

    # Prefer direct children.
    try:
        out = subprocess.check_output(
            ["pgrep", "-P", str(pid), "-f", "tools/run_stage_b_stateful_phase2.py"],
            text=True,
        )
        cands = [int(x) for x in out.split() if x.strip().isdigit()]
        if cands:
            return cands[0]
    except Exception:
        pass

    # Fallback: any matching process.
    try:
        out = subprocess.check_output(["pgrep", "-f", "tools/run_stage_b_stateful_phase2.py"], text=True)
        cands = [int(x) for x in out.split() if x.strip().isdigit()]
        if cands:
            return cands[-1]
    except Exception:
        pass

    return pid


def _read_tail(path: Path, n_lines: int) -> list[str]:
    # Simple, safe tail (log lines are not expected to be huge per line).
    try:
        data = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return []
    if n_lines <= 0:
        return []
    return data[-n_lines:]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Monitor Phase2 stateful Optuna background run")
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=_default_pid_file(repo_root),
        help="PID file for the background run",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Optuna sqlite db file (defaults to latest artifacts/optuna_studies/phase2_stateful_*.db)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Log file (defaults to latest artifacts/optuna_studies/phase2_stateful_default_no_flags_*.log)",
    )
    parser.add_argument("--tail", type=int, default=30, help="How many log lines to print")

    args = parser.parse_args()

    # PID status
    pid = None
    if args.pid_file and args.pid_file.exists():
        try:
            pid = int(args.pid_file.read_text().strip())
        except Exception:
            pid = None

    if pid is None:
        print(f"PID: (missing/invalid) file={args.pid_file}")
    else:
        resolved = _resolve_runner_pid(pid)
        if resolved != pid:
            print(
                f"PID: {pid} alive={_is_pid_alive(pid)} file={args.pid_file} (wrapper; runner_pid={resolved})"
            )
        else:
            print(f"PID: {pid} alive={_is_pid_alive(pid)} file={args.pid_file}")

    # Resolve DB + log
    db_path = args.db
    if db_path is None:
        db_path = _latest_glob(repo_root, "artifacts/optuna_studies/phase2_stateful_*.db")

    log_path = args.log
    if log_path is None:
        log_path = _latest_glob(repo_root, "artifacts/optuna_studies/phase2_stateful_default_no_flags_*.log")

    print(f"DB:  {db_path if db_path else '(not found)'}")
    print(f"LOG: {log_path if log_path else '(not found)'}")

    # Optuna DB status
    if db_path and db_path.exists():
        try:
            import optuna
            from optuna.trial import TrialState

            import datetime as dt

            storage = optuna.storages.RDBStorage(url=f"sqlite:///{db_path}")
            studies = storage.get_all_studies()
            if not studies:
                print("Optuna: no studies found yet")
            else:
                # Usually only one study for these default runs.
                for s in studies:
                    study = optuna.load_study(study_name=s.study_name, storage=storage)
                    trials = study.get_trials(deepcopy=False)
                    counts: dict[TrialState, int] = {}
                    for t in trials:
                        counts[t.state] = counts.get(t.state, 0) + 1

                    def c(state: TrialState) -> int:
                        return counts.get(state, 0)

                    print(f"Study: {s.study_name}")
                    print(f"Direction: {study.direction}")
                    print(
                        "Trials: "
                        f"total={len(trials)} "
                        f"RUNNING={c(TrialState.RUNNING)} "
                        f"COMPLETE={c(TrialState.COMPLETE)} "
                        f"PRUNED={c(TrialState.PRUNED)} "
                        f"FAIL={c(TrialState.FAIL)}"
                    )

                    # Timing/rate (use naive wall-clock; this environment logs local times)
                    now = dt.datetime.now()
                    starts = [t.datetime_start for t in trials if t.datetime_start is not None]
                    starts_naive = [st.replace(tzinfo=None) if getattr(st, "tzinfo", None) else st for st in starts]
                    if starts_naive:
                        t0 = min(starts_naive)
                        elapsed = now - t0
                        hours = max(elapsed.total_seconds() / 3600.0, 1e-9)
                        started_rate = len(trials) / hours
                        completed_rate = c(TrialState.COMPLETE) / hours
                        print(f"Wall: since_first_trial_start={str(elapsed).split('.')[0]}")
                        print(f"Rate: started/hr={started_rate:.3f} completed/hr={completed_rate:.3f}")

                    # Latest trial status
                    if trials:
                        last = trials[-1]
                        ds = last.datetime_start
                        dc = last.datetime_complete
                        ds_n = ds.replace(tzinfo=None) if ds and getattr(ds, "tzinfo", None) else ds
                        dc_n = dc.replace(tzinfo=None) if dc and getattr(dc, "tzinfo", None) else dc
                        print(
                            f"Last: trial={last.number} state={last.state} "
                            f"params={len(last.params)} value={last.value}"
                        )
                        if ds_n and not dc_n:
                            print(f"Last: start={ds_n} elapsed={str(now - ds_n).split('.')[0]}")
                        elif ds_n and dc_n:
                            print(f"Last: start={ds_n} duration={str(dc_n - ds_n).split('.')[0]}")

                    if c(TrialState.COMPLETE) > 0:
                        print(f"Best: value={study.best_value} params={len(study.best_params)}")
        except Exception as e:
            print(f"Optuna: failed to query db ({type(e).__name__}): {e}")

    # Log tail
    if log_path and log_path.exists() and args.tail > 0:
        print("\n--- log tail ---")
        for line in _read_tail(log_path, args.tail):
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
