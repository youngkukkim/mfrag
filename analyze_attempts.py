import argparse
import os
import pandas as pd


def resolve_attempts_path(path):
    if path.endswith('_attempts.csv'):
        return path
    base, ext = os.path.splitext(path)
    candidate = base + '_attempts.csv'
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f'Could not find attempts file for: {path}')


def pct(num, den):
    if den == 0:
        return 0.0
    return 100.0 * num / den


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str,
                        help='Path to results CSV or *_attempts.csv file')
    args = parser.parse_args()

    attempts_path = resolve_attempts_path(args.file_path)
    df = pd.read_csv(attempts_path)

    total = len(df)
    success_df = df[df['attempt_result'] == 'success']
    failure_df = df[df['attempt_result'] == 'failure']

    print(f'Attempts file:\t{attempts_path}')
    print(f'Total attempts:\t{total}')
    print(f'Successes:\t{len(success_df)} ({pct(len(success_df), total):.2f}%)')
    print(f'Failures:\t{len(failure_df)} ({pct(len(failure_df), total):.2f}%)')

    if total == 0:
        return

    print('\nFailure cases')
    if len(failure_df) == 0:
        print('No failures')
    else:
        failure_counts = failure_df['failure_case'].fillna('').replace('', 'unspecified').value_counts()
        for key, value in failure_counts.items():
            print(f'{key}\t{value}\t({pct(value, total):.2f}% of all, {pct(value, len(failure_df)):.2f}% of failures)')

    print('\nResult types')
    result_type_counts = df['result_type'].fillna('').replace('', 'unspecified').value_counts()
    for key, value in result_type_counts.items():
        print(f'{key}\t{value}\t({pct(value, total):.2f}%)')

    print('\nTerminal triggers')
    trigger_counts = df['terminal_trigger'].fillna('').replace('', 'none').value_counts()
    for key, value in trigger_counts.items():
        print(f'{key}\t{value}\t({pct(value, total):.2f}%)')

    print('\nReplay storage')
    replay_counts = df['replay_stored'].value_counts(dropna=False)
    for key, value in replay_counts.items():
        print(f'{key}\t{value}\t({pct(value, total):.2f}%)')

    print('\nTerminal reset')
    reset_counts = df['terminal_reset'].value_counts(dropna=False)
    for key, value in reset_counts.items():
        print(f'{key}\t{value}\t({pct(value, total):.2f}%)')

    print('\nAverages')
    cols = ['atom_count_before', 'atom_count_after', 'att_count_before', 'att_count_after', 'reward_step']
    for col in cols:
        if col in df:
            print(f'{col}\t{df[col].mean():.4f}')


if __name__ == '__main__':
    main()
