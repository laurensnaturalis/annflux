import os
import re

from annflux.tools.mixed import get_version


def extract_test_results(input_path: str, out_path:str):
    content = open(input_path, "r").read()
    summary_match = re.search(r"=+ (.*?) in [\d.]+s \((\d+:\d+:\d+)\) =+", content)
    if not summary_match:
        raise RuntimeError(f"Unknown test output format {open(input_path).readlines()[-1]}")
    summary_str = summary_match.group(1)
    time_taken = summary_match.group(2)

    counts = {key: int(n) for n, key in re.findall(r"(\d+) (failed|passed|skipped|warning)", summary_str)}
    failed = counts.get("failed", 0)
    passed = counts.get("passed", 0)
    skipped = counts.get("skipped", 0)
    warnings = counts.get("warning", 0)

    # split time into hours, minutes, and seconds
    hours, minutes, seconds = map(int, time_taken.split(':'))

    passed_condition = hours == 0 and minutes < 5 and failed == 0

    result = {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "warnings": warnings,
        "time": time_taken,
        "passed_condition": passed_condition
    }
    with open(out_path, 'w') as file:
        file.write("| Metric | Value |\n")
        file.write("|--------|-------|\n")
        file.write(f"| Passed | {result['passed']} |\n")
        file.write(f"| Failed | {result['failed']} |\n")
        file.write(f"| Skipped | {result['skipped']} |\n")
        file.write(f"| Warnings | {result['warnings']} |\n")
        file.write(f"| Time | {result['time']} |\n")
        file.write(f"| Tests successful (Failed==0 and Time < 5:00) | {'✅' if result['passed_condition'] else '❌'} |\n")
    return result


def run_coverage_func():
    os.makedirs("../release_assets", exist_ok=True)
    os.system(
        f"ruff check . > ../release_assets/ruff_check_{get_version()['version']}.txt"
    )
    os.environ["TEST_TYPE"] = "pytest"
    if os.path.exists(".coverage"):
        print("deleting existing .coverage")
        os.remove(".coverage")
    os.system('find . -name "*.pyc" -delete')
    os.system(f"pytest annflux/tests --cov -s -v > ../release_assets/tests_{get_version()['version']}.txt")
    os.system(f"coverage report -m --ignore-errors > ../release_assets/coverage_{get_version()['version']}.txt")
    extract_test_results(f"../release_assets/tests_{get_version()['version']}.txt", f"../release_assets/test_report_{get_version()['version']}.md")


if __name__ == '__main__':
    run_coverage_func()
