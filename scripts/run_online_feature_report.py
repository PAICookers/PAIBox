from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = Path("tests/paiir/pipeline/test_online_feature_matrix.py")
REPORT_DIR = ROOT / "tests" / "paiir" / "pipeline" / "debug" / "online_feature_report"
JUNIT_XML = REPORT_DIR / "junit.xml"
COVERAGE_XML = REPORT_DIR / "coverage.xml"
RAW_OUTPUT = REPORT_DIR / "pytest_output.txt"
MARKDOWN_REPORT = REPORT_DIR / "report.md"

FEATURE_TITLES = {
    "TestOnlineStageExpansionReport": "训练阶段展开与更新绑定",
    "TestOnlineBoundarySemanticsReport": "图入口边界语义",
    "TestOnlineExportBridgeReport": "backendv2 在线桥接导出",
    "TestOnlineRuntimeMappingReport": "输入映射与运行时编解码",
    "TestOnlineValidationNetworkReport": "统一验证网络编译",
}

KEY_COVERAGE_FILES = (
    "paiir/lowering/converter.py",
    "paiir/lowering/online_training.py",
    "paiir/pipeline/online.py",
    "backendv2/core_config.py",
    "backendv2/coreplacement.py",
    "backendv2/mapper.py",
    "backendv2/export/proto.py",
    "backendv2/proto/runtime.py",
)


@dataclass(frozen=True)
class TestCaseResult:
    feature_key: str
    test_name: str
    case_name: str
    nodeid: str
    status: str
    duration_s: float


def _run_pytest() -> subprocess.CompletedProcess[str]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(TEST_FILE),
        "-s",
        "-vv",
        "-p",
        "no:cacheprovider",
        f"--junitxml={JUNIT_XML}",
        f"--cov-report=xml:{COVERAGE_XML}",
    ]
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def _write_console(text: str) -> None:
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


def _load_junit_results() -> tuple[list[TestCaseResult], dict[str, float | int]]:
    tree = ET.parse(JUNIT_XML)
    root = tree.getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError("cannot find testsuite in junit xml")

    summary = {
        "tests": int(suite.attrib.get("tests", 0)),
        "failures": int(suite.attrib.get("failures", 0)),
        "errors": int(suite.attrib.get("errors", 0)),
        "skipped": int(suite.attrib.get("skipped", 0)),
        "time": float(suite.attrib.get("time", 0.0)),
    }

    results: list[TestCaseResult] = []
    for testcase in suite.iter("testcase"):
        classname = testcase.attrib.get("classname", "")
        test_name = testcase.attrib["name"]
        parts = classname.split(".")
        if not parts:
            continue
        feature_key = parts[-1]
        module_path = "/".join(parts[:-1]) + ".py"
        nodeid = f"{module_path}::{feature_key}::{test_name}"
        case_name = test_name[test_name.index("[") + 1 : -1] if "[" in test_name else test_name

        if testcase.find("failure") is not None:
            status = "failed"
        elif testcase.find("error") is not None:
            status = "error"
        elif testcase.find("skipped") is not None:
            status = "skipped"
        else:
            status = "passed"

        results.append(
            TestCaseResult(
                feature_key=feature_key,
                test_name=test_name,
                case_name=case_name,
                nodeid=nodeid,
                status=status,
                duration_s=float(testcase.attrib.get("time", 0.0)),
            )
        )

    return results, summary


def _parse_stdout_blocks(stdout: str) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = defaultdict(list)
    current_nodeid: str | None = None
    ignored_prefixes = (
        "- generated xml file:",
        "Coverage XML written to file",
    )

    for raw_line in stdout.splitlines():
        line = raw_line.rstrip()
        if line.startswith(str(TEST_FILE).replace("\\", "/") + "::"):
            current_nodeid = line.split(" ", 1)[0]
            continue

        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("="):
            current_nodeid = None
            continue
        if current_nodeid is None:
            continue
        if stripped in {"PASSED", "FAILED", "ERROR", "SKIPPED"}:
            continue
        if stripped.startswith(ignored_prefixes):
            continue

        blocks[current_nodeid].append(stripped)

    return blocks


def _load_coverage_rows() -> list[tuple[str, float, float]]:
    tree = ET.parse(COVERAGE_XML)
    root = tree.getroot()
    rows: list[tuple[str, float, float]] = []
    wanted = set(KEY_COVERAGE_FILES)

    for class_elem in root.iter("class"):
        filename = class_elem.attrib.get("filename", "").replace("\\", "/")
        if filename not in wanted:
            continue
        rows.append(
            (
                f"paibox/{filename}",
                float(class_elem.attrib.get("line-rate", 0.0)),
                float(class_elem.attrib.get("branch-rate", 0.0)),
            )
        )

    rows.sort(key=lambda item: item[0])
    return rows


def _feature_title(feature_key: str) -> str:
    return FEATURE_TITLES.get(feature_key, feature_key)


def _ordered_feature_keys(feature_groups: dict[str, list[TestCaseResult]]) -> list[str]:
    ordered_keys = [key for key in FEATURE_TITLES if key in feature_groups]
    ordered_keys.extend(
        key for key in sorted(feature_groups) if key not in FEATURE_TITLES
    )
    return ordered_keys


def _write_markdown_report(
    results: list[TestCaseResult],
    summary: dict[str, float | int],
    stdout_blocks: dict[str, list[str]],
    coverage_rows: list[tuple[str, float, float]],
    command: str,
) -> None:
    feature_groups: dict[str, list[TestCaseResult]] = defaultdict(list)
    for result in results:
        feature_groups[result.feature_key].append(result)

    lines: list[str] = []
    lines.append("# 在线核功能测试报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 测试文件：`{TEST_FILE.as_posix()}`")
    lines.append(f"- 运行命令：`{command}`")
    lines.append("- 说明：以下用例覆盖真实的在线编译、导出、映射与运行时辅助路径，不是单纯展示输出。")
    lines.append("")
    lines.append("## 总体结果")
    lines.append("")
    lines.append(f"- 用例总数：{summary['tests']}")
    lines.append(f"- 通过：{summary['tests'] - summary['failures'] - summary['errors'] - summary['skipped']}")
    lines.append(f"- 失败：{summary['failures']}")
    lines.append(f"- 错误：{summary['errors']}")
    lines.append(f"- 跳过：{summary['skipped']}")
    lines.append(f"- 总耗时：{summary['time']:.2f}s")
    lines.append("")
    lines.append("## 功能覆盖矩阵")
    lines.append("")
    lines.append("| 功能项 | 样例数 | 通过数 |")
    lines.append("| --- | ---: | ---: |")
    for feature_key in _ordered_feature_keys(feature_groups):
        items = feature_groups[feature_key]
        passed = sum(1 for item in items if item.status == "passed")
        lines.append(
            f"| {_feature_title(feature_key)} | {len(items)} | {passed} |"
        )
    lines.append("")
    lines.append("## 详细结果")
    lines.append("")
    for feature_key in _ordered_feature_keys(feature_groups):
        items = feature_groups[feature_key]
        lines.append(f"### {_feature_title(feature_key)}")
        lines.append("")
        for item in items:
            lines.append(f"#### {item.case_name}")
            lines.append("")
            lines.append(f"- 状态：`{item.status}`")
            lines.append(f"- 耗时：`{item.duration_s:.3f}s`")
            detail_lines = stdout_blocks.get(item.nodeid, [])
            if detail_lines:
                lines.append("- 结果摘要：")
                for detail in detail_lines:
                    lines.append(f"  - {detail}")
            lines.append("")
    lines.append("## 关键文件覆盖率")
    lines.append("")
    lines.append("| 文件 | 行覆盖率 | 分支覆盖率 |")
    lines.append("| --- | ---: | ---: |")
    for filename, line_rate, branch_rate in coverage_rows:
        lines.append(
            f"| `{filename}` | {line_rate * 100:.1f}% | {branch_rate * 100:.1f}% |"
        )
    lines.append("")
    lines.append("## 原始 pytest 输出")
    lines.append("")
    lines.append("```text")
    lines.append(RAW_OUTPUT.read_text(encoding="utf-8"))
    lines.append("```")
    lines.append("")

    MARKDOWN_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    result = _run_pytest()
    combined_output = result.stdout
    if result.stderr:
        combined_output = f"{combined_output}\n{result.stderr}".strip()

    RAW_OUTPUT.write_text(combined_output, encoding="utf-8")
    _write_console(combined_output + "\n")

    if result.returncode != 0:
        sys.stderr.write(f"\npytest failed with exit code {result.returncode}\n")
        return result.returncode

    junit_results, summary = _load_junit_results()
    stdout_blocks = _parse_stdout_blocks(result.stdout)
    coverage_rows = _load_coverage_rows()
    command = (
        f"{sys.executable} -m pytest {TEST_FILE.as_posix()} -s -vv "
        f"-p no:cacheprovider --junitxml={JUNIT_XML.as_posix()} "
        f"--cov-report=xml:{COVERAGE_XML.as_posix()}"
    )
    _write_markdown_report(
        junit_results,
        summary,
        stdout_blocks,
        coverage_rows,
        command,
    )
    _write_console(f"\nMarkdown report written to: {MARKDOWN_REPORT}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
