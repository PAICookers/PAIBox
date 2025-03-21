<div align="center">

# 高效编写测试项目指南

</div>

## 环境准备

PAIBox使用 `pytest` 工具组织测试项目与运行。若使用Poetry：

```bash
poetry install --with test
```

若使用conda等，则手动安装如下依赖至Python虚拟环境：

```toml
python = "^3.9"
pytest = "^8.0.0"
```

## 常用测试夹具

几个常用的与测试环境相关的夹具介绍。可直接在 `tests` 目录下的测试项目中使用这些夹具。

1. 指定测试项目的文件输出目录，例如，输出调试日志等信息。该夹具确保创建一个目录，并返回。若目录已存在，则清空目录（可选）。

   ```python
   import pytest
   from pathlib import Path

   @pytest.fixture(scope="module")
   def ensure_dump_dir():
       p = Path(__file__).parent / "debug"

       if not p.is_dir():
           p.mkdir(parents=True, exist_ok=True)
       else: # Optional if you want to clean up the directory
           for f in p.iterdir():
               f.unlink()

       yield p

   @pytest.fixture(scope="module")
   def ensure_dump_dir_and_clean():
       ...
       yield p

       # Clean up
       for f in p.iterdir():
           f.unlink()

   # In your test function at `test_items.py`, use it as follows:
   def test_foo(ensure_dump_dir):
       ...
   ```

2. 清除全局 `PAIBoxObject` 对象名字字典。该夹具在每次测试后，清除全局名字字典，从而避免命名冲突。需要注意的是，`autouse=True` 表示该夹具在每个测试函数执行前自动执行，无论测试函数是否需要。

   ```python
   import pytest
   from paibox.generic import clear_name_cache

   @pytest.fixture(autouse=True)
   def clean_name_dict():
       yield
       clear_name_cache(ignore_warn=True)
   ```

3. 需要测试文件写入，但不关心具体的目录与文件内容。该夹具将创建与系统无关的临时目录，且整个测试将在该目录下进行，测试后，临时目录自动被销毁，切回原来的目录。

   ```python
   import pytest
   import os
   import tempfile

   @pytest.fixture
   def cleandir():
       with tempfile.TemporaryDirectory() as newpath:
           old_cwd = os.getcwd()
           os.chdir(newpath)
           yield
           os.chdir(old_cwd)

   # In your test function at `test_items.py`, use it as follows:
   @pytest.mark.usefixtures("cleandir")
   def test_foo():
       ...
   ```

4. 测试后需要将某些全局配置恢复至默认值。该夹具在每次测试后，重置 `BACKEND_CONFIG` 与 `SynSys.CFLAG_ENABLE_WP_OPTIMIZATION` 为默认值。该夹具将自动执行。

   ```python
   def _reset_context() -> None:
   clear_name_cache(ignore_warn=True)
   pb.FRONTEND_ENV["t"] = 0
   pb.BACKEND_CONFIG.set_default()
   # To avoid overlapping with multi-chip coordinates
   pb.BACKEND_CONFIG.output_chip_addr = (9, 9)
   SynSys.CFLAG_ENABLE_WP_OPTIMIZATION = True

   @pytest.fixture(autouse=True)
   def context_reset():
       """Reset the context after each test automatically."""
       _reset_context()
       yield
       _reset_context()
   ```

5. 测试代码运行时间统计。该夹具将测量测试项目的运行时间，并打印至控制台。运行pytest时需添加 `-s` 参数以禁用输出捕获。

   ```python
   @pytest.fixture
   def perf_fixture(request):
       with measure_time(f"{request.node.name}"):
           yield

   def test_case1(perf_fixture):
       func1(...)
       func2(...)
   ```

   或者，亦可对测试项目的**部分代码**进行计时，使用上下文环境 `with`

   ```python
   from .utils import measure_time

   def test_case2():
       with measure_time("test case2"):
           func1(...)
           func2(...)
   ```

6. 固定种子的随机数生成器。该夹具返回一个固定的随机数生成器，通过该生成器生成的随机数可复现。

   ```python
   @pytest.fixture
   def fixed_rng() -> np.random.Generator:
       return np.random.default_rng(42)

   def test_foo(fixed_rng):
       fixed_rng.random(...)
   ```

## 日志系统

`v1.2.1` 引入了日志系统，用于开发过程中记录开发人员所关注的组件的运行情况。

在 [`_logging/registrations.py`](../paibox/_logging/registrations.py) 中通过 `paibox._logging.register_log` 添加一个新的日志：

```python
register_log("paibox", "paibox")
```

通过 `paibox._logging.register_artifact` 添加一个新的针对组件的日志。例如，添加一个新的日志用于记录 `Mapper` 对象内 `build_core_blocks` 函数的运行情况：

```python
register_artifact("build_core_blocks")
```

无论在何处对组件进行日志记录，通过调用 `paibox._logging.get_artifact_logger(__name__, <artifact_name>)` 获取日志记录器，而非标准的日志实现方式。例如，获取上述针对 `build_core_blocks` 函数的日志记录器：

```python
# In backend/mapper.py
from paibox import _logging
log = _logging.get_artifact_logger(__name__, "build_core_blocks")

class Mapper:
    ...
    def build_core_blocks(self):
        log.info("hi")
```

在测试函数中，通过 `paibox._logging.set_logs` 设置更详细日志级别；或者仅开启关注的组件的日志记录。使用 `DEFAULT_LOG_SETTINGS` 设置默认的配置。

```python
from paibox._logging import set_logs, DEFAULT_LOG_SETTINGS

set_logs(paibox=logging.INFO, backend=logging.DEBUG)
set_logs(**DEFAULT_LOG_SETTINGS)
```

## 更多

请参阅[官方文档](https://docs.pytest.org/en/stable/contents.html)
