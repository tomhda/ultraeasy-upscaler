"""Windows Job Object によるプロセスツリー管理。

目的 (oracle 必須):
  - 中継親 (npu_serve) が死ねば F/G ワーカーも死ぬ。
  - アプリ (ServeClient) が死ねば npu_serve＋F/G のツリー全体が消える。
  - `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` を使い、finally だけに依存しない。
  - Job ハンドルは子へ継承させない。
  - 所属登録前に子が実行開始しないよう、CREATE_SUSPENDED で起動 → Job 登録 → 再開。
  - Windows の入れ子 Job (アプリ所有 Job と中継親所有の子 Job) で構成する。

通常終了は QUIT→EOF、期限超過は TerminateProcess。
Linux 等の非 Windows 環境では Job は作らず ``subprocess.Popen`` に代替する
(単体試験・開発用。ツリー kill 保証はない)。
"""
from __future__ import annotations

import os
import subprocess
from typing import IO, Any

_WIN = os.name == "nt"

_CREATE_SUSPENDED = 0x00000004
_CREATE_NO_WINDOW = 0x08000000
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_STARTF_USESTDHANDLES = 0x00000100
_HANDLE_FLAG_INHERIT = 0x00000001
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JobObjectExtendedLimitInformation = 9
_WAIT_OBJECT_0 = 0x00000000
_WAIT_TIMEOUT = 0x00000102
_PROCESS_ALL = 0x001F0FFF
_DUPLICATE_SAME_ACCESS = 0x00000002


def job_supported() -> bool:
    """この環境で Job Object 管理が使えるか (Windows のみ True)。"""
    return _WIN


def _kernel32():  # pragma: no cover - Windows 専用
    import ctypes

    return ctypes.windll.kernel32


class JobHandle:
    """KILL_ON_JOB_CLOSE の Job。close (全ハンドル解放) で所属プロセスを kill。

    子プロセスへ継承させない (CreateJobObject の既定で非継承)。
    """

    def __init__(self) -> None:
        if not _WIN:  # pragma: no cover - 非 Windows では未使用
            raise OSError("Job Object is only available on Windows")
        import ctypes

        kernel32 = _kernel32()
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._handle = kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise OSError("CreateJobObjectW failed")
        self._kernel32 = kernel32
        self._closed = False

        # JOBOBJECT_EXTENDED_LIMIT_INFORMATION: 先頭の
        # JOBOBJECT_BASIC_LIMIT_INFORMATION (LimitFlags・他 8 フィールド) の後に
        # IoInfo・ProcessMemoryLimit 等が続く。LimitFlags 以外は 0 でよい。
        class _Basic(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class _Extended(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _Basic),
                ("IoInfo", ctypes.c_byte * 48),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        info = _Extended()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        kernel32.SetInformationJobObject.restype = ctypes.c_bool
        kernel32.SetInformationJobObject.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        ok = kernel32.SetInformationJobObject(
            self._handle,
            _JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            self.close()
            raise OSError("SetInformationJobObject(KILL_ON_JOB_CLOSE) failed")

    @property
    def handle(self) -> int:
        return self._handle

    def assign(self, process_handle: int) -> None:
        import ctypes

        self._kernel32.AssignProcessToJobObject.restype = ctypes.c_bool
        self._kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        if not self._kernel32.AssignProcessToJobObject(self._handle, process_handle):
            raise OSError("AssignProcessToJobObject failed")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._kernel32.CloseHandle(self._handle)
            except Exception:
                pass

    def __enter__(self) -> "JobHandle":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - GC 経路
        try:
            self.close()
        except Exception:
            pass


class JobChildProcess:
    """spawn_in_job が返す Popen 互換 (最小限) ラッパー。

    WorkerConn / ServeClient が使う API (stdin/stdout/stderr、pid、
    poll/wait/terminate/kill/returncode) を Popen と同じ形で提供する。
    Job ハンドルはこのオブジェクトが所有し、子の生存期間中保持する。
    """

    def __init__(
        self,
        pid: int,
        process_handle: int,
        stdin: IO[bytes],
        stdout: IO[bytes],
        stderr: IO[bytes],
        job: JobHandle,
        argv: list[str],
    ) -> None:
        self.pid = pid
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.returncode: int | None = None
        self._process_handle = process_handle
        self._job = job
        self._argv = argv
        self._closed = False

    def _query_exit(self) -> int | None:  # pragma: no cover - Windows 専用
        import ctypes

        kernel32 = _kernel32()
        code = ctypes.c_uint32()
        kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        if not kernel32.GetExitCodeProcess(self._process_handle, ctypes.byref(code)):
            return None
        if code.value == 259:  # STILL_ACTIVE
            return None
        return int(code.value) if code.value < 0x80000000 else -int(0x100000000 - code.value)

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if not _WIN:  # pragma: no cover
            return None
        self.returncode = self._query_exit()
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:  # pragma: no cover - Windows 専用
        import ctypes

        if self.returncode is not None:
            return self.returncode
        kernel32 = _kernel32()
        ms = 0xFFFFFFFF if timeout is None else max(0, int(timeout * 1000))
        ret = kernel32.WaitForSingleObject(self._process_handle, ms)
        if ret == _WAIT_TIMEOUT:
            raise subprocess.TimeoutExpired(self._argv, timeout)
        if ret != _WAIT_OBJECT_0:
            raise OSError(f"WaitForSingleObject failed: {ret}")
        self.returncode = self._query_exit()
        if self.returncode is None:
            raise OSError("process signaled but exit code unavailable")
        return self.returncode

    def terminate(self) -> None:  # pragma: no cover - Windows 専用
        if self.poll() is not None:
            return
        _terminate_handle(self._process_handle)

    def kill(self) -> None:  # pragma: no cover - Windows 専用
        if self.poll() is not None:
            return
        _terminate_handle(self._process_handle)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for stream in (self.stdin, self.stdout, self.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        if _WIN:  # pragma: no cover - Windows 専用
            try:
                _kernel32().CloseHandle(self._process_handle)
            except Exception:
                pass
        try:
            self._job.close()
        except Exception:
            pass

    def __del__(self) -> None:  # pragma: no cover - GC 経路
        try:
            self.close()
        except Exception:
            pass


def _terminate_handle(process_handle: int) -> None:  # pragma: no cover - Windows 専用
    import ctypes

    kernel32 = _kernel32()
    kernel32.TerminateProcess.restype = ctypes.c_bool
    kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel32.TerminateProcess(process_handle, 1)


def terminate_pid(pid: int) -> None:
    """PID 指定の強制終了 (障害注入後の回収用。Windows のみ実効)。"""
    if not _WIN:  # pragma: no cover - 非 Windows では未使用
        raise OSError("terminate_pid is only available on Windows")
    import ctypes

    kernel32 = _kernel32()  # pragma: no cover - Windows 専用
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
    handle = kernel32.OpenProcess(_PROCESS_ALL, False, pid)
    if not handle:
        return
    try:
        _terminate_handle(handle)
    finally:
        kernel32.CloseHandle(handle)


def _env_block(env: dict[str, str]) -> bytes:
    # CREATE_UNICODE_ENVIRONMENT 用の NUL 区切り・ソート済みブロック。
    items = sorted(env.items())
    return ("\0".join(f"{key}={value}" for key, value in items) + "\0\0").encode("utf-16-le")


def spawn_in_job(
    argv: list[str] | tuple[str, ...],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> JobChildProcess | subprocess.Popen:
    """子を Job に所属させて起動する。

    Windows では CREATE_SUSPENDED で起動 → Job 登録 → 再開する
    (登録前に子が実行開始しない)。stdin/stdout/stderr は全てパイプ
    (bufsize=0 相当の無バッファ) で返す。Job ハンドルは子へ継承させない。
    非 Windows では ``subprocess.Popen`` に代替する。
    """
    args = [os.fspath(arg) for arg in argv]
    if not _WIN:
        return subprocess.Popen(
            args,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
        )
    return _spawn_suspended(args, cwd=cwd, env=env)  # pragma: no cover - Windows 専用


def _spawn_suspended(  # pragma: no cover - Windows 専用
    args: list[str], *, cwd: str | None, env: dict[str, str] | None
) -> JobChildProcess:
    import ctypes
    import msvcrt

    kernel32 = _kernel32()

    job = JobHandle()

    # パイプ 3 本。子側端だけ継承可能にし、親側端は非継承化する。
    class _SA(ctypes.Structure):
        _fields_ = [
            ("nLength", ctypes.c_uint32),
            ("lpSecurityDescriptor", ctypes.c_void_p),
            ("bInheritHandle", ctypes.c_bool),
        ]

    sa = _SA(ctypes.sizeof(_SA), None, True)
    kernel32.CreatePipe.restype = ctypes.c_bool
    kernel32.CreatePipe.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(_SA),
        ctypes.c_uint32,
    ]
    kernel32.SetHandleInformation.restype = ctypes.c_bool
    kernel32.SetHandleInformation.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]

    def _pipe() -> tuple[int, int]:
        h_read = ctypes.c_void_p()
        h_write = ctypes.c_void_p()
        if not kernel32.CreatePipe(ctypes.byref(h_read), ctypes.byref(h_write), ctypes.byref(sa), 0):
            raise OSError("CreatePipe failed")
        return h_read.value, h_write.value

    stdin_r, stdin_w = _pipe()
    stdout_r, stdout_w = _pipe()
    stderr_r, stderr_w = _pipe()
    # 親側端を非継承化 (子へ漏らさない)。
    kernel32.SetHandleInformation(stdin_w, _HANDLE_FLAG_INHERIT, 0)
    kernel32.SetHandleInformation(stdout_r, _HANDLE_FLAG_INHERIT, 0)
    kernel32.SetHandleInformation(stderr_r, _HANDLE_FLAG_INHERIT, 0)

    class _SI(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("lpReserved", ctypes.c_void_p),
            ("lpDesktop", ctypes.c_void_p),
            ("lpTitle", ctypes.c_void_p),
            ("dwX", ctypes.c_uint32),
            ("dwY", ctypes.c_uint32),
            ("dwXSize", ctypes.c_uint32),
            ("dwYSize", ctypes.c_uint32),
            ("dwXCountChars", ctypes.c_uint32),
            ("dwYCountChars", ctypes.c_uint32),
            ("dwFillAttribute", ctypes.c_uint32),
            ("dwFlags", ctypes.c_uint32),
            ("wShowWindow", ctypes.c_uint16),
            ("cbReserved2", ctypes.c_uint16),
            ("lpReserved2", ctypes.c_void_p),
            ("hStdInput", ctypes.c_void_p),
            ("hStdOutput", ctypes.c_void_p),
            ("hStdError", ctypes.c_void_p),
        ]

    class _PI(ctypes.Structure):
        _fields_ = [
            ("hProcess", ctypes.c_void_p),
            ("hThread", ctypes.c_void_p),
            ("dwProcessId", ctypes.c_uint32),
            ("dwThreadId", ctypes.c_uint32),
        ]

    si = _SI()
    si.cb = ctypes.sizeof(_SI)
    si.dwFlags = _STARTF_USESTDHANDLES
    si.hStdInput = stdin_r
    si.hStdOutput = stdout_w
    si.hStdError = stderr_w
    pi = _PI()

    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    env_block = _env_block({str(k): str(v) for k, v in full_env.items()})
    cmdline = subprocess.list2cmdline(args)

    kernel32.CreateProcessW.restype = ctypes.c_bool
    kernel32.CreateProcessW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_bool,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(_SI),
        ctypes.POINTER(_PI),
    ]
    try:
        ok = kernel32.CreateProcessW(
            None,
            cmdline,
            None,
            None,
            True,
            _CREATE_SUSPENDED | _CREATE_NO_WINDOW | _CREATE_UNICODE_ENVIRONMENT,
            env_block,
            cwd,
            ctypes.byref(si),
            ctypes.byref(pi),
        )
        if not ok:
            raise OSError(f"CreateProcessW failed: {args[0]}")
        # 登録してから再開 (この順序が必須)。
        job.assign(pi.hProcess)
        kernel32.ResumeThread.argtypes = [ctypes.c_void_p]
        kernel32.ResumeThread.restype = ctypes.c_uint32
        kernel32.ResumeThread(pi.hThread)
    except BaseException:
        job.close()
        raise
    finally:
        # 子側端・スレッドハンドルは親では不要。
        for handle in (stdin_r, stdout_w, stderr_w):
            try:
                kernel32.CloseHandle(handle)
            except Exception:
                pass
        try:
            kernel32.CloseHandle(pi.hThread)
        except Exception:
            pass

    def _fdopen(handle: int, mode: str) -> IO[bytes]:
        flags = os.O_BINARY
        if "w" in mode or "+" in mode and "r" not in mode:
            flags |= os.O_WRONLY
        else:
            flags |= os.O_RDONLY
        fd = msvcrt.open_osfhandle(handle, flags)
        return os.fdopen(fd, mode + "b" if not mode.endswith("b") else mode, buffering=0)

    proc = JobChildProcess(
        pid=pi.dwProcessId,
        process_handle=pi.hProcess,
        stdin=_fdopen(stdin_w, "w"),
        stdout=_fdopen(stdout_r, "r"),
        stderr=_fdopen(stderr_r, "r"),
        job=job,
        argv=list(args),
    )
    return proc
