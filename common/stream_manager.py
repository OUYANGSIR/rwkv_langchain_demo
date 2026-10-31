import asyncio, threading, itertools
from typing import Any, Callable, Optional
import concurrent.futures

from langchain_core.messages import BaseMessage


class StreamManager:
    """后台跑一个 asyncio 事件循环,同时管理多个流式任务;可按 id 单独取消。"""
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self._ids = itertools.count(1)
        self._tasks = {}                      # id -> (future, chunks)

    def start(self, runnable: Any, inputs: str|BaseMessage, on_text: Optional[Callable] = None, config=None) -> int:
        """
        启动一个流式推理任务（立即返回，不阻塞）。
        :param runnable: llm 或链（如 prompt | llm | StrOutputParser()）
        :param inputs: 传 llm 时为提示词字符串；传链时为填模板的字典，如 {"question": "你好"}
        :param on_text: 可选回调 on_text(任务id, 文本片段)，在后台线程中调用
        :param config: 可选的 LangChain RunnableConfig（如 callbacks、tags）
        :return: 任务 id，用于 stop / text / running / wait
        """
        if inputs is BaseMessage:
            pass
        tid = next(self._ids)
        chunks = []
        async def work():
            async for piece in runnable.astream(inputs, config=config):
                text = piece if isinstance(piece, str) else getattr(piece, "content", str(piece))
                chunks.append(text)
                if on_text:
                    on_text(tid, text)
        fut = asyncio.run_coroutine_threadsafe(work(), self.loop)
        self._tasks[tid] = (fut, chunks)
        return tid

    def stop(self, tid: int) -> str:
        """
        取消任务并断开连接，返回到目前为止已生成的文本。
        对已结束的任务调用无副作用；id 不存在会抛 KeyError。
        """
        fut, chunks = self._tasks[tid]
        fut.cancel()
        return "".join(chunks)

    def stop_all(self) -> None:
        """取消所有任务（已结束的不受影响）。"""
        for tid in list(self._tasks):
            self.stop(tid)

    def text(self, tid: int) -> str:
        """返回目前累积的文本，不阻塞。任务运行中调用得到的是中间结果。"""
        return "".join(self._tasks[tid][1])

    def is_running(self, tid: int) -> bool:
        """任务是否还在运行。正常结束、被取消、出错之后都返回 False。"""
        return not self._tasks[tid][0].done()

    def error(self, tid: int) -> Optional[BaseException]:
        """
        返回任务的异常对象；任务还在运行、被取消或正常结束时返回 None。
        不会抛出异常，适合轮询时检查。
        """
        fut = self._tasks[tid][0]
        if not fut.done() or fut.cancelled():
            return None
        return fut.exception()

    def wait(self, tid: int, timeout: Optional[float] = None) -> str:
        """
        阻塞等待任务结束，返回完整文本。
        - 超时：返回当前文本，任务继续在后台运行（可再用 running 判断）。
        - 被 stop 取消：返回已生成的文本。
        - 任务出错（连接失败、服务端报错等）：重新抛出该异常。
          此时已生成的部分文本仍可用 text(tid) 取得。
        """
        try:
            self._tasks[tid][0].result(timeout)
        except (concurrent.futures.TimeoutError,
                concurrent.futures.CancelledError,
                asyncio.CancelledError):
            pass
        # 处理停止词
        for end_words in ["\n\nUser:", "</tool_call>\n"]:
            if self.text(tid).endswith(end_words):
                return self.text(tid)[:-len(end_words)].strip()
        return self.text(tid)

    def clear(self, tid: Optional[int] = None) -> Optional[str]:
        """
        清理任务记录。
        - 不传 tid：删除所有已结束的任务（运行中的不动）。
        - 传 tid：删除该任务；如果还在运行，会先取消。返回其文本。
        """
        if tid is not None:
            fut, chunks = self._tasks.pop(tid)
            if not fut.done():
                fut.cancel()
            return "".join(chunks)
        for t in [t for t, (fut, _) in list(self._tasks.items()) if fut.done()]:
            self._tasks.pop(t, None)
        return None